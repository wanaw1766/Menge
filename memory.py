"""
memory.py — SQLite-backed memory manager with military-grade duplicate prevention.
Includes locking methods for daily/weekly calendars with delete support.
"""

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple

import aiosqlite
import imagehash
from PIL import Image
import io

log = logging.getLogger("memory")

class MemoryManager:
    def __init__(self, db_path: Optional[str] = None, ttl_days: int = 30):
        if db_path is None:
            db_path = os.environ.get("MEMORY_DB_PATH", "data/memory.db")
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._db_path = db_path
        self._ttl_days = ttl_days
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self):
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._create_tables()
        await self._db.commit()
        await self._cleanup_old_hashes()
        log.info(f"✅ MemoryManager ready — db={self._db_path} | ttl={self._ttl_days}d")

    async def close(self):
        if self._db:
            await self._db.close()

    async def _create_tables(self):
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS content_hashes (
                hash        TEXT PRIMARY KEY,
                source      TEXT,
                seen_at     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS image_hashes (
                perceptual_hash TEXT PRIMARY KEY,
                source_channel  TEXT,
                seen_at         TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS recent_posts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_text TEXT NOT NULL,
                post_text   TEXT NOT NULL,
                image_phash TEXT,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS posted_messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source_channel  TEXT NOT NULL,
                source_msg_id   INTEGER NOT NULL,
                dest_msg_id     INTEGER NOT NULL,
                content_hash    TEXT NOT NULL,
                engine          TEXT,
                confidence      REAL,
                formatted_text  TEXT,
                posted_at       TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS daily_briefings (
                date_str    TEXT PRIMARY KEY,
                msg_id      INTEGER NOT NULL,
                events_json TEXT,
                posted_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS weekly_calendar (
                week_key    TEXT PRIMARY KEY,
                posted_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reminders (
                event_key   TEXT PRIMARY KEY,
                sent_at     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reminder_counts (
                date_str    TEXT PRIMARY KEY,
                count       INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS channel_offsets (
                channel     TEXT PRIMARY KEY,
                last_msg_id INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS kv_store (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL
            );
        """)

    @staticmethod
    def compute_phash(image_data: bytes, hash_size: int = 16) -> str:
        try:
            img = Image.open(io.BytesIO(image_data))
            phash = imagehash.phash(img, hash_size=hash_size)
            return str(phash)
        except Exception:
            return ""

    @staticmethod
    def hamming_distance(phash1: str, phash2: str) -> int:
        if not phash1 or not phash2 or len(phash1) != len(phash2):
            return 999
        bin1 = bin(int(phash1, 16))[2:].zfill(len(phash1)*4)
        bin2 = bin(int(phash2, 16))[2:].zfill(len(phash2)*4)
        return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))

    @staticmethod
    def hash_combined(text: str, image_data: Optional[bytes]) -> str:
        h = hashlib.sha256()
        if text:
            h.update(text.encode("utf-8", errors="replace"))
        if image_data:
            h.update(image_data[:4096])
        return h.hexdigest()

    async def is_duplicate(self, content_hash: str) -> bool:
        async with self._db.execute(
            "SELECT 1 FROM content_hashes WHERE hash=?", (content_hash,)
        ) as cur:
            return await cur.fetchone() is not None

    async def mark_seen(self, content_hash: str, source: str = ""):
        await self._db.execute(
            "INSERT OR IGNORE INTO content_hashes (hash, source, seen_at) VALUES (?, ?, ?)",
            (content_hash, source, _utcnow()),
        )
        await self._db.commit()

    async def is_image_duplicate(self, phash: str, max_distance: int = 3) -> bool:
        if not phash:
            return False
        async with self._db.execute("SELECT perceptual_hash FROM image_hashes") as cur:
            rows = await cur.fetchall()
        for row in rows:
            stored = row["perceptual_hash"]
            if stored and self.hamming_distance(phash, stored) <= max_distance:
                return True
        return False

    async def mark_image_seen(self, phash: str, source: str = ""):
        if phash:
            await self._db.execute(
                "INSERT OR IGNORE INTO image_hashes (perceptual_hash, source_channel, seen_at) VALUES (?, ?, ?)",
                (phash, source, _utcnow()),
            )
            await self._db.commit()

    async def store_recent_post(self, source_text: str, post_text: str, image_phash: Optional[str] = None):
        await self._db.execute(
            "INSERT INTO recent_posts (source_text, post_text, image_phash, created_at) VALUES (?, ?, ?, ?)",
            (source_text[:1000], post_text[:1000], image_phash, _utcnow()),
        )
        await self._db.execute("""
            DELETE FROM recent_posts
            WHERE id NOT IN (SELECT id FROM recent_posts ORDER BY id DESC LIMIT 200)
        """)
        await self._db.commit()

    async def get_recent_posts(self, limit: int = 150) -> List[Tuple[str, Optional[str], str]]:
        async with self._db.execute(
            "SELECT source_text, image_phash, created_at FROM recent_posts ORDER BY id DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
        return [(row["source_text"], row["image_phash"], row["created_at"]) for row in rows]

    async def get_recent_post_texts(self, limit: int = 150) -> List[str]:
        posts = await self.get_recent_posts(limit)
        return [text for text, _, _ in posts]

    async def log_posted(self, source_channel: str, source_msg_id: int, dest_msg_id: int,
                         content_hash: str, ai_verdict: dict, formatted_text: str):
        await self._db.execute(
            """INSERT INTO posted_messages
               (source_channel, source_msg_id, dest_msg_id, content_hash,
                engine, confidence, formatted_text, posted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (source_channel, source_msg_id, dest_msg_id, content_hash,
             ai_verdict.get("engine", ""), ai_verdict.get("confidence", 0.0),
             formatted_text, _utcnow()),
        )
        await self._db.commit()

    async def has_daily_briefing(self, date_str: str) -> bool:
        async with self._db.execute(
            "SELECT 1 FROM daily_briefings WHERE date_str=?", (date_str,)
        ) as cur:
            return await cur.fetchone() is not None

    async def save_daily_briefing(self, date_str: str, msg_id: int, events: list):
        await self._db.execute(
            "INSERT OR REPLACE INTO daily_briefings (date_str, msg_id, events_json, posted_at) VALUES (?, ?, ?, ?)",
            (date_str, msg_id, json.dumps(events, ensure_ascii=False), _utcnow()),
        )
        await self._db.commit()

    async def delete_daily_briefing(self, date_str: str):
        """
        Remove the daily briefing lock for date_str.
        Called when AI rejects the image or send fails, so a retry is allowed.
        """
        await self._db.execute(
            "DELETE FROM daily_briefings WHERE date_str=?", (date_str,)
        )
        await self._db.commit()

    async def get_daily_briefing_msg_id(self, date_str: str) -> Optional[int]:
        async with self._db.execute(
            "SELECT msg_id FROM daily_briefings WHERE date_str=?", (date_str,)
        ) as cur:
            row = await cur.fetchone()
        return row["msg_id"] if row else None

    async def has_weekly_posted(self, week_key: str) -> bool:
        async with self._db.execute(
            "SELECT 1 FROM weekly_calendar WHERE week_key=?", (week_key,)
        ) as cur:
            return await cur.fetchone() is not None

    async def save_weekly_posted(self, week_key: str):
        await self._db.execute(
            "INSERT OR REPLACE INTO weekly_calendar (week_key, posted_at) VALUES (?, ?)",
            (week_key, _utcnow()),
        )
        await self._db.commit()

    async def delete_weekly_posted(self, week_key: str):
        """
        Remove the weekly calendar lock for week_key.
        Called when AI rejects the image or send fails, so a retry is allowed.
        """
        await self._db.execute(
            "DELETE FROM weekly_calendar WHERE week_key=?", (week_key,)
        )
        await self._db.commit()

    async def has_reminder_been_sent(self, event_key: str) -> bool:
        async with self._db.execute(
            "SELECT 1 FROM reminders WHERE event_key=?", (event_key,)
        ) as cur:
            return await cur.fetchone() is not None

    async def mark_reminder_sent(self, event_key: str):
        await self._db.execute(
            "INSERT OR IGNORE INTO reminders (event_key, sent_at) VALUES (?, ?)",
            (event_key, _utcnow()),
        )
        await self._db.commit()

    async def get_reminder_count_today(self, date_str: str) -> int:
        async with self._db.execute(
            "SELECT count FROM reminder_counts WHERE date_str=?", (date_str,)
        ) as cur:
            row = await cur.fetchone()
        return row["count"] if row else 0

    async def increment_reminder_count(self, date_str: str):
        await self._db.execute(
            """INSERT INTO reminder_counts (date_str, count) VALUES (?, 1)
               ON CONFLICT(date_str) DO UPDATE SET count = count + 1""",
            (date_str,),
        )
        await self._db.commit()

    async def get_and_increment_motivational_index(self) -> int:
        async with self._db.execute(
            "SELECT value FROM kv_store WHERE key='motivational_index'"
        ) as cur:
            row = await cur.fetchone()
        current = int(row["value"]) if row else 0
        next_val = current + 1
        await self._db.execute(
            "INSERT OR REPLACE INTO kv_store (key, value) VALUES ('motivational_index', ?)",
            (str(next_val),),
        )
        await self._db.commit()
        return current

    async def get_last_msg_id(self, channel: str) -> int:
        async with self._db.execute(
            "SELECT last_msg_id FROM channel_offsets WHERE channel=?", (channel,)
        ) as cur:
            row = await cur.fetchone()
        return row["last_msg_id"] if row else 0

    async def set_last_msg_id(self, channel: str, msg_id: int):
        await self._db.execute(
            """INSERT INTO channel_offsets (channel, last_msg_id) VALUES (?, ?)
               ON CONFLICT(channel) DO UPDATE SET last_msg_id = ?""",
            (channel, msg_id, msg_id),
        )
        await self._db.commit()

    async def _cleanup_old_hashes(self):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self._ttl_days)).isoformat()
        await self._db.execute(
            "DELETE FROM content_hashes WHERE seen_at < ?", (cutoff,)
        )
        await self._db.execute(
            "DELETE FROM image_hashes WHERE seen_at < ?", (cutoff,)
        )
        # Delete recent posts older than 2 days
        cutoff_2d = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        await self._db.execute("DELETE FROM recent_posts WHERE created_at < ?", (cutoff_2d,))
        await self._db.commit()

    async def stats(self) -> dict:
        cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        async with self._db.execute("SELECT COUNT(*) as n FROM content_hashes") as cur:
            hashes = (await cur.fetchone())["n"]
        async with self._db.execute(
            "SELECT COUNT(*) as n FROM posted_messages WHERE posted_at > ?", (cutoff_24h,)
        ) as cur:
            posted_24h = (await cur.fetchone())["n"]
        return {
            "tracked_hashes": hashes,
            "posted_last_24h": posted_24h,
        }

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
