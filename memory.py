"""
memory.py — Supabase PostgreSQL primary + SQLite fallback.
- Same API as original — scraper/ai_engine see no difference.
- Supabase: encrypted, secure, survives Railway redeploys.
- SQLite: automatic fallback if Supabase unavailable.
- VIP posts (FOMC/NFP/CPI/GDP/PCE/Powell) kept 7 days.
- All other posts kept 2 days.
"""

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

# ── Supabase connection string ────────────────────────────────────────────────
# Set SUPABASE_DB in your .env to use Supabase
# Falls back to SQLite if not set or connection fails
SUPABASE_DB = os.environ.get(
    "SUPABASE_DB",
    "postgresql://postgres:Husam%40252536@db.kvwnirniwzmjtorgoofj.supabase.co:5432/postgres"
)

# VIP keywords — kept 7 days in similarity window
_VIP_KEYWORDS = [
    "fomc", "federal funds", "interest rate",
    "non-farm", "nfp", "payroll",
    "cpi", "consumer price",
    "pce", "core pce",
    "gdp", "advance gdp",
    "fed chair", "powell",
]


class MemoryManager:
    def __init__(self, db_path: Optional[str] = None, ttl_days: int = 30):
        if db_path is None:
            db_path = os.environ.get("MEMORY_DB_PATH", "data/memory.db")
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._db_path  = db_path
        self._ttl_days = ttl_days
        self._db: Optional[aiosqlite.Connection] = None
        self._pg       = None
        self._use_pg   = False

    # ── Init ──────────────────────────────────────────────────────────────────

    async def init(self):
        # Try Supabase
        await self._init_supabase()
        # Always init SQLite as fallback
        await self._init_sqlite()
        await self._cleanup_old_hashes()
        mode = "Supabase + SQLite fallback" if self._use_pg else "SQLite only"
        log.info(f"✅ MemoryManager ready — mode={mode} | ttl={self._ttl_days}d")

    async def _init_supabase(self):
        try:
            import asyncpg
            self._pg = await asyncpg.create_pool(
                SUPABASE_DB,
                min_size=1, max_size=5,
                command_timeout=10,
                ssl="require",
            )
            await self._create_pg_tables()
            self._use_pg = True
            log.info("✅ Supabase connected")
        except Exception as exc:
            log.warning(f"⚠️ Supabase unavailable ({exc}) — SQLite only")
            self._use_pg = False
            self._pg = None

    async def _init_sqlite(self):
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._create_sqlite_tables()
        await self._db.commit()

    async def close(self):
        if self._pg:
            await self._pg.close()
        if self._db:
            await self._db.close()

    # ── Table creation ────────────────────────────────────────────────────────

    async def _create_pg_tables(self):
        async with self._pg.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS content_hashes (
                    hash TEXT PRIMARY KEY, source TEXT,
                    seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
                CREATE TABLE IF NOT EXISTS image_hashes (
                    perceptual_hash TEXT PRIMARY KEY, source_channel TEXT,
                    seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
                CREATE TABLE IF NOT EXISTS recent_posts (
                    id SERIAL PRIMARY KEY, source_text TEXT NOT NULL,
                    post_text TEXT NOT NULL, image_phash TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
                CREATE TABLE IF NOT EXISTS posted_messages (
                    id SERIAL PRIMARY KEY, source_channel TEXT NOT NULL,
                    source_msg_id BIGINT NOT NULL, dest_msg_id BIGINT NOT NULL,
                    content_hash TEXT NOT NULL, engine TEXT,
                    confidence REAL, formatted_text TEXT,
                    posted_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
                CREATE TABLE IF NOT EXISTS daily_briefings (
                    date_str TEXT PRIMARY KEY, msg_id BIGINT NOT NULL,
                    events_json TEXT,
                    posted_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
                CREATE TABLE IF NOT EXISTS weekly_calendar (
                    week_key TEXT PRIMARY KEY,
                    posted_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
                CREATE TABLE IF NOT EXISTS reminders (
                    event_key TEXT PRIMARY KEY,
                    sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
                CREATE TABLE IF NOT EXISTS reminder_counts (
                    date_str TEXT PRIMARY KEY,
                    count INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS channel_offsets (
                    channel TEXT PRIMARY KEY,
                    last_msg_id BIGINT NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS kv_store (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    async def _create_sqlite_tables(self):
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS content_hashes (
                hash TEXT PRIMARY KEY, source TEXT, seen_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS image_hashes (
                perceptual_hash TEXT PRIMARY KEY, source_channel TEXT,
                seen_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS recent_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_text TEXT NOT NULL, post_text TEXT NOT NULL,
                image_phash TEXT, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS posted_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_channel TEXT NOT NULL, source_msg_id INTEGER NOT NULL,
                dest_msg_id INTEGER NOT NULL, content_hash TEXT NOT NULL,
                engine TEXT, confidence REAL, formatted_text TEXT,
                posted_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS daily_briefings (
                date_str TEXT PRIMARY KEY, msg_id INTEGER NOT NULL,
                events_json TEXT, posted_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS weekly_calendar (
                week_key TEXT PRIMARY KEY, posted_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS reminders (
                event_key TEXT PRIMARY KEY, sent_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS reminder_counts (
                date_str TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS channel_offsets (
                channel TEXT PRIMARY KEY, last_msg_id INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _pg_execute(self, query: str, *args) -> bool:
        if not self._use_pg or not self._pg:
            return False
        try:
            async with self._pg.acquire() as conn:
                await conn.execute(query, *args)
            return True
        except Exception as exc:
            log.warning(f"Supabase write failed ({exc}) — SQLite fallback")
            self._use_pg = False
            return False

    async def _pg_fetchrow(self, query: str, *args):
        if not self._use_pg or not self._pg:
            return None
        try:
            async with self._pg.acquire() as conn:
                return await conn.fetchrow(query, *args)
        except Exception as exc:
            log.warning(f"Supabase read failed ({exc}) — SQLite fallback")
            self._use_pg = False
            return None

    async def _pg_fetch(self, query: str, *args):
        if not self._use_pg or not self._pg:
            return None
        try:
            async with self._pg.acquire() as conn:
                return await conn.fetch(query, *args)
        except Exception as exc:
            log.warning(f"Supabase fetch failed ({exc}) — SQLite fallback")
            self._use_pg = False
            return None

    # ── Static ────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_phash(image_data: bytes, hash_size: int = 16) -> str:
        try:
            img = Image.open(io.BytesIO(image_data))
            return str(imagehash.phash(img, hash_size=hash_size))
        except Exception:
            return ""

    @staticmethod
    def hamming_distance(phash1: str, phash2: str) -> int:
        if not phash1 or not phash2 or len(phash1) != len(phash2):
            return 999
        bin1 = bin(int(phash1, 16))[2:].zfill(len(phash1) * 4)
        bin2 = bin(int(phash2, 16))[2:].zfill(len(phash2) * 4)
        return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))

    @staticmethod
    def hash_combined(text: str, image_data: Optional[bytes]) -> str:
        h = hashlib.sha256()
        if text:
            h.update(text.encode("utf-8", errors="replace"))
        if image_data:
            h.update(image_data[:4096])
        return h.hexdigest()

    # ── Content hashes ────────────────────────────────────────────────────────

    async def is_duplicate(self, content_hash: str) -> bool:
        row = await self._pg_fetchrow(
            "SELECT 1 FROM content_hashes WHERE hash=$1", content_hash)
        if row is not None:
            return True
        if self._use_pg:
            return False
        async with self._db.execute(
            "SELECT 1 FROM content_hashes WHERE hash=?", (content_hash,)) as cur:
            return await cur.fetchone() is not None

    async def mark_seen(self, content_hash: str, source: str = ""):
        ok = await self._pg_execute(
            "INSERT INTO content_hashes (hash,source,seen_at) VALUES ($1,$2,NOW()) ON CONFLICT DO NOTHING",
            content_hash, source)
        if not ok:
            await self._db.execute(
                "INSERT OR IGNORE INTO content_hashes (hash,source,seen_at) VALUES (?,?,?)",
                (content_hash, source, _utcnow()))
            await self._db.commit()

    # ── Image hashes ──────────────────────────────────────────────────────────

    async def is_image_duplicate(self, phash: str, max_distance: int = 3) -> bool:
        if not phash:
            return False
        rows = await self._pg_fetch("SELECT perceptual_hash FROM image_hashes")
        if rows is None:
            async with self._db.execute("SELECT perceptual_hash FROM image_hashes") as cur:
                rows = await cur.fetchall()
        for row in rows:
            stored = row["perceptual_hash"]
            if stored and self.hamming_distance(phash, stored) <= max_distance:
                return True
        return False

    async def mark_image_seen(self, phash: str, source: str = ""):
        if not phash:
            return
        ok = await self._pg_execute(
            "INSERT INTO image_hashes (perceptual_hash,source_channel,seen_at) VALUES ($1,$2,NOW()) ON CONFLICT DO NOTHING",
            phash, source)
        if not ok:
            await self._db.execute(
                "INSERT OR IGNORE INTO image_hashes (perceptual_hash,source_channel,seen_at) VALUES (?,?,?)",
                (phash, source, _utcnow()))
            await self._db.commit()

    # ── Recent posts ──────────────────────────────────────────────────────────

    async def store_recent_post(self, source_text: str, post_text: str,
                                image_phash: Optional[str] = None):
        ok = await self._pg_execute(
            "INSERT INTO recent_posts (source_text,post_text,image_phash,created_at) VALUES ($1,$2,$3,NOW())",
            source_text[:1000], post_text[:1000], image_phash)
        if not ok:
            await self._db.execute(
                "INSERT INTO recent_posts (source_text,post_text,image_phash,created_at) VALUES (?,?,?,?)",
                (source_text[:1000], post_text[:1000], image_phash, _utcnow()))
            await self._db.execute(
                "DELETE FROM recent_posts WHERE id NOT IN "
                "(SELECT id FROM recent_posts ORDER BY id DESC LIMIT 500)")
            await self._db.commit()

    async def get_recent_posts(self, limit: int = 150) -> List[Tuple[str, Optional[str], str]]:
        rows = await self._pg_fetch(
            "SELECT source_text,image_phash,created_at FROM recent_posts ORDER BY id DESC LIMIT $1",
            limit)
        if rows is None:
            async with self._db.execute(
                "SELECT source_text,image_phash,created_at FROM recent_posts ORDER BY id DESC LIMIT ?",
                (limit,)) as cur:
                rows = await cur.fetchall()
        return [(row["source_text"], row["image_phash"], str(row["created_at"])) for row in rows]

    async def get_recent_post_texts(self, limit: int = 150) -> List[str]:
        return [t for t, _, _ in await self.get_recent_posts(limit)]

    async def log_posted(self, source_channel: str, source_msg_id: int, dest_msg_id: int,
                         content_hash: str, ai_verdict: dict, formatted_text: str):
        ok = await self._pg_execute(
            """INSERT INTO posted_messages
               (source_channel,source_msg_id,dest_msg_id,content_hash,engine,confidence,formatted_text,posted_at)
               VALUES ($1,$2,$3,$4,$5,$6,$7,NOW())""",
            source_channel, source_msg_id, dest_msg_id, content_hash,
            ai_verdict.get("engine", ""), ai_verdict.get("confidence", 0.0), formatted_text)
        if not ok:
            await self._db.execute(
                """INSERT INTO posted_messages
                   (source_channel,source_msg_id,dest_msg_id,content_hash,engine,confidence,formatted_text,posted_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (source_channel, source_msg_id, dest_msg_id, content_hash,
                 ai_verdict.get("engine", ""), ai_verdict.get("confidence", 0.0),
                 formatted_text, _utcnow()))
            await self._db.commit()

    # ── Daily briefing ────────────────────────────────────────────────────────

    async def has_daily_briefing(self, date_str: str) -> bool:
        row = await self._pg_fetchrow(
            "SELECT 1 FROM daily_briefings WHERE date_str=$1", date_str)
        if row is not None:
            return True
        if self._use_pg:
            return False
        async with self._db.execute(
            "SELECT 1 FROM daily_briefings WHERE date_str=?", (date_str,)) as cur:
            return await cur.fetchone() is not None

    async def save_daily_briefing(self, date_str: str, msg_id: int, events: list):
        ej = json.dumps(events, ensure_ascii=False)
        ok = await self._pg_execute(
            """INSERT INTO daily_briefings (date_str,msg_id,events_json,posted_at) VALUES ($1,$2,$3,NOW())
               ON CONFLICT (date_str) DO UPDATE SET msg_id=$2,events_json=$3,posted_at=NOW()""",
            date_str, msg_id, ej)
        if not ok:
            await self._db.execute(
                "INSERT OR REPLACE INTO daily_briefings (date_str,msg_id,events_json,posted_at) VALUES (?,?,?,?)",
                (date_str, msg_id, ej, _utcnow()))
            await self._db.commit()

    async def delete_daily_briefing(self, date_str: str):
        ok = await self._pg_execute(
            "DELETE FROM daily_briefings WHERE date_str=$1", date_str)
        if not ok:
            await self._db.execute(
                "DELETE FROM daily_briefings WHERE date_str=?", (date_str,))
            await self._db.commit()

    async def get_daily_briefing_msg_id(self, date_str: str) -> Optional[int]:
        row = await self._pg_fetchrow(
            "SELECT msg_id FROM daily_briefings WHERE date_str=$1", date_str)
        if row is not None:
            return row["msg_id"]
        if self._use_pg:
            return None
        async with self._db.execute(
            "SELECT msg_id FROM daily_briefings WHERE date_str=?", (date_str,)) as cur:
            row = await cur.fetchone()
        return row["msg_id"] if row else None

    # ── Weekly calendar ────────────────────────────────────────────────────────

    async def has_weekly_posted(self, week_key: str) -> bool:
        row = await self._pg_fetchrow(
            "SELECT 1 FROM weekly_calendar WHERE week_key=$1", week_key)
        if row is not None:
            return True
        if self._use_pg:
            return False
        async with self._db.execute(
            "SELECT 1 FROM weekly_calendar WHERE week_key=?", (week_key,)) as cur:
            return await cur.fetchone() is not None

    async def save_weekly_posted(self, week_key: str):
        ok = await self._pg_execute(
            "INSERT INTO weekly_calendar (week_key,posted_at) VALUES ($1,NOW()) ON CONFLICT DO NOTHING",
            week_key)
        if not ok:
            await self._db.execute(
                "INSERT OR REPLACE INTO weekly_calendar (week_key,posted_at) VALUES (?,?)",
                (week_key, _utcnow()))
            await self._db.commit()

    async def delete_weekly_posted(self, week_key: str):
        ok = await self._pg_execute(
            "DELETE FROM weekly_calendar WHERE week_key=$1", week_key)
        if not ok:
            await self._db.execute(
                "DELETE FROM weekly_calendar WHERE week_key=?", (week_key,))
            await self._db.commit()

    # ── Reminders ──────────────────────────────────────────────────────────────

    async def has_reminder_been_sent(self, event_key: str) -> bool:
        row = await self._pg_fetchrow(
            "SELECT 1 FROM reminders WHERE event_key=$1", event_key)
        if row is not None:
            return True
        if self._use_pg:
            return False
        async with self._db.execute(
            "SELECT 1 FROM reminders WHERE event_key=?", (event_key,)) as cur:
            return await cur.fetchone() is not None

    async def mark_reminder_sent(self, event_key: str):
        ok = await self._pg_execute(
            "INSERT INTO reminders (event_key,sent_at) VALUES ($1,NOW()) ON CONFLICT DO NOTHING",
            event_key)
        if not ok:
            await self._db.execute(
                "INSERT OR IGNORE INTO reminders (event_key,sent_at) VALUES (?,?)",
                (event_key, _utcnow()))
            await self._db.commit()

    async def get_reminder_count_today(self, date_str: str) -> int:
        row = await self._pg_fetchrow(
            "SELECT count FROM reminder_counts WHERE date_str=$1", date_str)
        if row is not None:
            return row["count"]
        if self._use_pg:
            return 0
        async with self._db.execute(
            "SELECT count FROM reminder_counts WHERE date_str=?", (date_str,)) as cur:
            row = await cur.fetchone()
        return row["count"] if row else 0

    async def increment_reminder_count(self, date_str: str):
        ok = await self._pg_execute(
            """INSERT INTO reminder_counts (date_str,count) VALUES ($1,1)
               ON CONFLICT (date_str) DO UPDATE SET count=reminder_counts.count+1""",
            date_str)
        if not ok:
            await self._db.execute(
                """INSERT INTO reminder_counts (date_str,count) VALUES (?,1)
                   ON CONFLICT(date_str) DO UPDATE SET count=count+1""",
                (date_str,))
            await self._db.commit()

    async def get_and_increment_motivational_index(self) -> int:
        row = await self._pg_fetchrow(
            "SELECT value FROM kv_store WHERE key='motivational_index'")
        if row is not None:
            current = int(row["value"])
            await self._pg_execute(
                "INSERT INTO kv_store (key,value) VALUES ('motivational_index',$1) "
                "ON CONFLICT (key) DO UPDATE SET value=$1", str(current + 1))
            return current
        if self._use_pg:
            await self._pg_execute(
                "INSERT INTO kv_store (key,value) VALUES ('motivational_index','1') ON CONFLICT DO NOTHING")
            return 0
        async with self._db.execute(
            "SELECT value FROM kv_store WHERE key='motivational_index'") as cur:
            row = await cur.fetchone()
        current = int(row["value"]) if row else 0
        await self._db.execute(
            "INSERT OR REPLACE INTO kv_store (key,value) VALUES ('motivational_index',?)",
            (str(current + 1),))
        await self._db.commit()
        return current

    # ── Channel offsets ────────────────────────────────────────────────────────

    async def get_last_msg_id(self, channel: str) -> int:
        row = await self._pg_fetchrow(
            "SELECT last_msg_id FROM channel_offsets WHERE channel=$1", channel)
        if row is not None:
            return row["last_msg_id"]
        if self._use_pg:
            return 0
        async with self._db.execute(
            "SELECT last_msg_id FROM channel_offsets WHERE channel=?", (channel,)) as cur:
            row = await cur.fetchone()
        return row["last_msg_id"] if row else 0

    async def set_last_msg_id(self, channel: str, msg_id: int):
        ok = await self._pg_execute(
            "INSERT INTO channel_offsets (channel,last_msg_id) VALUES ($1,$2) "
            "ON CONFLICT (channel) DO UPDATE SET last_msg_id=$2",
            channel, msg_id)
        if not ok:
            await self._db.execute(
                "INSERT INTO channel_offsets (channel,last_msg_id) VALUES (?,?) "
                "ON CONFLICT(channel) DO UPDATE SET last_msg_id=?",
                (channel, msg_id, msg_id))
            await self._db.commit()

    # ── Cleanup ────────────────────────────────────────────────────────────────

    async def _cleanup_old_hashes(self):
        cutoff_30d = (datetime.now(timezone.utc) - timedelta(days=self._ttl_days)).isoformat()
        cutoff_7d  = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        cutoff_2d  = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()

        non_vip_pg = " AND ".join(
            [f"source_text NOT ILIKE '%{kw}%'" for kw in _VIP_KEYWORDS])
        non_vip_sq = " AND ".join(
            [f"source_text NOT LIKE '%{kw}%'" for kw in _VIP_KEYWORDS])

        if self._use_pg and self._pg:
            try:
                async with self._pg.acquire() as conn:
                    await conn.execute(
                        "DELETE FROM content_hashes WHERE seen_at < $1", cutoff_30d)
                    await conn.execute(
                        "DELETE FROM image_hashes WHERE seen_at < $1", cutoff_30d)
                    await conn.execute(
                        "DELETE FROM recent_posts WHERE created_at < $1", cutoff_7d)
                    await conn.execute(
                        f"DELETE FROM recent_posts WHERE created_at < $1 AND {non_vip_pg}",
                        cutoff_2d)
                log.info("🧹 Supabase cleanup done — VIP 7d, others 2d")
                return
            except Exception as exc:
                log.warning(f"Supabase cleanup failed: {exc}")

        # SQLite
        await self._db.execute(
            "DELETE FROM content_hashes WHERE seen_at < ?", (cutoff_30d,))
        await self._db.execute(
            "DELETE FROM image_hashes WHERE seen_at < ?", (cutoff_30d,))
        await self._db.execute(
            "DELETE FROM recent_posts WHERE created_at < ?", (cutoff_7d,))
        await self._db.execute(
            f"DELETE FROM recent_posts WHERE created_at < ? AND {non_vip_sq}",
            (cutoff_2d,))
        await self._db.commit()
        log.info("🧹 SQLite cleanup done — VIP 7d, others 2d")

    # ── Stats ──────────────────────────────────────────────────────────────────

    async def stats(self) -> dict:
        cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        hashes = posted_24h = 0
        mode = "supabase" if self._use_pg else "sqlite"
        try:
            if self._use_pg and self._pg:
                async with self._pg.acquire() as conn:
                    hashes = (await conn.fetchrow(
                        "SELECT COUNT(*) as n FROM content_hashes"))["n"]
                    posted_24h = (await conn.fetchrow(
                        "SELECT COUNT(*) as n FROM posted_messages WHERE posted_at > $1",
                        cutoff_24h))["n"]
            else:
                async with self._db.execute(
                    "SELECT COUNT(*) as n FROM content_hashes") as cur:
                    hashes = (await cur.fetchone())["n"]
                async with self._db.execute(
                    "SELECT COUNT(*) as n FROM posted_messages WHERE posted_at > ?",
                    (cutoff_24h,)) as cur:
                    posted_24h = (await cur.fetchone())["n"]
        except Exception:
            pass
        return {
            "tracked_hashes": hashes,
            "posted_last_24h": posted_24h,
            "db_mode": mode,
        }


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
