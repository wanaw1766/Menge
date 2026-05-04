"""
scraper.py — AXIOM INTEL channel scraper and forwarder (Unabridged)
- Pure AI‑based calendar detection (no caption dependency)
- Daily vs weekly detection using Gemini (primary) + Groq (fallback)
- Groups same‑time events with commas
- Weekly lock resets on Sunday 2 AM EAT
- Strong duplicate detection: short‑term memory, DB phash, AI similarity
- Posts to all destination channels simultaneously
"""

import asyncio
import base64
import io
import json
import logging
import mimetypes
import random
import re
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional, List

import pytz
from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
from telethon.errors import FloodWaitError, ChatWriteForbiddenError

from ai_engine import AIEngine, _add_signature
from memory import MemoryManager

log = logging.getLogger("scraper")

# Timezone for Addis Ababa (EAT, GMT+3)
EAT = pytz.timezone("Africa/Addis_Ababa")

# Allowed image MIME types (we can download these)
_IMG_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

# Keywords used to mark priority events (for logging, not for filtering)
_PRIORITY_KEYWORDS = [
    "fomc", "federal open market committee", "interest rate decision",
    "rate decision", "nfp", "non-farm payroll", "non-farm payrolls",
    "cpi", "consumer price index", "pce", "core pce", "gdp",
    "fed chair", "powell speaks", "jerome powell",
    "unemployment rate", "retail sales", "gold", "xau",
]

# Unique event names used to detect duplicate same‑day events
_UNIQUE_EVENT_NAMES = [
    "federal funds rate", "interest rate decision",
    "non-farm payroll", "nfp", "consumer price index", "cpi",
    "pce", "gdp", "fed chair", "powell speaks"
]

# Geopolitical keywords – used to filter reminders (reminders not sent for geopolitical events)
GEOPOLITICAL_KEYWORDS = [
    "trump", "iran", "hormuz", "war", "missile", "strike", "attack",
    "geopolitical", "oil supply", "ukraine", "russia", "biden", "putin", "xi"
]

# Flexible regex for parsing event lines from AI output
_EVENT_PATTERN = re.compile(
    r"(🔴|🟠)\s+(\d{1,2}:\d{2}\s*[AP]M)\s*[|\-:]\s*([A-Z]{3})\s*[:\-]?\s*(.+?)(?=\n|$)",
    re.IGNORECASE
)


def _is_reminder_eligible(event: dict) -> bool:
    """Return True if a reminder should be sent for this red event (non‑geopolitical)."""
    if event.get("impact") != "red":
        return False
    name_lower = event.get("name", "").lower()
    if any(kw in name_lower for kw in GEOPOLITICAL_KEYWORDS):
        return False
    return True


def _is_image(msg) -> bool:
    """Check if a Telegram message contains an image we can download."""
    if isinstance(msg.media, MessageMediaPhoto):
        return True
    if isinstance(msg.media, MessageMediaDocument):
        doc = msg.media.document
        if doc and doc.mime_type in _IMG_MIMES:
            return True
    return False


def _doc_mime(msg) -> str:
    """Return MIME type of a document image."""
    if isinstance(msg.media, MessageMediaDocument):
        return msg.media.document.mime_type or "image/jpeg"
    return "image/jpeg"


def _eat_now() -> datetime:
    """Current datetime in EAT timezone (Addis Ababa)."""
    return datetime.now(EAT)


def _eat_today_str() -> str:
    """YYYY-MM-DD for today in EAT."""
    return _eat_now().strftime("%Y-%m-%d")


def _eat_ai_date() -> str:
    """Date format for AI prompt – without year (e.g., 'Friday, May 1')."""
    return _eat_now().strftime("%A, %B %-d")


def _eat_date_line() -> str:
    """Date line used in daily calendar post (no year, e.g., 'Friday, May 1')."""
    return _eat_now().strftime("%A, %B %-d")


def _normalise_urls(text: str) -> str:
    """Remove tracking parameters from URLs."""
    if not text:
        return text
    return re.sub(r'(\?|&)(utm_[^&]+|fbclid=[^&]+|ref=[^&]+|source=[^&]+)', '', text)


def _extract_events_from_ff_text(text: str) -> List[dict]:
    """
    Parse AI output line by line, extract events, then GROUP them by time.
    Same time slot → names joined with commas.
    If any event in a time slot is red, the whole slot becomes red.
    Returns list sorted by time ascending.
    """
    raw = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _EVENT_PATTERN.search(line)
        if not m:
            continue
        emoji, time_12h, currency, name = m.groups()
        time_12h = time_12h.strip()
        try:
            dt = datetime.strptime(time_12h, "%I:%M %p")
        except ValueError:
            try:
                dt = datetime.strptime(time_12h, "%I:%M%p")
            except ValueError:
                log.warning(f"Could not parse time: {repr(time_12h)}")
                continue
        time_24h = dt.strftime("%H:%M")
        time_12h_clean = dt.strftime("%-I:%M %p")   # "3:30 PM" not "03:30 PM"
        raw.append({
            "name": name.strip(),
            "currency": currency.strip(),
            "impact": "red" if emoji == "🔴" else "orange",
            "time_12h": time_12h_clean,
            "time_24h": time_24h,
        })

    if not raw:
        log.error("❌ No events extracted! Raw text snippet:\n%s", text[:800])
        return []

    # Group by time_24h
    grouped = {}
    for e in raw:
        key = e["time_24h"]
        if key not in grouped:
            grouped[key] = {
                "time_12h": e["time_12h"],
                "time_24h": key,
                "currency": "USD",
                "has_red": e["impact"] == "red",
                "names": [e["name"]],
            }
        else:
            grouped[key]["names"].append(e["name"])
            if e["impact"] == "red":
                grouped[key]["has_red"] = True

    result = []
    for key in sorted(grouped.keys()):
        g = grouped[key]
        result.append({
            "name": ", ".join(g["names"]),      # comma‑joined names
            "currency": g["currency"],
            "impact": "red" if g["has_red"] else "orange",
            "time_12h": g["time_12h"],
            "time_24h": g["time_24h"],
        })

    log.info(f"✅ Extracted {len(result)} time-slot(s): {[e['name'] for e in result]}")
    return result


def _get_weekly_lock_key() -> str:
    """
    Returns a lock key based on the most recent Sunday at 2:00 AM EAT.
    Format: YYYYMMDD_HHMM (e.g., '20260503_0200')
    This resets every Sunday at 2 AM, allowing a new weekly calendar.
    """
    now = _eat_now()
    # Days to subtract to get last Sunday (Monday=0, Sunday=6)
    days_back = (now.weekday() + 1) % 7
    last_sunday = now.replace(hour=2, minute=0, second=0, microsecond=0) - timedelta(days=days_back)
    if now < last_sunday:
        last_sunday -= timedelta(days=7)
    return last_sunday.strftime("%Y%m%d_%H%M")


class ChannelScraper:
    """Main scraper class – polls source channels, processes messages, posts to destinations."""

    def __init__(self, config: dict, ai_engine: AIEngine, memory: MemoryManager):
        self._cfg = config
        self._ai = ai_engine
        self._mem = memory

        # Destination channels (can be multiple)
        self._dest_channels: List[str] = config.get("dest_channels", [])
        if not self._dest_channels:
            single = config.get("dest_channel", "")
            if single:
                self._dest_channels = [single]
        if not self._dest_channels:
            raise ValueError("No destination channels configured.")
        log.info(f"📤 Posting to {len(self._dest_channels)} destination(s): {self._dest_channels}")

        self._sources = config["source_channels"]
        self._min_delay = config["min_delay_seconds"]
        self._max_delay = config["max_delay_seconds"]
        self._lookback_hours = config["lookback_hours"]
        self._todays_vip_events: List[dict] = []

        # Short‑term memory for instant duplicate rejection
        self._recent_hashes = deque(maxlen=50)
        self._recent_phashes = deque(maxlen=50)

        # Telegram client setup
        session_string = config.get("session_string", "").strip()
        if session_string:
            session = StringSession(session_string)
        else:
            session = config.get("session_name", "manager_session")
        self._client = TelegramClient(session, config["api_id"], config["api_hash"])

    async def start(self):
        """Connect and authorise the Telegram client."""
        session_string = self._cfg.get("session_string", "").strip()
        if session_string:
            await self._client.connect()
            if not await self._client.is_user_authorized():
                raise RuntimeError("StringSession invalid or expired.")
        else:
            phone = self._cfg.get("phone", "")
            await self._client.start(phone=phone if phone else None)
        me = await self._client.get_me()
        log.info(f"✅ Logged in as: {me.first_name} (@{me.username or me.id})")

    async def stop(self):
        """Disconnect the Telegram client."""
        await self._client.disconnect()

    async def _ensure_connected(self) -> bool:
        """Ensure the client is connected; reconnect if needed."""
        if not self._client.is_connected():
            log.warning("Telethon disconnected — reconnecting …")
            try:
                await self._client.connect()
                if not await self._client.is_user_authorized():
                    log.error("Session expired.")
                    return False
                log.info("✅ Reconnected.")
            except Exception as exc:
                log.error(f"Reconnect failed: {exc}")
                return False
        return True

    async def _broadcast_text(self, text: str):
        """Send a text message to all destination channels."""
        sent = None
        for dest in self._dest_channels:
            try:
                sent = await self._client.send_message(dest, text, parse_mode="md")
                log.info(f"  → Text sent to {dest} | msg_id={sent.id}")
            except ChatWriteForbiddenError:
                log.error(f"❌ No permission to post to {dest}.")
            except FloodWaitError as fwe:
                log.warning(f"FloodWait {fwe.seconds}s on {dest}")
                await asyncio.sleep(fwe.seconds + 3)
                try:
                    sent = await self._client.send_message(dest, text, parse_mode="md")
                except Exception as exc:
                    log.error(f"Retry failed for {dest}: {exc}")
            except Exception as exc:
                log.error(f"Send text error on {dest}: {exc}", exc_info=True)
            await asyncio.sleep(1)
        return sent

    async def _broadcast_file_with_caption(self, file_bytes: bytes, mime: str,
                                           caption: str, reply_to: int = None):
        """Send an image with a caption (used for calendar posts)."""
        sent = None
        for dest in self._dest_channels:
            try:
                ext = mimetypes.guess_extension(mime) or ".png"
                buf = io.BytesIO(file_bytes)
                buf.name = f"calendar{ext}"
                buf.seek(0)
                sent = await self._client.send_file(
                    dest, buf, caption=caption, parse_mode="md",
                    force_document=False, reply_to=reply_to
                )
                log.info(f"  → File sent to {dest} | msg_id={sent.id}")
            except Exception as exc:
                log.error(f"Send file error on {dest}: {exc}")
                try:
                    sent = await self._client.send_message(
                        dest, caption, parse_mode="md", reply_to=reply_to
                    )
                except Exception as exc2:
                    log.error(f"Text fallback failed for {dest}: {exc2}")
            await asyncio.sleep(1)
        return sent

    async def _broadcast_media(self, text: str, image_data: Optional[bytes],
                               image_mime: str, reply_to: int = None):
        """Broadcast a post (text + optional image) to all destinations."""
        sent = None
        for dest in self._dest_channels:
            try:
                if image_data:
                    buf = io.BytesIO(image_data)
                    ext = mimetypes.guess_extension(image_mime) or ".jpg"
                    buf.name = f"media{ext}"
                    buf.seek(0)
                    sent = await self._client.send_file(
                        dest, buf, caption=text, parse_mode="md", reply_to=reply_to
                    )
                else:
                    sent = await self._client.send_message(
                        dest, text, parse_mode="md", reply_to=reply_to
                    )
                log.info(f"  → Post sent to {dest} | msg_id={sent.id}")
            except ChatWriteForbiddenError:
                log.error(f"❌ No permission to post to {dest}.")
            except FloodWaitError as fwe:
                log.warning(f"FloodWait {fwe.seconds}s on {dest}")
                await asyncio.sleep(fwe.seconds + 3)
                try:
                    if image_data:
                        buf = io.BytesIO(image_data)
                        ext = mimetypes.guess_extension(image_mime) or ".jpg"
                        buf.name = f"media{ext}"
                        buf.seek(0)
                        sent = await self._client.send_file(
                            dest, buf, caption=text, parse_mode="md", reply_to=reply_to
                        )
                    else:
                        sent = await self._client.send_message(
                            dest, text, parse_mode="md", reply_to=reply_to
                        )
                except Exception:
                    pass
            except Exception as exc:
                log.error(f"Send error on {dest}: {exc}", exc_info=True)
            await asyncio.sleep(1)
        return sent

    async def poll_and_forward(self):
        """Main polling loop – called every POLL_INTERVAL seconds."""
        stats = await self._mem.stats()
        log.info(f"Poll | sources={len(self._sources)} | hashes={stats['tracked_hashes']} | posted_24h={stats['posted_last_24h']}")
        if not await self._ensure_connected():
            log.warning("Skipping poll — not connected.")
            return
        # Check for upcoming reminders before polling
        await self._check_reminders()
        for channel in self._sources:
            try:
                await self._process_channel(channel)
            except FloodWaitError as fwe:
                log.warning(f"FloodWait {fwe.seconds}s — sleeping …")
                await asyncio.sleep(fwe.seconds + 5)
            except Exception as exc:
                log.error(f"Error on {channel}: {exc}", exc_info=True)
                await asyncio.sleep(5)

    async def _process_channel(self, channel: str):
        """Fetch new messages from one source channel and process each."""
        if not await self._ensure_connected():
            return
        last_id = await self._mem.get_last_msg_id(channel)
        cutoff = None
        if last_id == 0:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=self._lookback_hours)
        new_last_id = last_id
        collected = []
        try:
            async for msg in self._client.iter_messages(
                channel, limit=50,
                min_id=last_id if last_id else 0,
                offset_date=cutoff, reverse=True
            ):
                if msg.id <= last_id:
                    continue
                if not (msg.text or msg.media):
                    continue
                collected.append(msg)
                new_last_id = max(new_last_id, msg.id)
        except Exception as exc:
            log.error(f"iter_messages error on {channel}: {exc}", exc_info=True)
            await self._ensure_connected()
            return
        if not collected:
            log.debug(f"No new messages from {channel}")
            await self._mem.set_last_msg_id(channel, new_last_id)
            return
        log.info(f"📨 {len(collected)} new message(s) from {channel}")
        for msg in collected:
            await self._handle_message(msg, channel)
            await asyncio.sleep(random.uniform(2, 6))
        await self._mem.set_last_msg_id(channel, new_last_id)

    async def _handle_message(self, msg, source_channel: str):
        """
        Process a single Telegram message:
        - Download image if present
        - Detect if it's a ForexFactory calendar (using AI)
        - If yes, handle as daily/weekly calendar
        - Otherwise, normal news: duplicate checks, AI analysis, post
        """
        text = msg.text or msg.message or ""
        text = _normalise_urls(text)

        image_data: Optional[bytes] = None
        image_mime = "image/jpeg"
        phash: Optional[str] = None

        if msg.media and _is_image(msg):
            try:
                buf = io.BytesIO()
                await self._client.download_media(msg.media, file=buf)
                image_data = buf.getvalue()
                image_mime = _doc_mime(msg)
                phash = self._mem.compute_phash(image_data)
                log.debug(f"Image: {len(image_data):,} bytes | mime={image_mime} | phash={phash}")
            except Exception as exc:
                log.warning(f"Image download failed: {exc}")

        # ── Pure AI‑based calendar detection (no caption dependency) ──────────
        if image_data:
            is_ff = await self._image_looks_like_ff(image_data, image_mime)
            if is_ff:
                is_weekly = await self._is_weekly_image_ai(image_data, image_mime)
                await self._handle_ff_image(
                    image_data, image_mime, text, is_weekly, source_channel, msg.id
                )
                return  # Calendar handled, do not process as normal news

        # ── Normal news handling (non‑calendar) ───────────────────────────────
        content_hash = self._mem.hash_combined(text, image_data)

        # 1) Short‑term memory (instant)
        if content_hash in self._recent_hashes:
            log.info(f"[SKIP] Short‑term hash duplicate — {content_hash[:12]}…")
            return
        if phash and phash in self._recent_phashes:
            log.info(f"[SKIP] Short‑term phash duplicate — {phash}")
            return

        # 2) Database exact hash
        if await self._mem.is_duplicate(content_hash):
            log.info(f"[SKIP] Exact hash duplicate — {content_hash[:12]}…")
            return

        # 3) Database phash (Hamming ≤3)
        if phash and await self._mem.is_image_duplicate(phash, max_distance=3):
            log.info(f"[SKIP] Image phash duplicate (Hamming ≤3) — {phash}")
            await self._mem.mark_image_seen(phash, source_channel)
            return

        # 4) AI similarity check (aggressive)
        if await self._is_similar_to_recent(text, image_data, phash):
            log.info(f"[SKIP] Duplicate detected (AI similarity).")
            return

        # 5) Cross‑channel duplicate within 24h (optional extra check)
        if await self._is_recent_cross_channel_duplicate(content_hash, phash):
            log.info(f"[SKIP] Cross‑channel duplicate within 24h.")
            return

        # Mark as seen – now we are committed to processing this message
        await self._mem.mark_seen(content_hash, source=source_channel)
        if phash:
            await self._mem.mark_image_seen(phash, source_channel)
        self._recent_hashes.append(content_hash)
        if phash:
            self._recent_phashes.append(phash)

        log.info(f"🔍 Analysing msg {msg.id} from {source_channel} | text={len(text)}c | image={'✅' if image_data else '❌'}")
        verdict = await self._ai.analyse(text, image_data, image_mime)

        if not verdict.get("approved"):
            log.info(f"[REJECTED] reason='{verdict.get('reason')}' | issues={verdict.get('issues')}")
            return

        post_text = verdict.get("formatted_text", "").strip()
        if not post_text:
            return

        # Simulate typing delay
        delay = random.uniform(self._min_delay, self._max_delay)
        log.info(f"⏳ Waiting {delay:.1f}s before posting …")
        await asyncio.sleep(delay)
        await self._simulate_typing(len(post_text))

        sent = await self._broadcast_media(post_text, image_data, image_mime)
        if sent is None:
            return

        await self._mem.log_posted(
            source_channel, msg.id, sent.id, content_hash, verdict, post_text
        )
        await self._mem.store_recent_post(
            source_text=text[:1000], post_text=post_text[:1000], image_phash=phash
        )
        log.info(f"✅ Posted → msg_id={sent.id} | confidence={verdict.get('confidence')}")

    async def _is_recent_cross_channel_duplicate(self, content_hash: str, phash: Optional[str]) -> bool:
        """Check if the same content or image has been posted in the last 24 hours from any source."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        # Check content_hash
        if self._mem._use_pg and self._mem._pg:
            try:
                async with self._mem._pg.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT 1 FROM posted_messages WHERE posted_at > $1 AND content_hash = $2 LIMIT 1",
                        cutoff, content_hash
                    )
                if row:
                    return True
            except Exception:
                pass
        else:
            try:
                async with self._mem._db.execute(
                    "SELECT 1 FROM posted_messages WHERE posted_at > ? AND content_hash = ? LIMIT 1",
                    (cutoff, content_hash)
                ) as cur:
                    row = await cur.fetchone()
                if row:
                    return True
            except Exception:
                pass
        # Check phash across recent posts
        if phash:
            recent = await self._mem.get_recent_posts(limit=500)
            for _, old_phash, posted_at in recent:
                if old_phash and self._mem.hamming_distance(phash, old_phash) <= 3:
                    posted_dt = datetime.fromisoformat(posted_at)
                    if posted_dt > datetime.now(timezone.utc) - timedelta(hours=24):
                        return True
        return False

    async def _is_similar_to_recent(self, new_text: str, new_image: Optional[bytes],
                                    new_phash: Optional[str] = None) -> bool:
        """
        Use AI to compare new content with recent posts (up to 500).
        Returns True if a duplicate is found (aggressive threshold).
        """
        try:
            recent = await self._mem.get_recent_posts(limit=500)
            today_date = _eat_now().date()
            for old_text, old_phash, old_date_str in recent:
                old_date = datetime.fromisoformat(old_date_str).date()
                # Phash match
                if new_phash and old_phash:
                    if self._mem.hamming_distance(new_phash, old_phash) <= 3:
                        log.info("Phash match (Hamming ≤3) — duplicate")
                        return True
                if old_text and new_text:
                    new_lower = new_text.lower()
                    old_lower = old_text.lower()
                    # Critical event name same day
                    for name in _UNIQUE_EVENT_NAMES:
                        if name in new_lower and name in old_lower and old_date == today_date:
                            log.info(f"Critical event '{name}' matched same day — duplicate")
                            return True
                    # AI semantic similarity
                    same = await self._ai.is_same_story(
                        text_a=new_text[:500],
                        text_b=old_text[:500],
                        image_a=new_image,
                        image_b=None,
                    )
                    if same:
                        log.info("AI similarity duplicate (threshold 0.55)")
                        return True
        except Exception as exc:
            log.warning(f"Similarity check error: {exc}")
        return False

    async def _image_looks_like_ff(self, image_data: bytes, image_mime: str) -> bool:
        """Ask AI whether the image is a ForexFactory calendar screenshot."""
        try:
            prompt = (
                "Is this image a screenshot from ForexFactory.com showing an economic calendar? "
                "Only answer true if it clearly shows a ForexFactory calendar (date, events, times). "
                "Respond with JSON: {\"is_ff\": true} or {\"is_ff\": false}"
            )
            parts = [
                {"inline_data": {"mime_type": image_mime,
                                 "data": base64.b64encode(image_data).decode()}},
                prompt
            ]
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self._ai._gemini_vision.generate_content(parts)
                ),
                timeout=30
            )
            data = json.loads(re.sub(r"```+(?:json)?", "", resp.text).strip())
            return bool(data.get("is_ff", False))
        except Exception as exc:
            log.warning(f"FF image check failed: {exc} — assuming not FF")
            return False

    async def _is_weekly_image_ai(self, image_data: bytes, image_mime: str) -> bool:
        """
        Determine if an FF calendar image is weekly (date range or multiple day headers)
        or daily (single date). Uses Gemini + Groq fallback.
        Returns True if weekly, False if daily (or uncertain).
        """
        prompt = (
            "Analyse this ForexFactory calendar screenshot. Answer with JSON only.\n"
            "Look for a date range like 'Week of May 4 – May 11' or 'May 4, 2026 - May 11, 2026'.\n"
            "Also look for multiple day headers like 'Monday — May 4', 'Tuesday — May 5'.\n"
            "If you see more than one distinct date or a date range, it's weekly.\n"
            "If only a single date appears (e.g., 'Friday, May 1'), it's daily.\n"
            "If you are unsure, default to daily (false).\n"
            "Respond: {\"is_weekly\": true/false}"
        )
        # Try Gemini first
        try:
            parts = [{"inline_data": {"mime_type": image_mime, "data": base64.b64encode(image_data).decode()}}, prompt]
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._ai._gemini_vision.generate_content(parts)),
                timeout=25
            )
            data = json.loads(re.sub(r"```+(?:json)?", "", resp.text).strip())
            is_weekly = data.get("is_weekly", False)
            log.info(f"Weekly detection (Gemini) → {'weekly' if is_weekly else 'daily'}")
            return is_weekly
        except Exception as exc:
            log.warning(f"Gemini weekly detection failed ({exc}) — trying Groq…")

        # Fallback to Groq
        try:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{base64.b64encode(image_data).decode()}"}},
                {"type": "text", "text": prompt},
            ]
            resp = await asyncio.wait_for(
                self._ai._groq.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": content}],
                    temperature=0.1,
                    max_tokens=200,
                ),
                timeout=25,
            )
            data = json.loads(re.sub(r"```+(?:json)?", "", resp.choices[0].message.content).strip())
            is_weekly = data.get("is_weekly", False)
            log.info(f"Weekly detection (Groq) → {'weekly' if is_weekly else 'daily'}")
            return is_weekly
        except Exception as exc:
            log.error(f"Both engines failed for weekly detection: {exc} — defaulting to daily")
            return False  # default to daily to avoid misclassifying a daily as weekly

    def _select_vip_events(self, events: List[dict]) -> List[dict]:
        """Select all red events that are eligible for reminders (non‑geopolitical)."""
        eligible = [e for e in events if _is_reminder_eligible(e)]
        vip = sorted(eligible, key=lambda e: e.get("time_24h", "99:99"))
        log.info(f"VIP events for reminders: {[e.get('name') for e in vip]}")
        return vip

    async def _check_reminders(self):
        """Check if any red event is 10‑11 minutes away; send reminder if yes."""
        today_str = _eat_today_str()
        reminder_count = await self._mem.get_reminder_count_today(today_str)
        log.info(f"Reminder status: {reminder_count} sent today.")

        briefing_msg_id = await self._mem.get_daily_briefing_msg_id(today_str)
        if not briefing_msg_id or briefing_msg_id == -1:
            return

        # Get VIP events from daily briefing (if not already cached)
        vip_events = self._todays_vip_events
        if not vip_events:
            try:
                async with self._mem._db.execute(
                    "SELECT events_json FROM daily_briefings WHERE date_str=?", (today_str,)
                ) as cur:
                    row = await cur.fetchone()
                if row and row["events_json"]:
                    all_events = json.loads(row["events_json"])
                    vip_events = self._select_vip_events(all_events)
                    self._todays_vip_events = vip_events
            except Exception as exc:
                log.warning(f"Could not recover VIP events: {exc}")
            if not vip_events:
                return

        now = _eat_now()
        now_naive = now.replace(tzinfo=None)

        for event in vip_events:
            event_key = f"{today_str}_{event.get('name', '')}_{event.get('currency', '')}"
            if await self._mem.has_reminder_been_sent(event_key):
                continue

            event_time_str = event.get("time_24h", "")
            if not event_time_str:
                continue
            try:
                event_time = datetime.strptime(
                    f"{now.strftime('%Y-%m-%d')} {event_time_str}", "%Y-%m-%d %H:%M"
                )
            except ValueError:
                continue

            minutes_until = (event_time - now_naive).total_seconds() / 60
            if minutes_until < 0:
                continue

            log.info(f"⏰ {event.get('name')} at {event.get('time_12h')} — {minutes_until:.0f} min away")

            if 9 <= minutes_until <= 11:
                await self._send_reminder(
                    event, event_key, briefing_msg_id, today_str,
                    int(round(minutes_until))
                )
                await asyncio.sleep(2)

    async def _send_reminder(self, event: dict, event_key: str,
                             reply_to_msg_id: int, today_str: str, minutes_left: int):
        """Send a 10‑minute reminder for a red event."""
        log.info(f"⏰ Sending {minutes_left}-min reminder for {event.get('name')}")
        event_name = event.get("name", "Unknown Event")
        impact_emoji = "🔴" if event.get("impact") == "red" else "🟠"
        be_careful = await self._ai.get_be_careful_line(event_name)

        alert_text = (
            f"🚨 ALERT: {minutes_left} MINUTES REMAINING\n\n"
            f"{impact_emoji} {event_name}\n"
            f"🕒 {event.get('time_12h')}\n\n"
            f"REQUIRED ACTION:\n"
            f"✅ Secure open profits now\n"
            f"✅ Move Stop-Loss to Break-even\n"
            f"✅ No new entries during the release\n\n"
            f"{be_careful}"
        )
        alert_text = _add_signature(alert_text)

        for dest in self._dest_channels:
            try:
                sent = await self._client.send_message(
                    dest, alert_text, parse_mode="md", reply_to=reply_to_msg_id
                )
                if sent:
                    log.info(f"🚨 Reminder sent to {dest} → msg_id={sent.id}")
            except Exception as exc:
                log.error(f"Reminder send failed to {dest}: {exc}", exc_info=True)
            await asyncio.sleep(1)

        await self._mem.mark_reminder_sent(event_key)
        await self._mem.increment_reminder_count(today_str)
        log.info(f"Reminder sent. Daily total: {await self._mem.get_reminder_count_today(today_str)}")

    async def _simulate_typing(self, text_len: int):
        """Simulate human typing delay before sending."""
        duration = min(max(text_len / 180, 2), 14)
        if self._dest_channels:
            try:
                async with self._client.action(self._dest_channels[0], "typing"):
                    await asyncio.sleep(duration)
            except Exception:
                pass

    async def reminder_dispatcher_loop(self):
        """Background loop that checks reminders every 60 seconds."""
        log.info("🔔 Reminder dispatcher running …")
        while True:
            try:
                await self._check_reminders()
            except Exception as exc:
                log.error(f"Reminder dispatcher error: {exc}", exc_info=True)
            await asyncio.sleep(60)

    async def _handle_ff_image(self, image_data: bytes, image_mime: str, caption: str,
                               is_weekly: bool, source_channel: str, msg_id: int):
        """
        Handle a ForexFactory calendar image (daily or weekly).
        - Performs duplicate checks (phash)
        - Locks weekly calendar using Sunday 2 AM key
        - Builds and posts the final calendar message
        """
        today_str = _eat_today_str()
        today_ai_date = _eat_ai_date()
        phash = self._mem.compute_phash(image_data)

        # Duplicate check (do NOT mark seen yet)
        if phash and await self._mem.is_image_duplicate(phash, max_distance=3):
            log.info(f"[SKIP] FF image phash duplicate — {phash}")
            return

        if is_weekly:
            weekly_lock_key = _get_weekly_lock_key()
            if await self._mem.has_weekly_posted(weekly_lock_key):
                log.info(f"[SKIP] Weekly already posted for this Sunday cycle ({weekly_lock_key}).")
                return
            await self._mem.save_weekly_posted(weekly_lock_key)
            log.info("📆 Weekly FF image — analysing …")
            result = await self._ai.analyse_ff_image(
                image_data, image_mime,
                today_date=today_ai_date, is_weekly=True, week_range=""
            )
            if not result.get("approved"):
                await self._mem.delete_weekly_posted(weekly_lock_key)
                log.info(f"[SKIP] Weekly rejected: {result.get('reason')}")
                return
            post_text = result.get("formatted_text", "").strip()
            if not post_text:
                await self._mem.delete_weekly_posted(weekly_lock_key)
                return
            post_text = _add_signature(post_text)
            # Mark phash seen only after approval
            if phash:
                await self._mem.mark_image_seen(phash, source_channel)
            sent = await self._broadcast_file_with_caption(image_data, image_mime, post_text)
            if sent:
                log.info(f"📆 Weekly posted → msg_id={sent.id}")
            else:
                await self._mem.delete_weekly_posted(weekly_lock_key)

        else:  # Daily calendar
            if await self._mem.has_daily_briefing(today_str):
                log.info(f"[SKIP] Daily briefing already posted ({today_str}).")
                return
            # Lock with placeholder BEFORE AI call
            await self._mem.save_daily_briefing(today_str, -1, [])
            log.info(f"📅 Daily FF image — analysing … (date sent to AI: {today_ai_date})")

            result = await self._ai.analyse_ff_image(
                image_data, image_mime,
                today_date=today_ai_date, is_weekly=False
            )
            if not result.get("approved"):
                await self._mem.delete_daily_briefing(today_str)
                log.info(f"[SKIP] Daily rejected: {result.get('reason')}")
                return

            raw_text = result.get("formatted_text", "").strip()
            if not raw_text:
                await self._mem.delete_daily_briefing(today_str)
                return

            log.info(f"📄 Raw AI output:\n{raw_text}")

            # Parse and group events
            events = _extract_events_from_ff_text(raw_text)

            # Filter out geopolitical events (they don't appear in FF calendar anyway)
            events = [
                e for e in events
                if not any(kw in e.get("name", "").lower() for kw in GEOPOLITICAL_KEYWORDS)
            ]

            if not events:
                await self._mem.delete_daily_briefing(today_str)
                log.info("All events geopolitical — skipping.")
                return

            self._todays_vip_events = self._select_vip_events(events)
            log.info(f"VIP reminders: {[e.get('name') for e in self._todays_vip_events]}")

            # Build final post
            title = "TODAY'S USD 🇺🇸 HIGH IMPACT NEWS"
            date_line = _eat_date_line()

            lines = [title, "", date_line, ""]
            for e in events:
                emoji = "🔴" if e["impact"] == "red" else "🟠"
                lines.append(f"{emoji} {e['time_12h']} | {e['currency']}: {e['name']}")

            lines.append("")
            lines.append("Be careful during these releases.")
            post_text = "\n".join(lines)
            post_text = _add_signature(post_text)

            # Mark phash seen after successful processing
            if phash:
                await self._mem.mark_image_seen(phash, source_channel)

            sent = await self._broadcast_file_with_caption(image_data, image_mime, post_text)
            if sent:
                await self._mem.save_daily_briefing(today_str, sent.id, events)
                log.info(f"📅 Daily briefing posted → msg_id={sent.id}")
            else:
                await self._mem.delete_daily_briefing(today_str)
