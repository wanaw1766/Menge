"""
scraper.py — AXIOM INTEL channel scraper and forwarder (FIXED).
- Cross-platform time formatting (no %-d / %-I).
- Detailed logging for debugging.
- Geopolitical news always approved (via ai_engine exemption).
- FF calendar detection and posting.
- Duplicate image detection with adjustable distance (default 15).
"""

import asyncio
import io
import json
import logging
import mimetypes
import random
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple

import pytz
from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
from telethon.errors import FloodWaitError, ChatWriteForbiddenError

from ai_engine import AIEngine, _add_signature
from memory import MemoryManager

log = logging.getLogger("scraper")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

EAT = pytz.timezone("Africa/Addis_Ababa")
_IMG_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

_FF_CAPTION_KEYWORDS = (
    "forexfactory", "forex factory", "economic calendar",
    "high impact", "weekly news", "today news",
)
_VIP_KEYWORDS = [
    "fomc", "federal funds rate", "interest rate decision",
    "nfp", "non-farm payroll", "cpi", "pce", "gdp",
    "fed chair", "powell speaks",
]
_UNIQUE_EVENT_NAMES = [
    "federal funds rate", "interest rate decision",
    "non-farm payroll", "nfp", "consumer price index", "cpi",
]
GEOPOLITICAL_KEYWORDS = [
    "trump", "iran", "hormuz", "war", "missile", "strike",
    "ukraine", "russia", "putin", "xi",
]

_EVENT_PATTERN = re.compile(
    r"(🔴|🟠)\s+(\d{1,2}:\d{2}\s*[AP]M)\s*\|\s*([A-Z]{3})\s*:\s*(.+?)(?=\n|$)",
    re.IGNORECASE,
)

# ── Cross‑platform time helpers ──────────────────────────────────────────────
def format_12h(dt: datetime) -> str:
    hour = dt.hour % 12
    if hour == 0:
        hour = 12
    minute = dt.minute
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{hour}:{minute:02d} {ampm}"

def _extract_events(text: str) -> List[dict]:
    if not text:
        return []
    text = text.replace("\\n", "\n")
    raw = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _EVENT_PATTERN.search(line)
        if not m:
            continue
        emoji, time_str, currency, name = m.groups()
        time_str = time_str.strip().upper()
        for fmt in ["%I:%M %p", "%I:%M%p"]:
            try:
                dt = datetime.strptime(time_str, fmt)
                time_24h = dt.strftime("%H:%M")
                time_12h = format_12h(dt)
                break
            except ValueError:
                continue
        else:
            continue
        raw.append({
            "name": name.strip(),
            "currency": currency.strip(),
            "impact": "red" if emoji == "🔴" else "orange",
            "time_12h": time_12h,
            "time_24h": time_24h,
        })
    if not raw:
        return []
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
            "name": ", ".join(g["names"]),
            "currency": g["currency"],
            "impact": "red" if g["has_red"] else "orange",
            "time_12h": g["time_12h"],
            "time_24h": g["time_24h"],
        })
    return result

def _is_image(msg) -> bool:
    if isinstance(msg.media, MessageMediaPhoto):
        return True
    if isinstance(msg.media, MessageMediaDocument):
        doc = msg.media.document
        return doc and doc.mime_type in _IMG_MIMES
    return False

def _doc_mime(msg) -> str:
    if isinstance(msg.media, MessageMediaDocument):
        return msg.media.document.mime_type or "image/jpeg"
    return "image/jpeg"

def _eat_now() -> datetime:
    return datetime.now(EAT)

def _eat_today_str() -> str:
    return _eat_now().strftime("%Y-%m-%d")

def _eat_date_line() -> str:
    now = _eat_now()
    day = str(now.day).lstrip("0")
    return now.strftime(f"%A, %B {day}")

def _week_key() -> str:
    now = _eat_now()
    monday = now - timedelta(days=now.weekday())
    return monday.strftime("%Y-%m-%d")

def _week_range_str() -> str:
    now = _eat_now()
    monday = now - timedelta(days=now.weekday())
    friday = monday + timedelta(days=4)
    mon_day = str(monday.day).lstrip("0")
    fri_day = str(friday.day).lstrip("0")
    return f"{monday.strftime('%b')} {mon_day} – {friday.strftime('%b')} {fri_day}, {friday.year}"

def _looks_like_ff_caption(text: str) -> bool:
    return bool(text and any(kw in text.lower() for kw in _FF_CAPTION_KEYWORDS))

def _normalise_urls(text: str) -> str:
    if not text:
        return text
    return re.sub(r'(\?|&)(utm_[^&]+|fbclid=[^&]+)', '', text)

# ── ChannelScraper class ──────────────────────────────────────────────────────
class ChannelScraper:
    def __init__(self, config: dict, ai_engine: AIEngine, memory: MemoryManager):
        self._cfg = config
        self._ai = ai_engine
        self._mem = memory
        self._dest_channels = config.get("dest_channels", [])
        if not self._dest_channels:
            single = config.get("dest_channel", "")
            if single:
                self._dest_channels = [single]
        if not self._dest_channels:
            raise ValueError("No destination channels configured.")
        self._sources = config["source_channels"]
        self._min_delay = config.get("min_delay_seconds", 2)
        self._max_delay = config.get("max_delay_seconds", 5)
        self._lookback_hours = config.get("lookback_hours", 24)
        self._todays_vip = []

        session_string = config.get("session_string", "").strip()
        if session_string:
            session = StringSession(session_string)
        else:
            session = config.get("session_name", "manager_session")
        self._client = TelegramClient(session, config["api_id"], config["api_hash"])
        self._running = False

    async def start(self):
        await self._client.start()
        me = await self._client.get_me()
        log.info(f"Scraper started as {me.username or me.id}")
        # Test destination channels
        for dest in self._dest_channels:
            try:
                await self._client.send_message(dest, "🟢 AXIOM INTEL online – monitoring started", parse_mode=None)
                log.info(f"Test message sent to {dest}")
            except Exception as e:
                log.error(f"Cannot send to {dest}: {e}")

    async def stop(self):
        self._running = False
        await self._client.disconnect()
        log.info("Scraper stopped")

    async def _ensure_connected(self) -> bool:
        if not self._client.is_connected():
            try:
                await self._client.connect()
                if not await self._client.is_user_authorized():
                    log.error("Not authorized")
                    return False
            except Exception as e:
                log.error(f"Connection error: {e}")
                return False
        return True

    async def _send_text(self, text: str, reply_to: int = None):
        sent = None
        for dest in self._dest_channels:
            try:
                sent = await self._client.send_message(dest, text, parse_mode="md", reply_to=reply_to)
                log.debug(f"Sent text to {dest}")
            except FloodWaitError as e:
                log.warning(f"FloodWait {e.seconds}s, sleeping")
                await asyncio.sleep(e.seconds + 2)
                try:
                    sent = await self._client.send_message(dest, text, parse_mode="md", reply_to=reply_to)
                except Exception as ex:
                    log.error(f"Retry failed: {ex}")
            except Exception as e:
                log.error(f"Send error to {dest}: {e}")
            await asyncio.sleep(1)
        return sent

    async def _send_file(self, file_bytes: bytes, mime: str, caption: str, reply_to: int = None):
        sent = None
        for dest in self._dest_channels:
            try:
                ext = mimetypes.guess_extension(mime) or ".png"
                buf = io.BytesIO(file_bytes)
                buf.name = f"media{ext}"
                buf.seek(0)
                sent = await self._client.send_file(
                    dest, buf, caption=caption, parse_mode="md",
                    force_document=False, reply_to=reply_to
                )
                log.debug(f"Sent file to {dest}")
            except FloodWaitError as e:
                await asyncio.sleep(e.seconds + 2)
                try:
                    buf.seek(0)
                    sent = await self._client.send_file(
                        dest, buf, caption=caption, parse_mode="md",
                        force_document=False, reply_to=reply_to
                    )
                except Exception:
                    pass
            except Exception as e:
                log.error(f"File send error: {e}")
                try:
                    sent = await self._client.send_message(dest, caption, parse_mode="md", reply_to=reply_to)
                except Exception:
                    pass
            await asyncio.sleep(1)
        return sent

    async def _send_media(self, text: str, image_data: Optional[bytes], image_mime: str, reply_to: int = None):
        if image_data:
            return await self._send_file(image_data, image_mime, text, reply_to)
        return await self._send_text(text, reply_to)

    async def _simulate_typing(self, text_len: int):
        duration = min(max(text_len / 180, 2), 14)
        if self._dest_channels:
            try:
                async with self._client.action(self._dest_channels[0], "typing"):
                    await asyncio.sleep(duration)
            except Exception:
                pass

    # ── Main polling (called by main.py's loop) ───────────────────────────────
    async def poll_and_forward(self):
        for channel in self._sources:
            try:
                await self._process_channel(channel)
            except FloodWaitError as e:
                log.warning(f"FloodWait on {channel}: {e.seconds}s")
                await asyncio.sleep(e.seconds + 5)
            except Exception as e:
                log.error(f"Channel {channel} error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _process_channel(self, channel: str):
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
            if collected:
                log.info(f"Fetched {len(collected)} new messages from {channel}")
            else:
                log.debug(f"No new messages from {channel}")
        except Exception as e:
            log.error(f"iter_messages failed for {channel}: {e}")
            return

        for msg in collected:
            await self._handle_message(msg, channel)
            await asyncio.sleep(random.uniform(self._min_delay, self._max_delay))
        await self._mem.set_last_msg_id(channel, new_last_id)

    # ── Message handler (with logging) ───────────────────────────────────────
    async def _handle_message(self, msg, source_channel: str):
        text = _normalise_urls(msg.text or msg.message or "")
        image_data = None
        image_mime = "image/jpeg"
        phash = None

        if msg.media and _is_image(msg):
            try:
                buf = io.BytesIO()
                await self._client.download_media(msg.media, file=buf)
                image_data = buf.getvalue()
                image_mime = _doc_mime(msg)
                phash = self._mem.compute_phash(image_data)
                log.debug(f"Downloaded image, phash={phash[:8] if phash else 'none'}")
            except Exception as e:
                log.warning(f"Image download failed: {e}")

        # ── FF calendar detection ───────────────────────────────────────────
        if image_data:
            caption_is_ff = _looks_like_ff_caption(text)
            if caption_is_ff:
                _, ai_is_weekly = await self._ai.detect_ff_image(image_data, image_mime)
                cap_is_weekly = any(kw in text.lower() for kw in ("week", "weekly", "monday", "tuesday"))
                is_weekly = cap_is_weekly or ai_is_weekly
                log.info(f"FF calendar detected (caption) weekly={is_weekly}")
                await self._handle_ff_image(image_data, image_mime, is_weekly, source_channel)
                return
            else:
                ai_is_ff, ai_is_weekly = await self._ai.detect_ff_image(image_data, image_mime)
                if ai_is_ff:
                    log.info(f"FF calendar detected (AI) weekly={ai_is_weekly}")
                    await self._handle_ff_image(image_data, image_mime, ai_is_weekly, source_channel)
                    return

        # ── Regular news ────────────────────────────────────────────────────
        content_hash = self._mem.hash_combined(text, image_data)
        if await self._mem.is_duplicate(content_hash):
            log.debug("Duplicate content hash, skipping")
            return
        if phash and await self._mem.is_image_duplicate(phash, max_distance=15):
            log.info(f"Duplicate image (phash={phash[:8]}...), skipping")
            await self._mem.mark_image_seen(phash, source_channel)
            return

        await self._mem.mark_seen(content_hash, source=source_channel)
        if phash:
            await self._mem.mark_image_seen(phash, source_channel)

        if await self._is_similar_to_recent(text, image_data, phash):
            log.debug("Similar to recent post, skipping")
            return

        verdict = await self._ai.analyse(text, image_data, image_mime)
        if not verdict.get("approved"):
            log.info(f"AI rejected: {verdict.get('reason')}")
            return

        post_text = verdict.get("formatted_text", "").strip()
        if not post_text:
            log.warning("AI approved but formatted_text empty")
            return

        await asyncio.sleep(random.uniform(self._min_delay, self._max_delay))
        await self._simulate_typing(len(post_text))

        sent = await self._send_media(post_text, image_data, image_mime)
        if sent is None:
            log.error("Failed to send message to any destination")
            return

        await self._mem.log_posted(source_channel, msg.id, sent.id, content_hash, verdict, post_text)
        await self._mem.store_recent_post(source_text=text[:1000], post_text=post_text[:1000], image_phash=phash)
        log.info(f"Posted from {source_channel} (msg {msg.id}) -> {sent.id}")

    # ── Similarity check (with logging) ─────────────────────────────────────
    async def _is_similar_to_recent(self, new_text: str, new_image: Optional[bytes], new_phash: Optional[str]) -> bool:
        try:
            recent = await self._mem.get_recent_posts(limit=150)
            today_date = _eat_now().date()
            for old_text, old_phash, old_date_str in recent:
                if not old_date_str:
                    continue
                try:
                    old_date = datetime.fromisoformat(old_date_str).date()
                except (ValueError, TypeError):
                    continue
                if new_phash and old_phash:
                    dist = self._mem.hamming_distance(new_phash, old_phash)
                    if dist <= 10:
                        log.debug(f"Image hash distance {dist} <= 10, same image")
                        return True
                if not old_text or not new_text:
                    continue
                new_lower = new_text.lower()
                old_lower = old_text.lower()
                for name in _UNIQUE_EVENT_NAMES:
                    if name in new_lower and name in old_lower and old_date == today_date:
                        log.debug(f"Unique event '{name}' same day, duplicate")
                        return True
                key_terms = ["fomc", "fed rate", "powell", "nfp", "cpi", "gdp", "unemployment",
                             "gold", "oil", "trump", "tariff", "sanction", "ukraine", "dxy"]
                if old_date == today_date:
                    overlap = sum(1 for term in key_terms if term in new_lower and term in old_lower)
                    if overlap >= 2:
                        log.debug(f"Keyword overlap {overlap} >= 2, same day")
                        return True
                same = await self._ai.is_same_story(new_text[:500], old_text[:500], new_image)
                if same:
                    log.debug("AI similarity says same story")
                    return True
        except Exception as e:
            log.error(f"Similarity check error: {e}")
        return False

    # ── Reminders (fixed) ───────────────────────────────────────────────────
    async def _check_reminders(self):
        today_str = _eat_today_str()
        briefing_msg_id = await self._mem.get_daily_briefing_msg_id(today_str)
        if not briefing_msg_id or briefing_msg_id == -1:
            return
        if not self._todays_vip:
            try:
                async with self._mem._db.execute(
                    "SELECT events_json FROM daily_briefings WHERE date_str=?", (today_str,)
                ) as cur:
                    row = await cur.fetchone()
                if row and row["events_json"]:
                    all_events = json.loads(row["events_json"])
                    self._todays_vip = [e for e in all_events if self._is_reminder_eligible(e)]
                    self._todays_vip.sort(key=lambda x: x.get("time_24h", "99:99"))
            except Exception:
                return
        if not self._todays_vip:
            return
        now = _eat_now().replace(tzinfo=None)
        for event in self._todays_vip:
            event_key = f"{today_str}_{event.get('name', '')}_{event.get('currency', '')}"
            if await self._mem.has_reminder_been_sent(event_key):
                continue
            event_time_str = event.get("time_24h", "")
            if not event_time_str:
                continue
            try:
                event_time = datetime.strptime(f"{now.strftime('%Y-%m-%d')} {event_time_str}", "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            minutes_until = (event_time - now).total_seconds() / 60
            if 14 <= minutes_until <= 16:
                await self._send_reminder(event, event_key, briefing_msg_id, today_str)
                await asyncio.sleep(2)

    async def _send_reminder(self, event: dict, event_key: str, reply_to_msg_id: int, today_str: str):
        event_name = event.get("name", "Unknown Event")
        impact_emoji = "🔴" if event.get("impact") == "red" else "🟠"
        be_careful = await self._ai.get_be_careful_line(event_name)
        alert_text = (
            f"🚨 ALERT: 15 MINUTES REMAINING\n\n"
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
                await self._client.send_message(dest, alert_text, parse_mode="md", reply_to=reply_to_msg_id)
            except Exception as e:
                log.error(f"Reminder send error: {e}")
            await asyncio.sleep(1)
        await self._mem.mark_reminder_sent(event_key)
        await self._mem.increment_reminder_count(today_str)

    @staticmethod
    def _is_reminder_eligible(event: dict) -> bool:
        if event.get("impact") != "red":
            return False
        name_lower = event.get("name", "").lower()
        if any(kw in name_lower for kw in GEOPOLITICAL_KEYWORDS):
            return False
        return any(kw in name_lower for kw in _VIP_KEYWORDS)

    # ── FF image handler (with logging) ─────────────────────────────────────
    async def _handle_ff_image(self, image_data: bytes, image_mime: str, is_weekly: bool, source_channel: str):
        today_str = _eat_today_str()
        today_display = _eat_date_line()
        phash = self._mem.compute_phash(image_data)

        if phash and await self._mem.is_image_duplicate(phash, max_distance=15):
            log.info(f"FF image duplicate (phash={phash[:8]}...), skipping")
            return
        if phash:
            await self._mem.mark_image_seen(phash, source_channel)

        if is_weekly:
            wkey = _week_key()
            if await self._mem.has_weekly_posted(wkey):
                log.info(f"Weekly FF already posted for week {wkey}")
                return
            await self._mem.save_weekly_posted(wkey)
            result = await self._ai.analyse_ff_image(
                image_data, image_mime, today_date=today_display,
                is_weekly=True, week_range=_week_range_str()
            )
            if not result.get("approved"):
                log.warning(f"Weekly FF analysis failed: {result.get('reason')}")
                await self._mem.delete_weekly_posted(wkey)
                return
            post_text = result.get("formatted_text", "").strip()
            if not post_text:
                log.warning("Weekly FF approved but no formatted_text")
                await self._mem.delete_weekly_posted(wkey)
                return
            post_text = _add_signature(post_text)
            sent = await self._send_file(image_data, image_mime, post_text)
            if sent:
                log.info(f"Weekly FF posted (key {wkey})")
            else:
                await self._mem.delete_weekly_posted(wkey)
        else:
            if await self._mem.has_daily_briefing(today_str):
                log.info(f"Daily briefing already exists for {today_str}")
                return
            await self._mem.save_daily_briefing(today_str, -1, [])
            result = await self._ai.analyse_ff_image(
                image_data, image_mime, today_date=today_display, is_weekly=False
            )
            if not result.get("approved"):
                log.warning(f"Daily FF analysis failed: {result.get('reason')}")
                await self._mem.delete_daily_briefing(today_str)
                return
            raw_text = result.get("formatted_text", "").strip()
            if not raw_text:
                log.warning("Daily FF approved but no formatted_text")
                await self._mem.delete_daily_briefing(today_str)
                return

            events = _extract_events(raw_text)
            events = [e for e in events if not any(kw in e.get("name", "").lower() for kw in GEOPOLITICAL_KEYWORDS)]

            if not events:
                quiet = "🟢 No major news today — markets expected calm. Trade safe."
                quiet = _add_signature(quiet)
                sent = await self._send_file(image_data, image_mime, quiet)
                if sent:
                    await self._mem.save_daily_briefing(today_str, sent.id, [])
                else:
                    await self._mem.delete_daily_briefing(today_str)
                return

            self._todays_vip = [e for e in events if self._is_reminder_eligible(e)]
            has_red = any(e["impact"] == "red" for e in events)
            title = "TODAY'S USD 🇺🇸 HIGH IMPACT NEWS" if has_red else "TODAY'S USD 🇺🇸 NEWS"
            date_line = _eat_date_line()
            lines = [title, "", date_line, ""]
            for e in events:
                emoji = "🔴" if e["impact"] == "red" else "🟠"
                lines.append(f"{emoji} {e['time_12h']} | {e['currency']}: {e['name']}")
            if has_red:
                lines.append("")
                lines.append("Be careful during these releases.")
            post_text = "\n".join(lines)
            post_text = _add_signature(post_text)

            sent = await self._send_file(image_data, image_mime, post_text)
            if sent:
                await self._mem.save_daily_briefing(today_str, sent.id, events)
                log.info(f"Daily FF posted for {today_str}")
            else:
                await self._mem.delete_daily_briefing(today_str)
