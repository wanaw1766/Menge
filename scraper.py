"""
scraper.py — AXIOM INTEL channel scraper and forwarder.
- FF calendar: AI confirms daily/weekly in one call
- Daily: once per day
- Weekly: once per week (Monday-based key, consistent Sun-Sat)
- Geopolitical/FOMC/macro news: always posted, no limit
- Same-time events: comma-joined on one line
- Time: 12h AM/PM no leading zero (3:30 PM not 03:30 PM)
- All duplicate locks BEFORE AI call
- Silent logging
- AI calls protected by semaphore + similarity capped at 3
"""

import asyncio
import io
import json
import logging
import mimetypes
import random
import re
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

EAT = pytz.timezone("Africa/Addis_Ababa")
_IMG_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

# ── Keywords ──────────────────────────────────────────────────────────────────
_FF_CAPTION_KEYWORDS = (
    "forexfactory", "forex factory", "calendar", "economic calendar",
    "high impact", "impact news", "weekly news", "today news",
    "today's news", "weekly calendar", "this week",
    "fomc", "federal funds rate", "interest rate decision",
)

_VIP_KEYWORDS = [
    "fomc", "federal open market committee",
    "federal funds rate", "interest rate decision",
    "nfp", "non-farm payroll", "non-farm payrolls",
    "cpi", "consumer price index",
    "pce", "core pce",
    "gdp", "advance gdp", "preliminary gdp",
    "fed chair", "powell speaks", "jerome powell",
]

_UNIQUE_EVENT_NAMES = [
    "federal funds rate", "interest rate decision",
    "non-farm payroll", "nfp", "consumer price index", "cpi",
    "pce", "gdp", "fed chair", "powell speaks",
]

GEOPOLITICAL_KEYWORDS = [
    "trump", "iran", "hormuz", "war", "missile", "strike", "attack",
    "geopolitical", "oil supply", "ukraine", "russia", "biden", "putin", "xi",
]

# Bulletproof event regex — handles all AI output variations
_EVENT_PATTERN = re.compile(
    r"(🔴|🟠)\s+(\d{1,2}:\d{2}\s*[AP]M)\s*\|\s*([A-Z]{3})\s*:\s*(.+?)(?=\n|$)",
    re.IGNORECASE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_vip_event(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in _VIP_KEYWORDS)


def _is_reminder_eligible(event: dict) -> bool:
    if event.get("impact") != "red":
        return False
    name_lower = event.get("name", "").lower()
    if any(kw in name_lower for kw in GEOPOLITICAL_KEYWORDS):
        return False
    return _is_vip_event(name_lower)


def _is_image(msg) -> bool:
    if isinstance(msg.media, MessageMediaPhoto):
        return True
    if isinstance(msg.media, MessageMediaDocument):
        doc = msg.media.document
        if doc and doc.mime_type in _IMG_MIMES:
            return True
    return False


def _doc_mime(msg) -> str:
    if isinstance(msg.media, MessageMediaDocument):
        return msg.media.document.mime_type or "image/jpeg"
    return "image/jpeg"


def _eat_now() -> datetime:
    return datetime.now(EAT)


def _eat_today_str() -> str:
    return _eat_now().strftime("%Y-%m-%d")


def _eat_today_display() -> str:
    return _eat_now().strftime("%A, %B %-d, %Y")


def _eat_date_line() -> str:
    return _eat_now().strftime("%A, %B %-d")


def _week_key() -> str:
    """Always Monday's date — consistent across entire week Sun through Sat."""
    now    = _eat_now()
    monday = now - timedelta(days=now.weekday())
    return monday.strftime("%Y-%m-%d")


def _week_range_str() -> str:
    now    = _eat_now()
    monday = now - timedelta(days=now.weekday())
    friday = monday + timedelta(days=4)
    return f"{monday.strftime('%b %-d')} – {friday.strftime('%b %-d, %Y')}"


def _looks_like_ff_caption(text: str) -> bool:
    if not text:
        return False
    return any(kw in text.lower() for kw in _FF_CAPTION_KEYWORDS)


def _normalise_urls(text: str) -> str:
    if not text:
        return text
    return re.sub(
        r'(\?|&)(utm_[^&]+|fbclid=[^&]+|ref=[^&]+|source=[^&]+)', '', text
    )


def _text_similarity_ratio(a: str, b: str) -> float:
    """
    Fast token-overlap ratio — zero AI cost.
    Returns 0.0 (no overlap) to 1.0 (identical).
    Used to skip AI is_same_story when texts are obviously different.
    """
    if not a or not b:
        return 0.0
    tokens_a = set(re.findall(r'\b\w{4,}\b', a.lower()))
    tokens_b = set(re.findall(r'\b\w{4,}\b', b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union        = tokens_a | tokens_b
    return len(intersection) / len(union)


def _extract_events(text: str) -> List[dict]:
    """
    Parse AI output line by line, group same-time events.
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
        emoji, time_str, currency, name = m.groups()
        time_str = time_str.strip().upper()
        for fmt in ["%I:%M %p", "%I:%M%p"]:
            try:
                dt       = datetime.strptime(time_str, fmt)
                time_24h = dt.strftime("%H:%M")
                time_12h = dt.strftime("%-I:%M %p")
                break
            except ValueError:
                continue
        else:
            continue
        raw.append({
            "name":     name.strip(),
            "currency": currency.strip(),
            "impact":   "red" if emoji == "🔴" else "orange",
            "time_12h": time_12h,
            "time_24h": time_24h,
        })

    if not raw:
        return []

    grouped: dict = {}
    for e in raw:
        key = e["time_24h"]
        if key not in grouped:
            grouped[key] = {
                "time_12h": e["time_12h"],
                "time_24h": key,
                "currency": "USD",
                "has_red":  e["impact"] == "red",
                "names":    [e["name"]],
            }
        else:
            grouped[key]["names"].append(e["name"])
            if e["impact"] == "red":
                grouped[key]["has_red"] = True

    result = []
    for key in sorted(grouped.keys()):
        g = grouped[key]
        result.append({
            "name":     ", ".join(g["names"]),
            "currency": g["currency"],
            "impact":   "red" if g["has_red"] else "orange",
            "time_12h": g["time_12h"],
            "time_24h": g["time_24h"],
        })
    return result


# ── ChannelScraper ────────────────────────────────────────────────────────────

class ChannelScraper:
    def __init__(self, config: dict, ai_engine: AIEngine, memory: MemoryManager):
        self._cfg = config
        self._ai  = ai_engine
        self._mem = memory

        # ── Groq quota guard — max 2 AI calls running at the same time ────────
        self._ai_sem = asyncio.Semaphore(2)

        self._dest_channels: List[str] = config.get("dest_channels", [])
        if not self._dest_channels:
            single = config.get("dest_channel", "")
            if single:
                self._dest_channels = [single]
        if not self._dest_channels:
            raise ValueError("No destination channels configured.")

        self._sources        = config["source_channels"]
        self._min_delay      = config["min_delay_seconds"]
        self._max_delay      = config["max_delay_seconds"]
        self._lookback_hours = config["lookback_hours"]
        self._todays_vip: List[dict] = []

        session_string = config.get("session_string", "").strip()
        session = (
            StringSession(session_string) if session_string
            else config.get("session_name", "manager_session")
        )
        self._client = TelegramClient(session, config["api_id"], config["api_hash"])

    # ── Safe AI wrappers (all quota-protected) ────────────────────────────────

    async def _ai_detect_ff(self, image_data: bytes, image_mime: str):
        async with self._ai_sem:
            return await self._ai.detect_ff_image(image_data, image_mime)

    async def _ai_analyse(self, text: str, image_data, image_mime: str):
        async with self._ai_sem:
            return await self._ai.analyse(text, image_data, image_mime)

    async def _ai_analyse_ff(self, image_data: bytes, image_mime: str,
                             today_date: str, is_weekly: bool,
                             week_range: str = ""):
        async with self._ai_sem:
            return await self._ai.analyse_ff_image(
                image_data, image_mime,
                today_date=today_date,
                is_weekly=is_weekly,
                week_range=week_range,
            )

    async def _ai_is_same_story(self, text_a: str, text_b: str, image_a=None) -> bool:
        async with self._ai_sem:
            return await self._ai.is_same_story(
                text_a=text_a, text_b=text_b, image_a=image_a
            )

    async def _ai_be_careful(self, event_name: str) -> str:
        async with self._ai_sem:
            return await self._ai.get_be_careful_line(event_name)

    # ── Connect / disconnect ──────────────────────────────────────────────────

    async def start(self):
        session_string = self._cfg.get("session_string", "").strip()
        if session_string:
            await self._client.connect()
            if not await self._client.is_user_authorized():
                raise RuntimeError("StringSession invalid or expired.")
        else:
            phone = self._cfg.get("phone", "")
            await self._client.start(phone=phone if phone else None)

    async def stop(self):
        await self._client.disconnect()

    async def _ensure_connected(self) -> bool:
        if not self._client.is_connected():
            try:
                await self._client.connect()
                if not await self._client.is_user_authorized():
                    return False
            except Exception:
                return False
        return True

    # ── Broadcast ─────────────────────────────────────────────────────────────

    async def _send_text(self, text: str, reply_to: int = None):
        sent = None
        for dest in self._dest_channels:
            try:
                sent = await self._client.send_message(
                    dest, text, parse_mode="md", reply_to=reply_to
                )
            except FloodWaitError as fwe:
                await asyncio.sleep(fwe.seconds + 3)
                try:
                    sent = await self._client.send_message(
                        dest, text, parse_mode="md", reply_to=reply_to
                    )
                except Exception:
                    pass
            except (ChatWriteForbiddenError, Exception):
                pass
            await asyncio.sleep(1)
        return sent

    async def _send_file(self, file_bytes: bytes, mime: str,
                         caption: str, reply_to: int = None):
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
            except FloodWaitError as fwe:
                await asyncio.sleep(fwe.seconds + 3)
                try:
                    buf.seek(0)
                    sent = await self._client.send_file(
                        dest, buf, caption=caption, parse_mode="md",
                        force_document=False, reply_to=reply_to
                    )
                except Exception:
                    pass
            except Exception:
                try:
                    sent = await self._client.send_message(
                        dest, caption, parse_mode="md", reply_to=reply_to
                    )
                except Exception:
                    pass
            await asyncio.sleep(1)
        return sent

    async def _send_media(self, text: str, image_data: Optional[bytes],
                          image_mime: str, reply_to: int = None):
        if image_data:
            return await self._send_file(image_data, image_mime, text, reply_to)
        return await self._send_text(text, reply_to)

    # ── Main poll ─────────────────────────────────────────────────────────────

    async def poll_and_forward(self):
        if not await self._ensure_connected():
            return
        await self._check_reminders()
        for channel in self._sources:
            try:
                await self._process_channel(channel)
            except FloodWaitError as fwe:
                await asyncio.sleep(fwe.seconds + 5)
            except Exception as exc:
                log.error(f"Channel error {channel}: {exc}", exc_info=True)
                await asyncio.sleep(5)

    async def _process_channel(self, channel: str):
        if not await self._ensure_connected():
            return
        last_id = await self._mem.get_last_msg_id(channel)
        cutoff  = None
        if last_id == 0:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=self._lookback_hours)
        new_last_id = last_id
        collected   = []
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
            log.error(f"iter_messages error {channel}: {exc}", exc_info=True)
            return

        for msg in collected:
            await self._handle_message(msg, channel)
            await asyncio.sleep(random.uniform(2, 6))
        await self._mem.set_last_msg_id(channel, new_last_id)

    # ── Message handler ───────────────────────────────────────────────────────

    async def _handle_message(self, msg, source_channel: str):
        text = _normalise_urls(msg.text or msg.message or "")

        image_data: Optional[bytes] = None
        image_mime = "image/jpeg"
        phash: Optional[str] = None

        if msg.media and _is_image(msg):
            try:
                buf = io.BytesIO()
                await self._client.download_media(msg.media, file=buf)
                image_data = buf.getvalue()
                image_mime = _doc_mime(msg)
                phash      = self._mem.compute_phash(image_data)
            except Exception:
                pass

        # ── FF calendar detection ─────────────────────────────────────────────
        if image_data:
            caption_is_ff = _looks_like_ff_caption(text)

            if caption_is_ff:
                # Caption already confirmed FF — one AI call just for weekly flag
                _, ai_is_weekly = await self._ai_detect_ff(image_data, image_mime)
                cap_is_weekly = any(
                    kw in text.lower() for kw in
                    ("week", "weekly", "this week", "next week",
                     "monday", "tuesday", "wednesday", "thursday", "friday",
                     "mon —", "tue —", "wed —", "thu —", "fri —")
                ) if text else False
                is_weekly = cap_is_weekly or ai_is_weekly
                await self._handle_ff_image(
                    image_data, image_mime, is_weekly, source_channel
                )
                return
            else:
                # Caption doesn't confirm — AI decides both is_ff and is_weekly
                ai_is_ff, ai_is_weekly = await self._ai_detect_ff(
                    image_data, image_mime
                )
                if ai_is_ff:
                    await self._handle_ff_image(
                        image_data, image_mime, ai_is_weekly, source_channel
                    )
                    return

        # ── Regular news ──────────────────────────────────────────────────────
        content_hash = self._mem.hash_combined(text, image_data)

        if await self._mem.is_duplicate(content_hash):
            return
        if phash and await self._mem.is_image_duplicate(phash, max_distance=3):
            await self._mem.mark_image_seen(phash, source_channel)
            return

        # Lock BEFORE AI call — prevents race condition
        await self._mem.mark_seen(content_hash, source=source_channel)
        if phash:
            await self._mem.mark_image_seen(phash, source_channel)

        # ── Similarity check (free first, AI only when needed) ────────────────
        if await self._is_similar_to_recent(text, image_data, phash):
            return

        verdict = await self._ai_analyse(text, image_data, image_mime)
        if not verdict.get("approved"):
            return

        post_text = verdict.get("formatted_text", "").strip()
        if not post_text:
            return

        await asyncio.sleep(random.uniform(self._min_delay, self._max_delay))
        await self._simulate_typing(len(post_text))

        sent = await self._send_media(post_text, image_data, image_mime)
        if sent is None:
            return

        await self._mem.log_posted(
            source_channel, msg.id, sent.id, content_hash, verdict, post_text
        )
        await self._mem.store_recent_post(
            source_text=text[:1000], post_text=post_text[:1000], image_phash=phash
        )

    # ── Similarity — free checks first, AI only as last resort ───────────────

    async def _is_similar_to_recent(self, new_text: str, new_image: Optional[bytes],
                                    new_phash: Optional[str] = None) -> bool:
        """
        3-layer guard — each layer only runs if the previous didn't decide:

        Layer 1 — phash (zero cost, instant)
                  Catches identical or near-identical images.

        Layer 2 — keyword + token overlap (zero cost, instant)
                  Catches same event name (NfP, CPI …) posted twice today.
                  Also skips AI when texts are obviously unrelated
                  (overlap ratio < 0.25).

        Layer 3 — AI is_same_story (costs 1 Groq call)
                  Only fires when token overlap is in the grey zone (0.25–0.75),
                  meaning the texts share some words but aren't obviously the
                  same story. Capped at 3 AI calls per message so a burst of
                  50 similar posts can't drain the quota.
        """
        try:
            # Pull only 20 recent posts — was 150, 90 % quota reduction
            recent     = await self._mem.get_recent_posts(limit=20)
            today_date = _eat_now().date()
            ai_calls   = 0  # hard cap — never more than 3 AI calls here

            for old_text, old_phash, old_date_str in recent:
                old_date = datetime.fromisoformat(old_date_str).date()

                # ── Layer 1: phash (free) ─────────────────────────────────────
                if new_phash and old_phash:
                    if self._mem.hamming_distance(new_phash, old_phash) <= 3:
                        return True

                if not (old_text and new_text):
                    continue

                new_lower = new_text.lower()
                old_lower = old_text.lower()

                # ── Layer 2a: unique event name (free) ────────────────────────
                for name in _UNIQUE_EVENT_NAMES:
                    if name in new_lower and name in old_lower:
                        if old_date == today_date:
                            return True

                # ── Layer 2b: token overlap (free) ────────────────────────────
                ratio = _text_similarity_ratio(new_text, old_text)
                if ratio >= 0.75:
                    # Texts are very similar — treat as duplicate without AI
                    return True
                if ratio < 0.25:
                    # Texts are clearly different — skip AI entirely for this pair
                    continue

                # ── Layer 3: AI (costs quota) — only in grey zone 0.25–0.75 ──
                if ai_calls >= 3:
                    # Hard cap reached — stop checking, assume not duplicate
                    break
                same = await self._ai_is_same_story(
                    text_a=new_text[:500],
                    text_b=old_text[:500],
                    image_a=new_image,
                )
                ai_calls += 1
                if same:
                    return True

        except Exception as exc:
            log.error(f"Similarity check error: {exc}")
        return False

    # ── Reminders ─────────────────────────────────────────────────────────────

    def _select_vip_events(self, events: List[dict]) -> List[dict]:
        eligible = [e for e in events if _is_reminder_eligible(e)]
        return sorted(eligible, key=lambda e: e.get("time_24h", "99:99"))

    async def _check_reminders(self):
        today_str       = _eat_today_str()
        briefing_msg_id = await self._mem.get_daily_briefing_msg_id(today_str)
        if not briefing_msg_id or briefing_msg_id == -1:
            return

        vip_events = self._todays_vip
        if not vip_events:
            try:
                async with self._mem._db.execute(
                    "SELECT events_json FROM daily_briefings WHERE date_str=?",
                    (today_str,)
                ) as cur:
                    row = await cur.fetchone()
                if row and row["events_json"]:
                    all_events = json.loads(row["events_json"])
                    vip_events = self._select_vip_events(all_events)
                    self._todays_vip = vip_events
            except Exception:
                return
            if not vip_events:
                return

        now       = _eat_now()
        now_naive = now.replace(tzinfo=None)

        for event in vip_events:
            event_key = (
                f"{today_str}_{event.get('name', '')}_{event.get('currency', '')}"
            )
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

            if 14 <= minutes_until <= 16:
                await self._send_reminder(
                    event, event_key, briefing_msg_id, today_str
                )
                await asyncio.sleep(2)

    async def _send_reminder(self, event: dict, event_key: str,
                             reply_to_msg_id: int, today_str: str):
        event_name   = event.get("name", "Unknown Event")
        impact_emoji = "🔴" if event.get("impact") == "red" else "🟠"
        be_careful   = await self._ai_be_careful(event_name)

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
                await self._client.send_message(
                    dest, alert_text, parse_mode="md", reply_to=reply_to_msg_id
                )
            except Exception:
                pass
            await asyncio.sleep(1)

        await self._mem.mark_reminder_sent(event_key)
        await self._mem.increment_reminder_count(today_str)

    # ── FF image handler ──────────────────────────────────────────────────────

    async def _handle_ff_image(self, image_data: bytes, image_mime: str,
                               is_weekly: bool, source_channel: str):
        today_str     = _eat_today_str()
        today_display = _eat_today_display()
        phash         = self._mem.compute_phash(image_data)

        # Lock phash BEFORE AI call
        if phash:
            if await self._mem.is_image_duplicate(phash, max_distance=3):
                return
            await self._mem.mark_image_seen(phash, source_channel)

        if is_weekly:
            wkey   = _week_key()
            wrange = _week_range_str()

            if await self._mem.has_weekly_posted(wkey):
                return

            # Lock BEFORE AI
            await self._mem.save_weekly_posted(wkey)

            result = await self._ai_analyse_ff(
                image_data, image_mime,
                today_date=today_display,
                is_weekly=True,
                week_range=wrange,
            )
            if not result.get("approved"):
                await self._mem.delete_weekly_posted(wkey)
                return

            post_text = result.get("formatted_text", "").strip()
            if not post_text:
                await self._mem.delete_weekly_posted(wkey)
                return

            post_text = _add_signature(post_text)
            sent = await self._send_file(image_data, image_mime, post_text)
            if not sent:
                await self._mem.delete_weekly_posted(wkey)

        else:
            # Daily — once per day
            if await self._mem.has_daily_briefing(today_str):
                return

            # Lock with placeholder BEFORE AI
            await self._mem.save_daily_briefing(today_str, -1, [])

            result = await self._ai_analyse_ff(
                image_data, image_mime,
                today_date=today_display,
                is_weekly=False,
            )
            if not result.get("approved"):
                await self._mem.delete_daily_briefing(today_str)
                return

            raw_text = result.get("formatted_text", "").strip()
            if not raw_text:
                await self._mem.delete_daily_briefing(today_str)
                return

            events = _extract_events(raw_text)

            # Filter geopolitical from calendar
            events = [
                e for e in events
                if not any(
                    kw in e.get("name", "").lower()
                    for kw in GEOPOLITICAL_KEYWORDS
                )
            ]

            if not events:
                quiet = "🟢 No major news today — markets expected calm. Trade safe."
                quiet = _add_signature(quiet)
                sent  = await self._send_file(image_data, image_mime, quiet)
                if sent:
                    await self._mem.save_daily_briefing(today_str, sent.id, [])
                else:
                    await self._mem.delete_daily_briefing(today_str)
                return

            self._todays_vip = self._select_vip_events(events)

            has_red   = any(e["impact"] == "red" for e in events)
            title     = (
                "TODAY'S USD 🇺🇸 HIGH IMPACT NEWS" if has_red
                else "TODAY'S USD 🇺🇸 NEWS"
            )
            date_line = _eat_date_line()

            lines = [title, "", date_line, ""]
            for e in events:
                emoji = "🔴" if e["impact"] == "red" else "🟠"
                lines.append(
                    f"{emoji} {e['time_12h']} | {e['currency']}: {e['name']}"
                )

            if has_red:
                lines.append("")
                lines.append("Be careful during these releases.")

            post_text = "\n".join(lines)
            post_text = _add_signature(post_text)

            sent = await self._send_file(image_data, image_mime, post_text)
            if sent:
                await self._mem.save_daily_briefing(today_str, sent.id, events)
            else:
                await self._mem.delete_daily_briefing(today_str)

    async def _simulate_typing(self, text_len: int):
        duration = min(max(text_len / 180, 2), 14)
        if self._dest_channels:
            try:
                async with self._client.action(self._dest_channels[0], "typing"):
                    await asyncio.sleep(duration)
            except Exception:
                pass
