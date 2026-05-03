"""
main.py — AXIOM INTEL Telegram Manager
Institutional Senior Trader Edition

What this bot does:
  • Watches ALL source channels for new messages
  • Approves only real geopolitical/macro news — blocks memes, TA, charts, opinions
  • Double duplicate protection: hash + AI similarity
  • ForexFactory calendar image posted manually → bot posts daily briefing (max 1/day)
  • Weekly FF image posted Sunday → weekly high impact news post
  • 15-min reminders for VIP USD events (FOMC, NFP, CPI, GDP, PCE, Powell)
  • Every post ends with [Squad 4xx](https://t.me/Squad_4xx)
  • Posts to ALL destination channels simultaneously
  • No forecast, no previous, no NOTE line anywhere
  • Times: 12-hour AM/PM EAT (GMT+3)
  • /stop /start /status commands via Saved Messages

Environment variables:
  Required:
    TELEGRAM_API_ID       — from my.telegram.org
    TELEGRAM_API_HASH     — from my.telegram.org
    SESSION_STRING        — from generate_session.py (preferred)
    SOURCE_CHANNELS       — comma-separated: @ch1,@ch2,@ch3
    DEST_CHANNELS         — comma-separated: @Squad_4xx,@ch2
    GEMINI_API_KEY        — Gemini 2.5 Flash key
    GROQ_API_KEY          — Groq key (fallback)

  Optional:
    OWNER_ID              — your Telegram user ID (for /stop /start /status)
    DEST_CHANNEL          — legacy single-channel fallback
    TELEGRAM_PHONE        — phone if not using StringSession
    SESSION_NAME          — file session name (default: manager_session)
    CHANNEL_CATEGORY      — focus description injected into AI prompt
    POLL_INTERVAL         — seconds between scrape cycles (default: 60)
    MIN_DELAY             — min seconds before posting (default: 8)
    MAX_DELAY             — max seconds before posting (default: 30)
    LOOKBACK_HOURS        — how far back on first run (default: 2)
    DB_PATH               — SQLite path (default: memory.db)
    HASH_TTL_DAYS         — days to keep hashes (default: 30)
    SUPABASE_DB           — Supabase PostgreSQL connection string
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime

import pytz
from dotenv import load_dotenv

load_dotenv()

from scraper import ChannelScraper
from ai_engine import AIEngine
from memory import MemoryManager

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("main")

EAT = pytz.timezone("Africa/Addis_Ababa")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _require(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        log.error(f"❌  Missing required env var: {key}")
        sys.exit(1)
    return val


def _parse_dest_channels() -> list:
    """
    Parse destination channels.
    DEST_CHANNELS=@ch1,@ch2,@ch3  ← preferred (comma-separated)
    DEST_CHANNEL=@ch1              ← legacy fallback
    Both can coexist — deduped automatically.
    """
    raw_multi = os.environ.get("DEST_CHANNELS", "").strip()
    channels = [c.strip() for c in raw_multi.split(",") if c.strip()] if raw_multi else []

    raw_single = os.environ.get("DEST_CHANNEL", "").strip()
    if raw_single and raw_single not in channels:
        channels.append(raw_single)

    if not channels:
        log.error("❌  No destination channels. Set DEST_CHANNELS or DEST_CHANNEL.")
        sys.exit(1)

    log.info(f"📤  Destination channels ({len(channels)}): {channels}")
    return channels


# ─── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    "api_id":         int(_require("TELEGRAM_API_ID")),
    "api_hash":       _require("TELEGRAM_API_HASH"),
    "phone":          os.getenv("TELEGRAM_PHONE", ""),
    "session_string": os.getenv("SESSION_STRING", ""),
    "session_name":   os.getenv("SESSION_NAME", "manager_session"),

    # Source channels — all watched, FF image accepted from any of them
    "source_channels": [
        c.strip() for c in _require("SOURCE_CHANNELS").split(",") if c.strip()
    ],

    # Destination channels — posts go to all of them
    "dest_channels": _parse_dest_channels(),

    "gemini_api_key": _require("GEMINI_API_KEY"),
    "groq_api_key":   _require("GROQ_API_KEY"),

    "channel_category": os.getenv(
        "CHANNEL_CATEGORY",
        "Geopolitical events (wars, sanctions, elections), "
        "Central Bank policy (FED, ECB, BOE, BOJ), "
        "Macroeconomic data (CPI, NFP, GDP, PCE), "
        "Gold (XAU) safe-haven flows, "
        "Oil (WTI/Brent) supply disruptions, "
        "Major FX pairs and USD flows. "
        "NO trading signals. NO technical analysis. NO memes. NO opinions.",
    ),

    "poll_interval_seconds": int(os.getenv("POLL_INTERVAL", "60")),
    "min_delay_seconds":     float(os.getenv("MIN_DELAY", "8")),
    "max_delay_seconds":     float(os.getenv("MAX_DELAY", "30")),
    "lookback_hours":        int(os.getenv("LOOKBACK_HOURS", "2")),
    "db_path":               os.getenv("DB_PATH", "memory.db"),
    "hash_ttl_days":         int(os.getenv("HASH_TTL_DAYS", "30")),
}

# Owner Telegram user ID — for /stop /start /status via saved messages
OWNER_ID = int(os.getenv("OWNER_ID", "0"))


# ─── Graceful shutdown ─────────────────────────────────────────────────────────
_shutdown = asyncio.Event()
_bot_running = True   # controls whether poll/reminder loops are active


def _handle_signal(sig, _frame):
    log.info(f"Signal {sig.name} — shutting down …")
    _shutdown.set()


signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ─── Command listener — saved messages ────────────────────────────────────────
async def command_listener(scraper: ChannelScraper, memory: MemoryManager):
    global _bot_running

    if not OWNER_ID:
        log.info("ℹ️  OWNER_ID not set — /stop /start /status disabled")
        return

    log.info(f"🎮  Command listener active (owner_id={OWNER_ID})")
    last_id = 0
    started_at = datetime.now(EAT)

    while not _shutdown.is_set():
        try:
            async for msg in scraper._client.iter_messages("me", limit=5):
                if msg.id <= last_id:
                    continue
                last_id = max(last_id, msg.id)
                text = (msg.text or "").strip().lower()
                if not text.startswith("/"):
                    continue

                log.info(f"📨  Command received: {text!r}")

                if text == "/stop":
                    _bot_running = False
                    await scraper._client.send_message(
                        "me", "⏸️ Bot stopped.\nSend /start to resume."
                    )

                elif text == "/start":
                    _bot_running = True
                    await scraper._client.send_message(
                        "me", "▶️ Bot started.\nScraping resumed."
                    )

                elif text == "/status":
                    stats = await memory.stats()
                    uptime = datetime.now(EAT) - started_at
                    h, rem = divmod(int(uptime.total_seconds()), 3600)
                    m = rem // 60
                    state = "▶️ Running" if _bot_running else "⏸️ Stopped"
                    db_mode = stats.get("db_mode", "sqlite")
                    status = (
                        f"📊 AXIOM INTEL STATUS\n\n"
                        f"State: {state}\n"
                        f"Uptime: {h}h {m}m\n"
                        f"Hashes tracked: {stats['tracked_hashes']}\n"
                        f"Posts (24h): {stats['posted_last_24h']}\n"
                        f"Database: {db_mode}\n"
                        f"Time (EAT): {datetime.now(EAT).strftime('%H:%M:%S')}"
                    )
                    await scraper._client.send_message("me", status)

                else:
                    await scraper._client.send_message(
                        "me",
                        "❓ Unknown command.\n\nAvailable:\n/start\n/stop\n/status"
                    )

        except Exception as exc:
            log.warning(f"Command listener error: {exc}")

        try:
            await asyncio.wait_for(_shutdown.wait(), timeout=15)
        except asyncio.TimeoutError:
            pass


# ─── Poll loop ─────────────────────────────────────────────────────────────────
async def poll_loop(scraper: ChannelScraper):
    log.info("✅  Poll loop started.")
    interval = CONFIG["poll_interval_seconds"]
    while not _shutdown.is_set():
        if _bot_running:
            try:
                await scraper.poll_and_forward()
            except Exception as exc:
                log.error(f"Poll cycle error: {exc}", exc_info=True)
        else:
            log.debug("Bot paused — skipping poll")
        try:
            await asyncio.wait_for(_shutdown.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


# ─── Reminder loop ─────────────────────────────────────────────────────────────
async def reminder_loop(scraper: ChannelScraper):
    log.info("🔔  Reminder loop started.")
    while not _shutdown.is_set():
        if _bot_running:
            try:
                await scraper._check_reminders()
            except Exception as exc:
                log.error(f"Reminder loop error: {exc}", exc_info=True)
        try:
            await asyncio.wait_for(_shutdown.wait(), timeout=60)
        except asyncio.TimeoutError:
            pass
    log.info("🔔  Reminder loop stopped.")


# ─── Main ──────────────────────────────────────────────────────────────────────
async def run():
    log.info("🚀  AXIOM INTEL starting …")
    log.info(f"📡  Watching {len(CONFIG['source_channels'])} source channel(s)")
    log.info(f"📤  Posting to {len(CONFIG['dest_channels'])} destination channel(s)")
    log.info("🔒  Duplicate protection: hash + AI similarity")
    log.info("📌  Signature: [Squad 4xx](https://t.me/Squad_4xx)")
    log.info("🚫  Blocking: memes, TA charts, analysis images, opinions, duplicates")

    if CONFIG["session_string"]:
        log.info("🔑  Auth: StringSession ✅")
    else:
        log.info("🔑  Auth: File session")

    memory = MemoryManager(db_path=CONFIG["db_path"], ttl_days=CONFIG["hash_ttl_days"])
    await memory.init()

    ai = AIEngine(
        gemini_key=CONFIG["gemini_api_key"],
        groq_key=CONFIG["groq_api_key"],
        channel_category=CONFIG["channel_category"],
    )

    scraper = ChannelScraper(config=CONFIG, ai_engine=ai, memory=memory)
    await scraper.start()

    try:
        await asyncio.gather(
            poll_loop(scraper),
            reminder_loop(scraper),
            command_listener(scraper, memory),
        )
    finally:
        log.info("🛑  Shutting down …")
        await scraper.stop()
        await memory.close()
        log.info("👋  Done.")


if __name__ == "__main__":
    asyncio.run(run())
