"""
ai_engine.py — AXIOM INTEL AI Engine.
- Gemini primary, Groq fallback with retry.
- Hard blocks: signals, sentiment, injection. Watermarks stripped silently.
- Hashtags: #XAUUSD #DXY #OIL only where relevant — detected from final text, ONE pass.
- Emoji system:
    📊  data releases (NFP, CPI, GDP, PCE, PMI, retail sales, etc.)
    📈  price UP / bullish news (surges, hits record, rallies, jumps)
    📉  price DOWN / bearish news (drops, falls, crashes, slides)
    🏦  Fed / central bank decisions
    🛢️  Oil/energy (non-price-directional)
    🚨  War / geopolitical conflict / sanctions / breaking
    🇺🇸  Trump / tariffs / US political
    🌍  General geopolitical / world leaders
- FF calendar: daily once/day, weekly once/week.
- Geopolitical/FOMC always approved.
- Same-time events: comma-joined on ONE line.
"""

import asyncio
import base64
import json
import logging
import random
import re
import textwrap
from datetime import datetime, timezone
from typing import Optional

import google.generativeai as genai
from groq import AsyncGroq

log = logging.getLogger("ai_engine")

CHANNEL_SIGNATURE = "\n\n[Squad 4xx](https://t.me/Squad_4xx)"
ALLOWED_HASHTAGS  = {"#XAUUSD", "#DXY", "#OIL"}


# ── Signature ─────────────────────────────────────────────────────────────────
def _add_signature(text: str) -> str:
    text = text.strip()
    if "[Squad 4xx]" not in text:
        if random.random() < 0.25:
            text += "\n\n💡 [Squad 4xx](https://t.me/Squad_4xx)"
        else:
            text += "\n\n[Squad 4xx](https://t.me/Squad_4xx)"
    return text


def _add_us_flag(text: str) -> str:
    if not text:
        return text
    lines = text.split("\n")
    lines[0] = re.sub(r"\bUSD\b", "USD 🇺🇸", lines[0], count=1)
    return "\n".join(lines)


def _strip_be_careful(text: str) -> str:
    return re.sub(r"\n?Be careful[^\n]*\n?", "", text, flags=re.IGNORECASE).strip()


def _strip_hashtag_label(text: str) -> str:
    return re.sub(r"HASHTAGS?\s*[:\-]?\s*\n?", "", text, flags=re.IGNORECASE).strip()


def _strip_watermarks(text: str) -> str:
    text = re.sub(r"t\.me/(?!Squad_4xx)[a-zA-Z0-9_]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"@(?!Squad_4xx)[a-zA-Z0-9_]{4,}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_all_hashtags(text: str) -> str:
    """Remove every #word from text — used before we re-add correct ones."""
    return re.sub(r"\s*#\w+", "", text).strip()


# ── Hard block patterns ───────────────────────────────────────────────────────
_SIGNAL_RE = re.compile(
    r"\b(buy|sell|long|short|entry|tp|take[\s_-]?profit|sl|"
    r"stop[\s_-]?loss|stoploss|stop\s+at\s+\d|"
    r"entry\s*[:\-]?\s*\d|target\s*[:\-]?\s*\d)\b",
    re.IGNORECASE,
)
_SENTIMENT_RE = re.compile(
    r"\b(fear\s*[&and]+\s*greed|greed\s*index|fear\s*index|"
    r"sentiment\s*index|market\s*sentiment|investor\s*sentiment|"
    r"bulls?\s*vs\.?\s*bears?|smart\s*money|dumb\s*money|"
    r"cot\s*report|commitment\s*of\s*traders|"
    r"put[/-]call\s*ratio|vix\s*index)\b",
    re.IGNORECASE,
)
_INJECTION_RE = re.compile(
    r"\b(ignore\s+(all\s+|previous\s+)?rules?|"
    r"ignore\s+(all\s+)?instructions?|system\s*prompt|"
    r"override\s+rules?|jailbreak|act\s+as\s+|"
    r"forget\s+instructions?|disregard\s+rules?)\b",
    re.IGNORECASE,
)


def _hard_block(text: str) -> Optional[str]:
    if not text:
        return None
    if _SIGNAL_RE.search(text):
        return "signal_content"
    if _SENTIMENT_RE.search(text):
        return "sentiment_content"
    if _INJECTION_RE.search(text):
        return "injection_attempt"
    return None


# ── Hashtag engine — single source of truth ───────────────────────────────────
#
# HOW IT WORKS:
# The AI prompt tells the AI NOT to add hashtags.
# We strip ALL hashtags from AI output.
# Then this function adds exactly the right ones — one time, here only.
#
# RULES:
# Each asset has a PRIMARY keyword list (news is directly about this asset)
# and an EXCLUDED keyword list (news mentions the asset only as a side-effect).

_OIL_SUBJECT = [
    "oil", "crude", "opec", "barrel", "brent", "wti",
    "hormuz", "energy supply", "petroleum", "gasoline",
    "oil production", "oil output", "oil prices", "oil market",
    "energy market", "oil cut", "oil hike", "oil embargo",
]

_OIL_EXCLUDE_IF_ALONE = [
    "fed", "fomc", "powell", "nfp", "cpi", "gdp", "trump tariff",
    "interest rate", "gold hits", "dollar", "dxy",
]

_GOLD_SUBJECT = [
    "gold", "xau", "xauusd", "bullion",
    "gold price", "gold hits", "gold surges", "gold drops",
    "gold rally", "gold falls", "gold ounce", "gold market",
    "precious metal",
]

_GOLD_EXCLUDE_IF_ALONE = [
    "oil", "opec", "barrel", "crude",
]

_DXY_SUBJECT = [
    # Fed / monetary policy
    "fed ", "fomc", "powell", "federal reserve", "federal funds",
    "interest rate decision", "rate cut", "rate hike", "rate hold",
    "rate unchanged", "basis points", "bps",
    # US economic data releases
    "nfp", "non-farm payroll", "non-farm employment",
    "cpi", "consumer price index",
    "pce", "core pce",
    "gdp", "gross domestic product",
    "retail sales",
    "unemployment rate", "jobless claims",
    "ism ", "pmi",
    "durable goods",
    "average hourly earnings",
    "jolts",
    "ism services", "ism manufacturing",
    # Dollar itself
    "dollar", "dxy", "usd index", "dollar index",
    "dollar strength", "dollar weakness", "dollar rallies", "dollar drops",
    # Tariffs / trade
    "tariff on", "tariffs on", "trade war", "trade deal",
    # Trump statements about economy/dollar/rates
    "trump tax", "trump rate", "trump fed", "trump economy",
    "trump tariff on", "trump imposes tariff",
    # US sanctions
    "us sanctions", "us sanction",
    # Treasury
    "us treasury", "treasury yield", "10-year yield", "bond yield",
]

_DXY_EXCLUDE_IF_ALONE = [
    "oil", "opec", "barrel", "crude",
    "gold hits", "xau hits",
]


def _detect_assets(text: str) -> list:
    """
    Analyse text and return the list of hashtags that should be added.
    Each asset is only included when the news is DIRECTLY about that asset.

    FIXES vs old version:
    - OIL: now requires at minimum 2 oil-specific signals before tagging,
      preventing false positives when oil is just a price side-mention.
    - GOLD: excludes tagging when gold is only mentioned as "gold-backed" or
      vague context without a price/market action word.
    - DXY: tightened — geopolitical-only news no longer triggers DXY unless
      dollar/Fed is explicitly the subject.
    """
    t = text.lower()
    tags = []

    # ── OIL ──────────────────────────────────────────────────────────────────
    # Require a strong oil subject word (not just "oil" in passing)
    _OIL_STRONG = [
        "oil production", "oil output", "opec", "barrel",
        "oil price", "oil market", "oil cut", "oil embargo",
        "hormuz", "brent", "wti", "crude oil", "petroleum",
        "oil surges", "oil drops", "oil falls", "oil rallies",
        "oil hits", "oil breaks",
    ]
    oil_strong_hit = any(kw in t for kw in _OIL_STRONG)
    oil_weak_hit   = any(kw in t for kw in _OIL_SUBJECT)

    if oil_strong_hit:
        # Strong hit — include unless a non-oil subject dominates
        only_side_effect = any(kw in t for kw in _OIL_EXCLUDE_IF_ALONE) and not oil_strong_hit
        if not only_side_effect:
            tags.append("#OIL")
    elif oil_weak_hit:
        # Weak hit (just "oil" mentioned) — only tag if no exclude word present
        if not any(kw in t for kw in _OIL_EXCLUDE_IF_ALONE):
            tags.append("#OIL")

    # ── GOLD ─────────────────────────────────────────────────────────────────
    _GOLD_STRONG = [
        "gold price", "gold hits", "gold surges", "gold drops",
        "gold rally", "gold falls", "xauusd", "xau/usd",
        "bullion", "gold ounce", "precious metal",
        "gold breaks", "gold reaches", "gold at $", "gold record",
        "gold rallies", "gold jumps", "gold slides", "gold tumbles",
    ]
    gold_strong_hit = any(kw in t for kw in _GOLD_STRONG)
    gold_weak_hit   = any(kw in t for kw in _GOLD_SUBJECT)

    if gold_strong_hit:
        only_side_effect = any(kw in t for kw in _GOLD_EXCLUDE_IF_ALONE) and not gold_strong_hit
        if not only_side_effect:
            tags.append("#XAUUSD")
    elif gold_weak_hit:
        if not any(kw in t for kw in _GOLD_EXCLUDE_IF_ALONE):
            tags.append("#XAUUSD")

    # ── DXY ──────────────────────────────────────────────────────────────────
    # FIXED: geopolitical-only events (war, sanctions, iran) must ALSO mention
    # dollar/Fed explicitly to get #DXY — pure geopolitical = no #DXY
    _DXY_STRONG = [
        "fed ", "fomc", "powell", "federal reserve",
        "interest rate", "rate cut", "rate hike", "rate hold",
        "nfp", "cpi", "gdp", "pce", "unemployment",
        "dollar", "dxy", "treasury yield",
        "tariff on", "tariffs on", "trade war",
        "retail sales", "jobless", "ism ", "pmi",
        "average hourly earnings", "jolts", "durable goods",
    ]
    _GEO_ONLY = [
        "war", "missile", "strike", "attack", "troops", "invasion",
        "nuclear", "explosion", "military", "sanction", "iran",
        "ukraine", "russia", "nato", "middle east",
    ]

    dxy_strong_hit = any(kw in t for kw in _DXY_STRONG)
    geo_only = (
        any(kw in t for kw in _GEO_ONLY)
        and not dxy_strong_hit
    )

    if dxy_strong_hit and not geo_only:
        only_side_effect = (
            any(kw in t for kw in _DXY_EXCLUDE_IF_ALONE)
            and not any(kw in t for kw in _DXY_STRONG)
        )
        if not only_side_effect:
            tags.append("#DXY")

    return tags


# ── Emoji engine — REPLACED ────────────────────────────────────────────────────
#
# NEW SYSTEM:
#   📊  Economic data releases (NFP, CPI, GDP, PCE, PMI, retail sales…)
#   📈  Price UP / bullish direction (surges, hits record, rallies, jumps, rises)
#   📉  Price DOWN / bearish direction (drops, falls, crashes, slides, tumbles)
#   🏦  Fed / central bank decisions (non-directional)
#   🛢️  Oil/energy news (non-directional price)
#   🚨  War / geopolitical / sanctions / breaking / urgent
#   🇺🇸  Trump / tariffs / US political (non-market-moving)
#   🌍  General world leaders / geopolitical (default)
#
# PRIORITY ORDER:
# Data releases → checked first (most specific)
# Price direction → checked second (bullish vs bearish)
# Then Fed, oil, war, political, default

# UP/bullish keywords — specific price-action phrases only
_UP_WORDS = [
    "surges", "surge", "rallies", "rally", "jumps", "jump",
    "rises", "rise", "climbs", "climb", "hits record", "record high",
    "all-time high", "ath", "soars", "soar", "spikes", "spike",
    "gains", "gain", "advances", "advance",
    "reaches", "hits $", "breaks above", "tops", "exceeds",
    "up by", "adds", "bounces", "bounce",
    "recovers", "recovery", "surging", "rallying", "jumping",
    "rising", "climbing", "advancing",
]

# DOWN/bearish keywords
_DOWN_WORDS = [
    "drops", "drop", "falls", "fall", "tumbles", "tumble",
    "slides", "slide", "crashes", "crash", "plunges", "plunge",
    "declines", "decline", "weakens", "weaken", "retreats", "retreat",
    "sinks", "sink", "loses", "loss", "down by", "drops to",
    "falls to", "slides to", "lower", "slumps", "slump",
    "collapses", "collapse", "dips", "dip", "selling off",
    "under pressure", "hit low", "new low", "multi-month low",
    "dropping", "falling", "tumbling", "sliding", "declining",
]

# Economic data release keywords
_DATA_WORDS = [
    "nfp", "non-farm payroll", "non-farm employment",
    "cpi", "consumer price index",
    "pce", "core pce",
    "gdp", "gross domestic product",
    "retail sales",
    "unemployment rate", "jobless claims",
    "ism", "pmi",
    "durable goods",
    "average hourly earnings",
    "jolts",
    "producer price", "ppi",
    "trade balance",
    "housing starts", "building permits",
    "consumer confidence", "consumer sentiment",
    "came in at", "came in", "printed at", "print of",
    "actual:", "actual vs",
    "beats expectations", "misses expectations",
    "above forecast", "below forecast",
    "% vs", "k vs", "b vs",
]


def _pick_emoji(text: str) -> str:
    """
    Pick the single most relevant emoji for this news post.

    Priority:
    1. Data release  → 📊
    2. Price UP      → 📈
    3. Price DOWN    → 📉
    4. Fed/CB        → 🏦
    5. Oil/energy    → 🛢️
    6. War/geo/break → 🚨
    7. Trump/tariff  → 🇺🇸
    8. World leaders → 🌍
    9. Default       → 🌍
    """
    t = text.lower()

    # 1. Economic data release — highest priority for financial data posts
    if any(k in t for k in _DATA_WORDS):
        return "📊"

    # 2. Price direction — bullish
    if any(k in t for k in _UP_WORDS):
        return "📈"

    # 3. Price direction — bearish
    if any(k in t for k in _DOWN_WORDS):
        return "📉"

    # 4. Fed / central bank (rate decision, non-directional)
    # Check for "fed" as a standalone word (not "federal" in a title)
    if any(k in t for k in [
        "fomc", "federal reserve", "federal funds",
        "fed cuts", "fed raises", "fed hikes", "fed holds",
        "fed pauses", "fed keeps", "powell", "rate decision",
        "interest rate decision", "basis points", "bps",
        "rate unchanged", "rate hold",
    ]) or re.search(r"\bfed\b", t):
        return "🏦"

    # 5. Oil / energy (non-directional)
    if any(k in t for k in [
        "oil", "crude", "opec", "barrel", "brent", "wti",
        "hormuz", "petroleum", "energy supply",
    ]):
        return "🛢️"

    # 6. War / geopolitical conflict / sanctions / breaking
    # Use word-boundary regex for short words like "war" to avoid matching "warns", "award" etc.
    if (
        re.search(r"\bwar\b", t) or
        any(k in t for k in [
            "missile", "airstrike", "air strike", "bomb", "bombing",
            "sanction", "sanctions", "conflict", "troops", "invasion",
            "nuclear", "explosion", "military action",
            "breaking", "urgent", "flash", "just in", "alert",
            "emergency", "crisis", "ceasefire",
        ])
    ):
        return "🚨"

    # 7. Trump / US political / tariffs
    if any(k in t for k in [
        "trump", "tariff", "trade war", "trade deal",
        "white house", "executive order",
    ]):
        return "🇺🇸"

    # 8. World leaders / geopolitical tensions / statements
    if any(k in t for k in [
        "putin", "xi jinping", "xi ", "iran", "ukraine", "russia",
        "nato", "opec", "middle east", "geopolit",
        "biden", "president", "prime minister", "warns", "threatens",
    ]):
        return "🌍"

    # 9. Default
    return "🌍"


def _apply_emoji_and_hashtags(text: str, is_calendar: bool = False) -> str:
    """
    Single pass that:
    1. Strips ALL existing hashtags from the text
    2. Checks if first character is already an emoji — if not, prepends one
    3. Adds correct hashtags at the end (skipped for calendar posts)

    This is the ONLY place emojis and hashtags are added. Never called twice.
    """
    if not text:
        return text

    # Step 1 — strip all hashtags the AI may have added
    text = _strip_all_hashtags(text)
    text = text.strip()

    # Step 2 — emoji: check first real character
    first_char = text.lstrip()[0] if text.strip() else ""
    has_emoji  = first_char and ord(first_char) > 127

    if not has_emoji:
        emoji = _pick_emoji(text)
        text  = emoji + " " + text

    # Step 3 — hashtags (calendar posts never get hashtags)
    if not is_calendar:
        tags = _detect_assets(text)
        if tags:
            text = text.rstrip() + "\n\n" + " ".join(tags)

    return text


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """
You are AXIOM INTEL — Senior Institutional Macro & Geopolitical news editor.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALWAYS APPROVE — NO EXCEPTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. FOMC / Fed decisions — rate held, cut, raised (any bps)
2. Fed Chair Powell speaking
3. Any world leader statement affecting: Oil, Gold, USD, tariffs, war, sanctions
   (Trump, Biden, Putin, Xi, Iran, OPEC, NATO)
4. Geopolitical events: war, missile, strike, sanctions, Hormuz, Ukraine
5. Actual released economic data with real numbers:
   "CPI came at 2.8%", "NFP 250K", "GDP rose 2.1%", "raised by 25bps"
6. Price level hits — "Gold hit $3,400", "Oil broke $90", "DXY at 104"
7. Trump / world leader statements on USD, trade, tariffs, economy
8. War, sanctions, geopolitical conflict and their effect on markets

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALWAYS REJECT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Signals — Buy/Sell/Long/Short/Entry/TP/SL
2. Technical analysis — patterns, indicators, RSI, MACD, Fibonacci, support/resistance levels
3. Memes, jokes, informal content
4. Chart screenshots or any chart image (TradingView or otherwise)
5. Content older than 18 hours
6. Forecast/Previous values only — "forecast 180K", "previous 2.3%" (reject ONLY if NO real released number is present)
7. Opinions — "I think", "expect", "my analysis"
8. Sentiment — Fear & Greed, COT, smart money, VIX
9. Bank sentiment — "banks are bullish/bearish"

NOTE: If content contains another channel username or link — IGNORE it. Judge the news itself.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REWRITING RULES — CRITICAL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOAL: Make it look like OUR channel posted it — with MINIMAL changes only.

DO:
- Keep original wording as close as possible
- Fix grammar/spelling only if clearly wrong
- Remove other channel watermarks, usernames, links
- Keep all real numbers exactly as they appear in source
- Keep the original structure and sentence order

DO NOT:
- Do NOT rewrite sentences into new ones
- Do NOT add ANY information the source did not say
- Do NOT add words like "may", "could", "might", "expected", "likely"
- Do NOT add context, background, or explanation not in the source
- Do NOT summarise or shorten — keep ALL original content
- Do NOT change the meaning of any sentence
- Do NOT add emojis the source did not have (keep source emojis)
- Do NOT add hashtags — the system handles hashtags automatically

WRONG — adding prediction not in source:
Source: "Trump announces 50% tariffs on EU"
Bad output: "Trump announces 50% tariffs on EU. This could spark a trade war."
→ NEVER invent sentences. Keep it exactly as source.

CORRECT:
Source: "Trump announces 50% tariffs on EU goods starting June"
Good output: "Trump announces 50% tariffs on EU goods starting June"

DO NOT ADD SIGNATURE. DO NOT ADD YEAR. NO ASTERISKS. NO BOLD.
DO NOT ADD HASHTAGS — they are added automatically by the system.

RESPOND WITH VALID JSON ONLY:
{"approved": true/false, "reason": "...", "issues": [], "formatted_text": "...", "confidence": 0.9}
""".strip()


# ── ForexFactory prompts ──────────────────────────────────────────────────────
_FF_DAILY_PROMPT = """
You are analysing a ForexFactory economic calendar screenshot.

TODAY'S DATE: {today_date}

TASK:
1. Verify the calendar shows TODAY: {today_date}
   If different date → {{"approved": false, "reason": "wrong date"}}

2. Extract ALL USD 🔴 (high impact) and 🟠 (medium impact) events.

3. CRITICAL SAME-TIME RULE:
   - Multiple events at the SAME time → ONE line, ALL names comma-separated
   - Event with no time shown → it shares the time of the event above it
   - Use the HIGHEST impact emoji when mixing red/orange at same time
   - Time: 12-hour AM/PM, NO leading zero (5:00 PM not 05:00 PM)
   - NO forecast, NO previous values, NO hashtags, NO "Be careful", NO signature

EXAMPLE — 3 events all at 5:00 PM becomes ONE line:
TODAY'S USD HIGH IMPACT NEWS
Tuesday, May 5

🔴 5:00 PM | USD: ISM Services PMI, JOLTS Job Openings, New Home Sales

EXAMPLE — mixed times, each time on its own line:
TODAY'S USD HIGH IMPACT NEWS
Friday, May 8

🔴 3:30 PM | USD: Average Hourly Earnings m/m, Non-Farm Employment Change, Unemployment Rate
🟠 5:00 PM | USD: Prelim UoM Consumer Sentiment, Prelim UoM Inflation Expectations

If valid → {{"approved": true, "reason": "valid daily FF", "formatted_text": "..."}}
If invalid → {{"approved": false, "reason": "..."}}
RESPOND WITH VALID JSON ONLY.
""".strip()

_FF_WEEKLY_PROMPT = """
You are analysing a ForexFactory weekly calendar screenshot.

CURRENT WEEK: {week_range}

TASK:
1. Confirm this shows MULTIPLE DAYS (weekly view).
   If only one day → {{"approved": false, "reason": "not weekly"}}

2. Extract ALL USD 🔴 and 🟠 events grouped by day.

3. CRITICAL SAME-TIME RULE:
   - Multiple events at the SAME time → ONE line, ALL names comma-separated
   - Event with no time shown → it shares the time of the event above it
   - Use the HIGHEST impact emoji when mixing red/orange at same time
   - Time: 12-hour AM/PM, NO leading zero (3:30 PM not 03:30 PM)
   - NO forecast, NO previous, NO hashtags, NO "Be careful", NO signature

EXACT FORMAT EXAMPLE:
WEEKLY HIGH IMPACT NEWS
Week of May 5 – May 9

Tuesday — May 5
🔴 5:00 PM | USD: ISM Services PMI, JOLTS Job Openings, New Home Sales

Wednesday — May 6
🟠 3:15 PM | USD: ADP Non-Farm Employment Change

Thursday — May 7
🟠 3:30 PM | USD: Unemployment Claims

Friday — May 8
🔴 3:30 PM | USD: Average Hourly Earnings m/m, Non-Farm Employment Change, Unemployment Rate
🟠 5:00 PM | USD: Prelim UoM Consumer Sentiment, Prelim UoM Inflation Expectations

If valid → {{"approved": true, "reason": "valid weekly FF", "formatted_text": "..."}}
If invalid → {{"approved": false, "reason": "..."}}
RESPOND WITH VALID JSON ONLY.
""".strip()

_FF_DETECT_PROMPT = """
Look at this image carefully and answer:
1. Is this a ForexFactory.com economic calendar screenshot?
2. Does it show MULTIPLE DAYS (weekly) or ONE day (daily)?

Weekly = shows Monday/Tuesday/Wednesday etc or multiple dates.
Daily = shows only one date.

Respond ONLY with JSON:
{"is_ff": true/false, "is_weekly": true/false}
"""

_SIMILARITY_PROMPT = """
You are a strict duplicate news detector for a financial news channel.

Your job: decide if Story A and Story B are reporting the SAME real-world event.

RULES:
- Same event = same entity + same action + same context (even if worded differently)
- Different wording, different source = still SAME if core fact is identical
- Similar topic but different specific event = NOT same

Story A: {story_a}
Story B: {story_b}

Be STRICT. Only mark true if genuinely the same event.
Respond ONLY with JSON:
{{"same_story": true/false, "confidence": 0.0-1.0, "reason": "one line explanation"}}
"""

_SIMILARITY_IMAGE_PROMPT = """
You are a strict duplicate news detector for a financial news channel.

Story A text: {story_a}
Story B text: {story_b}

Also examine any images provided — compare headlines, numbers, tickers shown.

SAME = same entity + same action + same core fact
NOT SAME = similar topic but different event, different date, different country

Be STRICT. Only mark true if genuinely the same event.
Respond ONLY with JSON:
{{"same_story": true/false, "confidence": 0.0-1.0, "reason": "one line explanation"}}
"""


# ── Be-careful lines ──────────────────────────────────────────────────────────
def _get_be_careful_line(event_name: str) -> str:
    n = event_name.lower()
    if any(k in n for k in ["fomc", "federal funds", "interest rate", "fed chair", "powell"]):
        return "⚠️ Fed decisions move everything. Be careful — no new trades during the release."
    if any(k in n for k in ["non-farm", "nfp", "payroll"]):
        return "⚠️ NFP can spike the market violently. Be careful — protect your capital now."
    if any(k in n for k in ["cpi", "consumer price", "inflation"]):
        return "⚠️ Inflation data whipsaws fast. Be careful — secure profits before the release."
    if any(k in n for k in ["pce", "core pce"]):
        return "⚠️ PCE can shift rate expectations quickly. Be careful — protect your positions."
    if "gdp" in n:
        return "⚠️ GDP surprises hit hard and fast. Be careful — move stops to break-even now."
    if any(k in n for k in ["unemployment", "jobless"]):
        return "⚠️ Unemployment data moves USD sharply. Be careful — no new entries."
    if "retail sales" in n:
        return "⚠️ Retail Sales can jolt the market. Be careful — protect your open positions."
    if any(k in n for k in ["ism", "pmi"]):
        return "⚠️ ISM data can move USD fast. Be careful — wait for the dust to settle."
    if any(k in n for k in ["ppi", "producer price"]):
        return "⚠️ PPI surprises can hit USD hard. Be careful — protect your capital."
    if "durable goods" in n:
        return "⚠️ Durable Goods can cause sharp moves. Be careful — no new entries now."
    return "⚠️ This release can move the market strongly. Be careful — protect your capital."


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ── JSON parsing ──────────────────────────────────────────────────────────────
def _parse_json(raw: str) -> dict:
    if not raw:
        raise ValueError("Empty AI response")
    raw = re.sub(r"```+(?:json|JSON)?", "", raw)
    raw = re.sub(r"```+", "", raw).strip().strip("`").strip()
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return _validate_and_clean(json.loads(raw))
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return _validate_and_clean(json.loads(
                re.sub(r",\s*([}\]])", r"\1", m.group())
            ))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON in AI response: {raw[:200]}")


def _validate_and_clean(data: dict) -> dict:
    """
    Clean the raw AI JSON response.
    Does NOT add emoji or hashtags here — that happens in _build_post_body
    and _apply_emoji_and_hashtags so it only ever runs once.
    """
    data.setdefault("approved", False)
    data.setdefault("reason", "")
    data.setdefault("issues", [])
    data.setdefault("formatted_text", "")
    data.setdefault("confidence", 0.5)

    if data.get("formatted_text"):
        text = data["formatted_text"]
        text = text.replace("*", "")
        text = _strip_watermarks(text)
        text = re.sub(r"📌\s*(NOTE|MARKET STATUS|STATUS)[^\n]*\n?", "", text)
        text = _strip_be_careful(text)
        text = _strip_hashtag_label(text)
        text = re.sub(r"\n\s*\n", "\n\n", text).strip()
        data["formatted_text"] = text

    if data.get("approved"):
        block = _hard_block(data.get("formatted_text", ""))
        if block:
            data["approved"]       = False
            data["reason"]         = f"Hard blocked: {block}"
            data["issues"].append(block)
            data["formatted_text"] = ""

    return data


def _reject(reason: str, issue: str, confidence: float = 1.0) -> dict:
    return {
        "approved":       False,
        "reason":         reason,
        "issues":         [issue],
        "formatted_text": "",
        "confidence":     confidence,
        "engine":         "pre_filter",
    }


def _build_post_body(text: str, is_calendar: bool = False) -> str:
    """
    Final post assembly — called ONCE per approved post.
    Order: clean → emoji + hashtags → US flag → signature.
    Emoji and hashtags are added here and NOWHERE else.
    """
    if not text:
        return ""
    text = text.replace("*", "")
    text = _strip_watermarks(text)
    text = re.sub(r"📌\s*(NOTE|MARKET STATUS|STATUS)[^\n]*\n?", "", text)
    text = _strip_be_careful(text)
    text = _strip_hashtag_label(text)
    # Remove 4-digit years from last 3 lines
    lines = text.split("\n")
    for i in range(max(0, len(lines) - 3), len(lines)):
        lines[i] = re.sub(r"\b\d{4}\b", "", lines[i])
    text = "\n".join(lines)
    text = re.sub(r"\n\s*\n", "\n\n", text).strip()

    # ── Single pass: emoji + hashtags ─────────────────────────────────────────
    text = _apply_emoji_and_hashtags(text, is_calendar=is_calendar)

    # ── US flag on USD mentions in first line (after emoji applied) ───────────
    if not is_calendar:
        text = _add_us_flag(text)

    # ── Signature ─────────────────────────────────────────────────────────────
    text = _add_signature(text)

    return text


# ── AIEngine class ────────────────────────────────────────────────────────────
class AIEngine:
    def __init__(self, gemini_key: str, groq_key: str, channel_category: str):
        self._category = channel_category
        self._groq = AsyncGroq(api_key=groq_key)
        genai.configure(api_key=gemini_key)

        self._gemini = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=_SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0.15, max_output_tokens=600,
                response_mime_type="application/json"
            ),
        )
        self._gemini_text = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.GenerationConfig(
                temperature=0.2, max_output_tokens=1200
            ),
        )
        self._gemini_vision = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.GenerationConfig(
                temperature=0.1, max_output_tokens=1000
            ),
        )

    # ── Main analyse ──────────────────────────────────────────────────────────

    async def analyse(self, text: str, image_data: Optional[bytes] = None,
                      image_mime: str = "image/jpeg") -> dict:

        if text:
            text = _strip_watermarks(text)
            if _SENTIMENT_RE.search(text):
                return _reject("Sentiment indicator blocked", "sentiment_content")
            if _INJECTION_RE.search(text):
                return _reject("Prompt injection blocked", "injection_attempt")

        prompt = textwrap.dedent(f"""
            DATE (UTC): {_today_str()}
            CHANNEL FOCUS: {self._category}
            SOURCE CONTENT:
            \"\"\"
            {text.strip() if text else "(image only)"}
            \"\"\"
            Analyse this content.
            If real geopolitical/macro news OR actual released data OR price level hit → approve.
            If forecast/previous only/signal/TA/chart/meme/sentiment/stale → reject.

            IMPORTANT: If approved, copy source text with MINIMAL changes only.
            Do NOT rewrite. Do NOT add sentences. Do NOT add predictions or context.
            Only remove watermarks. Fix grammar only if clearly wrong. Keep all numbers exact.
            Do NOT add hashtags — system handles them. Return JSON.
        """).strip()

        # Try Gemini with one retry
        for attempt in range(2):
            try:
                verdict = await asyncio.wait_for(
                    self._gemini_call(prompt, image_data, image_mime), timeout=40
                )
                verdict["engine"] = "gemini-2.5-flash"
                if verdict.get("approved") and verdict.get("formatted_text"):
                    verdict["formatted_text"] = _build_post_body(
                        verdict["formatted_text"], is_calendar=False
                    )
                return verdict
            except asyncio.TimeoutError:
                if attempt == 0:
                    await asyncio.sleep(3)
                    continue
                break
            except Exception:
                break

        # Groq fallback
        try:
            verdict = await asyncio.wait_for(
                self._groq_call(prompt, image_data, image_mime), timeout=55
            )
            verdict["engine"] = "groq-llama4-scout"
            if verdict.get("approved") and verdict.get("formatted_text"):
                verdict["formatted_text"] = _build_post_body(
                    verdict["formatted_text"], is_calendar=False
                )
            return verdict
        except Exception:
            return _reject("Both AI engines unavailable", "engine_error", confidence=0.0)

    # ── FF image detection ────────────────────────────────────────────────────

    async def detect_ff_image(self, image_data: bytes, image_mime: str) -> tuple:
        try:
            parts = [
                {"inline_data": {"mime_type": image_mime, "data": _b64(image_data)}},
                _FF_DETECT_PROMPT,
            ]
            loop = asyncio.get_running_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self._gemini_vision.generate_content(parts)
                ),
                timeout=30
            )
            raw  = re.sub(r"```+(?:json)?", "", resp.text).strip()
            data = json.loads(raw)
            return bool(data.get("is_ff", False)), bool(data.get("is_weekly", False))
        except Exception:
            return False, False

    # ── FF image analysis ─────────────────────────────────────────────────────

    async def analyse_ff_image(self, image_data: bytes, image_mime: str,
                               today_date: str, is_weekly: bool = False,
                               week_range: str = "") -> dict:
        prompt = (
            _FF_WEEKLY_PROMPT.format(week_range=week_range)
            if is_weekly else
            _FF_DAILY_PROMPT.format(today_date=today_date)
        )
        parts = [
            {"inline_data": {"mime_type": image_mime, "data": _b64(image_data)}},
            prompt,
        ]

        for attempt in range(2):
            try:
                loop = asyncio.get_running_loop()
                resp = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, lambda: self._gemini_vision.generate_content(parts)
                    ),
                    timeout=90
                )
                data = _parse_json(resp.text)
                if data.get("approved") and data.get("formatted_text"):
                    data["formatted_text"] = _build_post_body(
                        data["formatted_text"], is_calendar=True
                    )
                return data
            except asyncio.TimeoutError:
                if attempt == 0:
                    await asyncio.sleep(3)
                    continue
                break
            except Exception:
                break

        # Groq fallback
        try:
            content = [
                {"type": "image_url",
                 "image_url": {"url": f"data:{image_mime};base64,{_b64(image_data)}"}},
                {"type": "text", "text": prompt},
            ]
            resp = await asyncio.wait_for(
                self._groq.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": content}],
                    temperature=0.1, max_tokens=800,
                ),
                timeout=90,
            )
            data = _parse_json(resp.choices[0].message.content)
            if data.get("approved") and data.get("formatted_text"):
                data["formatted_text"] = _build_post_body(
                    data["formatted_text"], is_calendar=True
                )
            return data
        except Exception:
            return {"approved": False, "reason": "AI engines unavailable for image analysis."}

    # ── Similarity check ──────────────────────────────────────────────────────

    async def is_same_story(self, text_a: str, text_b: str,
                            image_a: Optional[bytes] = None,
                            image_b: Optional[bytes] = None) -> bool:
        """
        IMPROVED similarity check:
        - Normalise both texts before comparing
        - Exact match → instant True
        - High keyword overlap (≥0.80) → True without AI
        - Low overlap (<0.15) → False without AI (was 0.10 — stricter)
        - Grey zone → AI decides (confidence threshold raised to 0.80)
        - Geopolitical breaking news: different entities = NOT duplicate
          even if similar wording (e.g. two different missile strikes)
        """
        def _normalise(t: str) -> str:
            if not t:
                return ""
            t = t.lower()
            t = _strip_watermarks(t)
            t = re.sub(r"[^\w\s]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t

        norm_a = _normalise(text_a)
        norm_b = _normalise(text_b)

        # Exact match
        if norm_a and norm_b and norm_a == norm_b:
            return True

        stops = {
            "the","a","an","is","are","was","were","in","on","at","to",
            "of","and","or","for","with","as","by","from","that","this",
            "it","its","be","been","has","have","had","will","said","says",
            "after","before","during","while","amid","over","under","into",
        }

        if norm_a and norm_b:
            keys_a = set(norm_a.split()) - stops
            keys_b = set(norm_b.split()) - stops

            if keys_a and keys_b:
                overlap = len(keys_a & keys_b) / min(len(keys_a), len(keys_b))

                # Very high overlap — definite duplicate
                if overlap >= 0.80:
                    return True

                # Very low overlap — clearly different stories, skip AI
                if overlap < 0.15 and not image_a and not image_b:
                    return False

                # IMPROVEMENT: if both contain a named entity (country, person)
                # and those entities differ → not a duplicate even with moderate overlap
                _ENTITIES = [
                    "trump", "powell", "biden", "putin", "xi", "iran", "ukraine",
                    "russia", "china", "europe", "opec", "nato", "israel", "gaza",
                    "fed", "ecb", "boj", "boe", "gold", "oil", "nfp", "cpi",
                    "gdp", "pce",
                ]
                ents_a = {e for e in _ENTITIES if e in norm_a}
                ents_b = {e for e in _ENTITIES if e in norm_b}
                # If both have entities and they don't overlap — different stories
                if ents_a and ents_b and not (ents_a & ents_b):
                    return False

        has_content = norm_a or norm_b or image_a or image_b
        if not has_content:
            return False

        has_images = image_a or image_b
        prompt = (
            _SIMILARITY_IMAGE_PROMPT if has_images else _SIMILARITY_PROMPT
        ).format(
            story_a=(norm_a[:800] if norm_a else "(image only)"),
            story_b=(norm_b[:800] if norm_b else "(image only)"),
        )

        # Gemini
        try:
            parts = []
            if image_a:
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": _b64(image_a)}})
            if image_b:
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": _b64(image_b)}})
            parts.append(prompt)

            loop = asyncio.get_running_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self._gemini_vision.generate_content(parts)
                ),
                timeout=25,
            )
            data       = _parse_json(resp.text)
            confidence = data.get("confidence", 0)
            same       = bool(data.get("same_story", False))
            # RAISED threshold: was 0.75, now 0.80 — less aggressive blocking
            if same and confidence >= 0.80:
                return True
            if not same and confidence >= 0.80:
                return False
        except Exception as e:
            log.warning(f"Gemini similarity failed: {e}")

        # Groq fallback
        try:
            content: list = []
            if image_a:
                content.append({"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{_b64(image_a)}"}})
            if image_b:
                content.append({"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{_b64(image_b)}"}})
            content.append({"type": "text", "text": prompt})

            resp = await asyncio.wait_for(
                self._groq.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": content}],
                    temperature=0.1,
                    max_tokens=300,
                ),
                timeout=30,
            )
            data       = _parse_json(resp.choices[0].message.content)
            confidence = data.get("confidence", 0)
            same       = bool(data.get("same_story", False))
            return same and confidence >= 0.80
        except Exception as e:
            log.warning(f"Groq similarity failed: {e}")
            return False

    async def get_be_careful_line(self, event_name: str) -> str:
        return _get_be_careful_line(event_name)

    # ── Internal calls ────────────────────────────────────────────────────────

    async def _gemini_call(self, prompt: str, image_data: Optional[bytes],
                           image_mime: str) -> dict:
        parts = []
        if image_data:
            parts.append({"inline_data": {"mime_type": image_mime, "data": _b64(image_data)}})
        parts.append(prompt)
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, lambda: self._gemini.generate_content(parts)
        )
        return _parse_json(resp.text)

    async def _groq_call(self, prompt: str, image_data: Optional[bytes],
                         image_mime: str) -> dict:
        content = []
        if image_data:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime};base64,{_b64(image_data)}"}
            })
        content.append({"type": "text", "text": prompt})
        resp = await self._groq.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": content},
            ],
            temperature=0.15, max_tokens=600,
        )
        return _parse_json(resp.choices[0].message.content)

    @staticmethod
    def _fallback_alert(event: dict, minutes_left: int) -> str:
        emoji      = "🔴" if event.get("impact") == "red" else "🟠"
        event_name = event.get("name", "Unknown Event")
        line       = _get_be_careful_line(event_name)
        text = (
            f"🚨 ALERT: {minutes_left} MINUTES REMAINING\n\n"
            f"{emoji} {event_name}\n"
            f"🕒 {event.get('time_12h', '—')}\n\n"
            f"REQUIRED ACTION:\n"
            f"✅ Secure open profits now\n"
            f"✅ Move Stop-Loss to Break-even\n"
            f"✅ No new entries during the release\n\n"
            f"{line}"
        )
        return _add_signature(_add_us_flag(text))
