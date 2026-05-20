"""
ai_engine.py — Final version.

FIXES (this version):
1. _FF_IMAGE_PROMPT now explicitly tells AI to GROUP same-time events on
   ONE line with comma-separated names — no splitting by time slot.
2. FF image prompt reinforced: reject any non-ForexFactory calendar source.
3. Weekly prompt also updated to group same-time events.
4. Random 💡 signature 25% of the time.
5. "Be careful" reminder lines — short, event-specific.

CRITICAL BUGS FIXED:
1. Decimal/Prediction Regex Bug: Decimals in prices ($2.50) no longer break sentences.
2. 4-Digit Price Eraser: Replaced `\b\d{4}\b` with `\b20\d{2}\b` to protect asset prices.
3. Unescaped HTML Crash: Added safe HTML escaping for Telethon compatibility.
4. Groq JSON Format: Added `response_format={"type": "json_object"}` to prevent Llama output crashes.
"""

import asyncio
import base64
import json
import logging
import random
import re
import textwrap
from datetime import datetime, timezone
from typing import Optional, List

import google.generativeai as genai
from groq import AsyncGroq

log = logging.getLogger("ai_engine")

CHANNEL_SIGNATURE = "\n\n[Squad 4xx](https://t.me/Squad_4xx)"
ALLOWED_HASHTAGS_SET = {"#XAUUSD", "#DXY", "#OIL"}

# ── Hashtag keyword maps — CODE decides hashtags, not AI ─────────────────────
_HASHTAG_XAUUSD_KEYWORDS = [
    "gold", "xau", "xauusd", "xau/usd", "bullion",
    "precious metal", "spot gold", "gold price",
    "gold drops", "gold falls", "gold rises", "gold surges",
    "gold crashes", "gold spike", "gold jumps", "gold climbs",
    "gold hits", "gold at $", "gold to $",
]

_HASHTAG_OIL_KEYWORDS = [
    "oil", "crude", "brent", "wti", "opec", "barrel",
    "petroleum", "energy supply", "hormuz", "oil supply",
    "oil price", "oil drops", "oil falls", "oil rises",
    "oil surges", "oil crashes", "oil spike", "oil jumps",
    "oil collapses", "oil at $", "oil to $",
]

_HASHTAG_DXY_KEYWORDS = [
    "dxy", "dollar index", "usd index", "us dollar",
    "dollar surges", "dollar falls", "dollar drops",
    "dollar strengthens", "dollar weakens", "dollar rises",
    "dollar jumps", "dollar collapses", "dollar climbs",
    "dxy falls", "dxy rises", "dxy drops", "dxy jumps",
    "greenback", "dollar at ", "dollar to ",
    "tariff", "tariffs", "trade war", "fed rate",
    "fomc", "federal funds", "interest rate decision",
    "powell", "federal reserve",
    "eur/usd", "gbp/usd", "usd/jpy", "aud/usd",
]


def _detect_hashtags(text: str) -> str:
    if not text:
        return ""
    if "TODAY'S USD" in text or "WEEKLY HIGH IMPACT" in text:
        return ""

    lower = text.lower()
    tags = []

    if any(kw in lower for kw in _HASHTAG_XAUUSD_KEYWORDS):
        tags.append("#XAUUSD")
    if any(kw in lower for kw in _HASHTAG_OIL_KEYWORDS):
        tags.append("#OIL")
    if any(kw in lower for kw in _HASHTAG_DXY_KEYWORDS):
        tags.append("#DXY")

    return " ".join(tags)


def _apply_hashtags(text: str) -> str:
    if not text:
        return text

    if "TODAY'S USD" in text or "WEEKLY HIGH IMPACT" in text:
        return re.sub(r"#\w+", "", text).strip()

    hashtags = _detect_hashtags(text)
    clean = re.sub(r"#\w+", "", text).strip()

    if hashtags:
        return clean + "\n\n" + hashtags
    return clean


def _add_signature(text: str) -> str:
    text = text.strip()
    if "Squad 4xx" not in text:
        if random.random() < 0.25:
            text += '\n\n💡 <a href="https://t.me/Squad_4xx">Squad 4xx</a>'
        else:
            text += '\n\n<a href="https://t.me/Squad_4xx">Squad 4xx</a>'
    return text


def _escape_html(text: str) -> str:
    """Escape HTML special characters in plain text before sending."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            )


def _add_us_flag_emoji(text: str) -> str:
    if not text:
        return text
    lines = text.split('\n')
    if not lines:
        return text
    first_line = lines[0]
    new_line = re.sub(r'\bUS\b', 'US 🇺🇸', first_line, count=1)
    new_line = re.sub(r'\bUSD\b', 'USD 🇺🇸', new_line, count=1)
    lines[0] = new_line
    return '\n'.join(lines)


def _strip_be_careful(text: str) -> str:
    return re.sub(r'\n?Be careful[^\n]*\n?', '', text, flags=re.IGNORECASE).strip()


def _strip_predictions(text: str) -> str:
    """Hard-strip any AI-added prediction/opinion sentences (safely ignoring decimals)."""
    if not text:
        return text
        
    _PREDICT_KEYWORDS = (
        r'could\s+(go|rise|fall|drop|reach|push|move|head)|'
        r'may\s+(lead|push|cause|result|trigger|move)|'
        r'might\s+(rise|fall|drop|push|move)|'
        r'this\s+(suggests?|indicates?|signals?|means?)|'
        r'suggesting\b|indicating\b|implying\b|'
        r'watch\s+for\b|heading\s+(to|toward)|'
        r'next\s+(move|target|level)|'
        r'bulls?\s+(could|may|might)|bears?\s+(could|may|might)|'
        r'bullish\s+(momentum|signal|outlook)|'
        r'bearish\s+(momentum|signal|outlook)'
    )
    
    # Split by actual sentence boundaries (., !, ?, \n) but IGNORE decimals like 2.50
    sentences = re.split(r'(?<!\d)\.(?!\d)|[!?\n]', text)
    cleaned_sentences = []
    
    for sentence in sentences:
        if not sentence:
            continue
        if re.search(_PREDICT_KEYWORDS, sentence, re.IGNORECASE):
            continue
        cleaned_sentences.append(sentence.strip())
        
    cleaned = ". ".join(cleaned_sentences) + "."
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned if cleaned != "." else ""


_AI_FILLER_PHRASES = [
    r'\bas investors\b[^.]*',
    r'\bamid\s+(concerns?|fears?|uncertainty|tensions?|pressure)[^,.\n]*',
    r'\bfollowing\s+the\s+(better|worse|stronger|weaker)[^,.\n]*',
    r'\bin\s+response\s+to\b[^,.\n]*',
    r'\bas\s+markets\b[^.]*',
    r'\bdriven\s+by\b[^,.\n]*',
    r'\bthis\s+comes\s+as\b[^.]*',
    r'\bmeeting\s+expectations?\b[^.]*',
    r'\bnoteworthy\b[^,.\n]*',
    r'\bnotably\b[^,.\n]*',
    r'\bsignificantly\b[^,.\n]*',
    r'\bsignificant\s+(move|drop|rise|impact)[^,.\n]*',
    r'\bflight\s+to\s+safety\b[^.]*',
    r'\brisk[- ]off\b[^,.\n]*',
    r'\brisk[- ]on\b[^,.\n]*',
    r'\bsafe[- ]haven\s+demand\b[^,.\n]*',
    r'\bmarket\s+participants?\b[^.]*',
    r'\btraders?\s+(are|were)\s+(watching|monitoring|reacting)[^.]*',
]

_AI_FILLER_RE = re.compile(
    '(' + '|'.join(_AI_FILLER_PHRASES) + ')',
    re.IGNORECASE
)


def _strip_ai_filler(text: str) -> str:
    if not text:
        return text
    cleaned = _AI_FILLER_RE.sub('', text)
    cleaned = re.sub(r'\s*,\s*,', ',', cleaned)
    cleaned = re.sub(r'\s*\.\s*\.', '.', cleaned)
    cleaned = re.sub(r',\s*\.', '.', cleaned)
    cleaned = re.sub(r'\s+\.', '.', cleaned)
    cleaned = re.sub(r'\s+,', ',', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def _get_be_careful_line(event_name: str) -> str:
    n = event_name.lower()
    if any(kw in n for kw in ["fomc", "federal funds", "interest rate", "fed chair", "powell", "federal reserve"]):
        return "⚠️ Fed decisions move everything. Be careful — no new trades during the release."
    if any(kw in n for kw in ["non-farm", "nfp", "payroll"]):
        return "⚠️ NFP can spike the market violently. Be careful — protect your capital now."
    if any(kw in n for kw in ["cpi", "consumer price", "inflation"]):
        return "⚠️ Inflation data whipsaws fast. Be careful — secure profits before the release."
    if any(kw in n for kw in ["pce", "core pce"]):
        return "⚠️ PCE can shift rate expectations quickly. Be careful and protect your positions."
    if "gdp" in n:
        return "⚠️ GDP surprises hit hard and fast. Be careful — move stops to break-even now."
    if any(kw in n for kw in ["unemployment rate", "jobless"]):
        return "⚠️ Unemployment data moves USD sharply. Be careful — no new entries during release."
    if any(kw in n for kw in ["retail sales"]):
        return "⚠️ Retail Sales can jolt the market. Be careful — protect your open positions."
    if any(kw in n for kw in ["ism manufacturing", "ism non-manufacturing", "ism services"]):
        return "⚠️ ISM data can move USD fast. Be careful — stay out until the dust settles."
    if any(kw in n for kw in ["employment cost", "eci"]):
        return "⚠️ Employment Cost data affects rate outlook. Be careful and reduce your exposure."
    if any(kw in n for kw in ["ppi", "producer price"]):
        return "⚠️ PPI surprises can hit USD hard. Be careful — protect your capital."
    if any(kw in n for kw in ["trade balance", "current account"]):
        return "⚠️ Trade data can move USD unexpectedly. Be careful during the release."
    if any(kw in n for kw in ["durable goods"]):
        return "⚠️ Durable Goods can cause sharp moves. Be careful — no new entries now."
    return "⚠️ This release can move the market strongly. Be careful — protect your capital."


_SYSTEM_PROMPT = """
You are AXIOM INTEL — a strict financial news gatekeeper.

YOUR MISSION:
Only approve news that has a DIRECT, CONFIRMED, REAL impact on these markets:
- Gold (XAU/USD)
- Oil (WTI / Brent / Crude)
- US Dollar (DXY / USD)

If the news does NOT directly move or affect one of these three markets RIGHT NOW — REJECT IT.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ APPROVE ONLY THESE — nothing else
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. GOLD PRICE MOVE
   Gold/XAU already dropped, rose, crashed, surged, spiked
   with an actual price or % stated.
   Example: "Gold drops $67 to $2,954" ✅
   Example: "XAU/USD falls 1.2%" ✅
   NOT: "Gold looking weak today" ❌ (no confirmed move)

2. OIL PRICE MOVE
   Oil/Crude/Brent/WTI already dropped, rose, collapsed, surged
   with actual price or % stated.
   Example: "WTI crude falls 4% to $68.30" ✅
   NOT: "Oil markets under pressure" ❌ (vague, no confirmed move)

3. USD / DXY MOVE
   Dollar/DXY already dropped, rose, strengthened, weakened
   with actual level or % stated.
   Example: "DXY falls to 99.80" ✅
   NOT: "Dollar facing headwinds" ❌ (vague, no confirmed move)

4. FOMC / FED DECISION (actual decision only)
   Actual rate decision announced RIGHT NOW.
   Example: "Fed cuts rates 25bps to 4.75%" ✅
   NOT: "Fed expected to cut rates next month" ❌ (forecast)
   NOT: "Fed member says inflation cooling" ❌ (commentary)

5. MAJOR ECONOMIC DATA — ACTUAL RELEASED FIGURE
   NFP, CPI, GDP, PCE — actual released number stated.
   Example: "US CPI comes in at 3.4%" ✅
   Example: "NFP adds 180K jobs" ✅
   NOT: "CPI expected at 3.2% on Friday" ❌ (forecast)
   NOT: "Analysts expect strong NFP" ❌ (forecast)

6. WORLD LEADER STATEMENTS — ALWAYS APPROVE IF ABOUT:
   Any official statement, tweet, or post from a world leader
   (Trump, Biden, Putin, Xi, Netanyahu, Khamenei, MBS, etc.)
   that directly concerns ANY of these topics:
   - Oil supply, Hormuz strait, OPEC, energy embargo, sanctions
   - War, military action, ceasefire, invasion
   - Tariffs, trade war, economic sanctions
   - Central bank policy, dollar policy, gold reserves
   - Any statement that will clearly move Gold, Oil or USD

   DO NOT require a price move to be stated.
   The statement itself IS the market-moving event.

   Example: Trump posts "Iran in State of Collapse — Open Hormuz" ✅
   Example: Putin announces new oil export restrictions ✅
   Example: Trump announces 30% tariffs on China ✅
   Example: Netanyahu threatens Iran nuclear sites ✅
   affects_markets for these: ["OIL"] for Hormuz/energy,
   ["DXY"] for tariffs/trade, ["XAUUSD","OIL"] for war/conflict

7. GEOPOLITICAL — CONFIRMED GOLD OR OIL PRICE MOVE
   War/conflict/sanctions that are CONFIRMED to have moved
   Gold or Oil price RIGHT NOW — price must be stated.
   Example: "Israel strikes Iran — Oil spikes $4 to $88" ✅
   NOT: "Tensions rise in Middle East" ❌ (no leader, no price)
   NOT: "Iran threatens Hormuz" ❌ (anonymous source, no leader)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ REJECT ALL OF THESE — no exceptions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Signals: Buy/Sell/Long/Short/Entry/TP/SL
- Technical analysis: patterns, indicators, support/resistance
- Predictions: "could go", "might reach", "watch for", "heading to"
- Forecasts: "expected", "analysts expect", "forecast", "projected"
- Opinions: "bullish", "bearish", "looking weak/strong"
- Sentiment: Fear & Greed, COT, smart money, bank positioning
- Commentary: "markets are nervous", "risk-off mood", "investors worried"
- Vague moves: "gold under pressure", "oil volatile", "dollar struggles"
  (no actual confirmed price/% stated = REJECT)
- Central bank speeches WITHOUT a rate decision
  (Powell talking = REJECT unless actual rate changed)
- Geopolitical WITHOUT confirmed Gold/Oil price move
  (war news alone = REJECT unless price stated)
- Stock market moves (S&P, Nasdaq, Dow) — NOT our markets
- Crypto (Bitcoin, Ethereum) — NOT our markets
- Economic data WITHOUT actual released number
- Stale news older than 18 hours
- Memes, jokes, promotions, watermarks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE TEST — ask yourself before approving:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Is this a world leader statement about Oil, Hormuz, War,
   Tariffs, Sanctions, or Gold? → APPROVE IMMEDIATELY
2. Did Gold, Oil, or USD ACTUALLY move RIGHT NOW? (confirmed price/%)
3. Was there an ACTUAL Fed rate decision announced?
4. Was there an ACTUAL economic data release with real number?

If question 1 is YES → APPROVE.
If ANY of 2, 3, 4 is YES → APPROVE.
If ALL are NO → REJECT.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATTING (if approved) — STRICT RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — SHORT AND CLEAN:
Maximum 2 lines of body text after the headline.
Do NOT repeat the headline in the body.
Do NOT copy the full statement word for word.
Write ONE clean factual sentence summarising what happened.

RULE 2 — BEAUTIFUL SENTENCE:
Write naturally. One strong sentence. Clear and direct.
No "Thank you for your attention", no filler, no greetings.
Remove any personal opinions from the leader ("Which I believe").
Keep ONLY the core fact.

RULE 3 — IF NEWS HAS AN IMAGE ATTACHED:
Write even shorter — headline + ONE sentence only.
The image carries the visual. Text must be minimal.

RULE 4 — LEADER ATTRIBUTION:
End with a clean attribution line if it is a leader statement:
— Donald Trump  OR  — Vladimir Putin  OR  — Netanyahu
Short name only. No titles, no "President", no "PM".

RULE 5 — FORMAT STRUCTURE:
[EMOJI] [SHORT HEADLINE — one line, factual, clean]

[ONE sentence — the core fact only]

— [Leader name if applicable]

EXAMPLE — Trump Hormuz tweet:
CORRECT:
🛢️ Iran Declares "State of Collapse" — Requests Hormuz Strait Be Opened

Iran has informed the US it is in a state of collapse 
and is requesting the Hormuz Strait be opened.

— Donald Trump

WRONG (too long, repeating, ugly):
🛢️ Iran Has Informed Us They Are in a State of Collapse
Iran has just informed us that they are in a State of Collapse.
They want us to Open the Hormuz Strait as soon as possible as
they try to figure out their leadership situation. Thank you
for your attention to this matter! President DONALD J. TRUMP

RULE 6 — HASHTAGS (STRICT — only confirmed affected market):
- Hormuz / oil supply threat → #OIL only
- Tariffs affecting dollar → #DXY only
- Gold price confirmed moved → #XAUUSD only
- War with confirmed oil move → #OIL
- Do NOT add all three unless ALL THREE are confirmed affected
- When in doubt → use fewer hashtags, not more

Plain text only — NO asterisks, NO bold, NO markdown.
Do NOT add signature (added automatically).

RESPOND WITH VALID JSON ONLY — NO MARKDOWN FENCES — NO TRAILING COMMAS:
{"approved": true, "reason": "brief reason", "issues": [], "formatted_text": "...", "confidence": 0.9, "affects_markets": ["OIL"]}

affects_markets — STRICT:
- "XAUUSD" → Gold price directly moved or confirmed impacted
- "OIL"    → Oil supply or price directly affected
- "DXY"    → Dollar directly moved OR Fed rate decision OR tariffs
- []       → No confirmed direct market impact
- Hormuz threat → ["OIL"] only — not XAUUSD, not DXY
- Tariffs → ["DXY"] — add XAUUSD only if gold is mentioned
- War → ["OIL"] if oil mentioned, ["XAUUSD"] if gold mentioned
- Fed rate cut → ["DXY"] only unless gold/oil also stated
""".strip()

_VIDEO_CAPTION_PROMPT = """
You are AXIOM INTEL — a strict financial news gatekeeper.

A video has been received. Read the caption carefully.

APPROVE ONLY if the caption confirms one of these:
- Gold/XAU price ALREADY moved — actual price or % stated
- Oil/Crude/Brent/WTI price ALREADY moved — actual price or % stated
- USD/DXY ALREADY moved — actual level or % stated
- FOMC actual rate decision announced right now
- Major economic data ACTUAL released figure (NFP, CPI, GDP, PCE)
- War/conflict that DIRECTLY caused Gold or Oil to move — price stated
- World leader statement (Trump, Putin, Xi, Netanyahu, etc.) about:
  Oil supply, Hormuz, sanctions, tariffs, war, trade — APPROVE even
  without a price move — the statement itself moves markets

REJECT if:
- No actual confirmed price move stated (vague language = REJECT)
- War/conflict news WITHOUT a Gold or Oil price stated
- Fed speech WITHOUT actual rate change
- Forecast, expected, analyst opinion
- Signals, TA, predictions
- "Gold looking weak", "oil under pressure" — no confirmed number = REJECT
- Promotions, memes, entertainment
- Stock market, crypto — not our markets

If APPROVED:
- Use 📉 drops, 📈 rises, 📊 volatile, 🚨 breaking, 🛢️ oil, 💵 dollar
- One factual headline — exactly what happened
- 1-2 sentences copied from caption — no added words
- No predictions, no opinions, no hashtags
- Do NOT add signature (added automatically)
- Plain text only — no asterisks

Caption: {caption}

RESPOND WITH VALID JSON ONLY:
{{"approved": true/false, "reason": "brief reason", "formatted_text": "...", "confidence": 0.0-1.0, "affects_markets": []}}
""".strip()

_VIDEO_VISUAL_PROMPT = """
You are AXIOM INTEL — a Senior Institutional Macro & Geopolitical news editor.

You are looking at frames extracted from a video. The video caption is also provided.

CAPTION: {caption}

YOUR JOB:
Look at the frames AND read the caption together.

APPROVE if you see OR the caption describes any FACTUAL event:
- Active combat, explosions, fire, smoke from strikes
- Military vehicles, troops, weapons, missile launches
- Drone strikes, airstrikes, artillery shelling
- Destroyed buildings, bombed areas, war damage
- Breaking news chyron on screen about war/conflict/market crash
- World leaders making statements about war, sanctions, oil, tariffs
- Oil infrastructure under threat or attack
- News ticker showing actual price drop 📉 or surge 📈
- Any market move already confirmed on screen (price shown)

REJECT if:
- Frames show charts with TA indicators, drawings, support/resistance lines
- Someone talking to camera giving trading predictions or signals
- Promotional content, ads
- No visual war/conflict/market-move evidence AND caption is vague or empty

If APPROVED — write a clean formatted post:
- Use 📉 for drops/falls/crashes, 📈 for rises/surges, 📊 for volatility/swings
- Use 🚨 for breaking war/geopolitical news, 💥 for explosions/strikes
- One clear factual headline — exactly what happened
- 1-2 sentences of factual detail from caption + what you see in frames
- NO prediction words: "could", "may", "might", "suggesting", "indicating"
- Plain text only — no asterisks, no bold, no hashtags
- Do NOT add signature (added automatically)

RESPOND WITH VALID JSON ONLY:
{{"approved": true/false, "reason": "brief reason", "formatted_text": "...", "confidence": 0.0-1.0, "visual_confirmed": true/false}}
""".strip()

_SIMILARITY_PROMPT = """
You are a duplicate news detector. Compare the two stories. If they describe the same real-world event – even if worded differently, in different languages, or with minor spelling mistakes – respond with same_story=true.
Be aggressive. If any reasonable chance they are the same, mark true.

Story A: {story_a}
Story B: {story_b}

Respond in JSON: {{"same_story": true/false, "confidence": 0.0-1.0, "reason": "..."}}
"""

_MULTIMODAL_SIMILARITY_PROMPT = """
You are a duplicate news detector. Compare the two items (text + optional images). Decide if they are the SAME real-world event.

Item A text: {text_a}
Item B text: {text_b}
(Images are compared visually if both exist)

Be aggressive: if there is any reasonable chance they are the same, mark same_story=true.

Respond with JSON: {{"same_story": true, "confidence": 0.0-1.0, "reason": "..."}}
"""

_FF_IMAGE_PROMPT = """
You are analysing a ForexFactory economic calendar screenshot.

TODAY'S DATE: {today_date}

STEP 1 — SOURCE VALIDATION (CRITICAL):
This tool ONLY accepts screenshots from ForexFactory.com.
Look for "forexfactory.com", the ForexFactory logo, or the exact FF calendar layout.
If this is any other website's calendar (Investing.com, DailyFX, TradingEconomics, 
myfxbook, etc.) — immediately respond: {{"approved": false, "reason": "not forexfactory"}}

STEP 2 — DATE CHECK:
Look at the date shown in the screenshot (header, "Today" label, column header, etc).
The screenshot must show the SAME month and day as {today_date}.
IMPORTANT: Ignore formatting differences — "May 1", "May 01", "Fri May 1", "05/01" all 
count as the same date. Only reject if the month OR day is clearly different.
Do NOT reject just because the year is missing or the format looks different.

STEP 3 — EXTRACT EVENTS:
Extract ALL USD high-impact (🔴) and medium-impact (🟠) events visible.

GROUPING RULE — CRITICAL:
If two or more events share the EXACT SAME TIME, put them ALL on ONE line with
comma-separated names. Do NOT create separate lines for same-time events.

CORRECT (same time → one line):
🔴 3:30 PM | USD: Advance GDP q/q, Core PCE Price Index m/m, Employment Cost Index q/q

WRONG (do NOT do this — split lines for same time):
🔴 3:30 PM | USD: Advance GDP q/q
🔴 3:30 PM | USD: Core PCE Price Index m/m
🔴 3:30 PM | USD: Employment Cost Index q/q

STEP 4 — FORMAT:
- Do NOT include the year in the date line (e.g. "Friday, May 1" — no year).
- Do NOT add any hashtags.
- No forecast, no previous data, no NOTE, no commentary.
- Do NOT add "Be careful" line — added automatically by the system.
- Keep original times. Use 12-hour AM/PM format. No timezone label. No leading zero on hour.
- Plain text only — no asterisks, no bold.

EXACT OUTPUT FORMAT EXAMPLE:

TODAY'S USD HIGH IMPACT NEWS
Friday, May 1

🔴 3:30 PM | USD: Advance GDP q/q, Core PCE Price Index m/m, Employment Cost Index q/q
🟠 5:00 PM | USD: Unemployment Claims

RULES SUMMARY:
- Only USD events (🔴 and 🟠 only)
- Same-time events → ONE line, comma-separated names
- Different-time events → separate lines
- 12-hour AM/PM, no leading zero (3:30 PM not 03:30 PM)
- Do NOT add signature or "Be careful" line

If screenshot clearly shows a DIFFERENT month or day → {{"approved": false, "reason": "wrong date"}}
If not ForexFactory → {{"approved": false, "reason": "not forexfactory"}}
If valid ForexFactory today → {{"approved": true, "reason": "valid FF today image", "formatted_text": "..."}}
RESPOND WITH VALID JSON ONLY.
""".strip()

_FF_WEEKLY_IMAGE_PROMPT = """
You are analysing a ForexFactory.com calendar screenshot for the weekly outlook.

SOURCE VALIDATION (CRITICAL):
Only accept ForexFactory.com screenshots. If this is any other calendar source
(Investing.com, DailyFX, TradingEconomics, etc.) respond:
{{"approved": false, "reason": "not forexfactory"}}

CURRENT WEEK: {week_range}

EXTRACTION RULES:
- Only USD high-impact (🔴) and medium-impact (🟠) events.
- No forecast, no previous data, no hashtags.
- Do NOT include the year anywhere — NO year in week range, NO year in dates.
  WRONG: "Week of May 11 – May 18, 2026"
  CORRECT: "Week of May 11 – May 18"
- Do NOT add "Be careful" line — added automatically.
- No timezone conversion. 12-hour AM/PM only. No leading zero on hours.
- Plain text, no bold. No signature.

GROUPING RULE — CRITICAL:
If two or more events share the EXACT SAME TIME on the same day,
put them ALL on ONE line with comma-separated names.

CORRECT:
🔴 3:30 PM | USD: NFP, Unemployment Rate, Average Hourly Earnings

WRONG:
🔴 3:30 PM | USD: NFP
🔴 3:30 PM | USD: Unemployment Rate

SPACING RULE — CRITICAL:
Put ONE blank line between each day section.
Put NO blank lines between events within the same day.

EXACT FORMAT (follow this precisely):
WEEKLY HIGH IMPACT NEWS
Week of May 11 – May 18

Monday — May 11
🟠 3:00 PM | USD: Some Event

Tuesday — May 12
🔴 3:30 PM | USD: Core CPI m/m, CPI m/m, CPI y/y

Wednesday — May 13
🔴 3:30 PM | USD: Core PPI m/m, PPI m/m

Thursday — May 14
🔴 3:30 PM | USD: Core Retail Sales m/m, Retail Sales m/m, Unemployment Claims

If valid ForexFactory weekly:
{{"approved": true, "reason": "valid FF weekly image", "formatted_text": "WEEKLY HIGH IMPACT NEWS\\nWeek of May 11 – May 18\\n\\nMonday — May 11\\n🟠 3:00 PM | USD: Some Event\\n\\nTuesday — May 12\\n🔴 3:30 PM | USD: Core CPI m/m, CPI m/m"}}

Otherwise: {{"approved": false, "reason": "..."}}
RESPOND WITH VALID JSON ONLY.
""".strip()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _parse_json(raw: str) -> dict:
    if not raw:
        raise ValueError("Empty response from AI engine.")
    raw = re.sub(r"```+(?:json|JSON)?", "", raw)
    raw = re.sub(r"
```+", "", raw)
    raw = raw.strip().strip("`").strip()
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return _validate_and_clean(json.loads(raw))
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        candidate = re.sub(r",\s*([}\]])", r"\1", m.group())
        try:
            return _validate_and_clean(json.loads(candidate))
        except json.JSONDecodeError:
            pass
    log.warning(f"_parse_json failed. Raw snippet: {raw[:200]}")
    raise ValueError(f"No valid JSON found in AI response:\n{raw[:300]}")


def _validate_and_clean(data: dict) -> dict:
    data.setdefault("approved", False)
    data.setdefault("reason", "")
    data.setdefault("issues", [])
    data.setdefault("formatted_text", "")
    data.setdefault("confidence", 0.5)
    data.setdefault("affects_markets", [])

    if data.get("formatted_text"):
        raw_txt = data["formatted_text"]
        
        # ኤችቲኤምኤል ምልክቶችን በደህንነት መለወጥ (Telethon Crash መከላከያ)
        if not raw_txt.startswith("TODAY'S USD") and not raw_txt.startswith("WEEKLY HIGH"):
            raw_txt = _escape_html(raw_txt)

        raw_txt = raw_txt.replace("*", "")
        raw_txt = re.sub(r"📌\s*(NOTE|MARKET STATUS|STATUS)[^\n]*\n?", "", raw_txt).strip()
        raw_txt = _strip_be_careful(raw_txt)
        raw_txt = _strip_predictions(raw_txt)
        raw_txt = _strip_ai_filler(raw_txt)
        text = raw_txt

        clean_text = re.sub(r"#\w+", "", text).strip()

        if "TODAY'S USD" in text or "WEEKLY HIGH IMPACT" in text:
            data["formatted_text"] = clean_text
        else:
            ai_markets = [m.upper() for m in data.get("affects_markets", [])]
            valid_markets = {"XAUUSD", "OIL", "DXY"}
            ai_markets = [m for m in ai_markets if m in valid_markets]

            if ai_markets:
                hashtags = " ".join(f"#{m}" for m in ai_markets)
                log.debug(f"Hashtags from AI affects_markets: {hashtags}")
            else:
                hashtags = _detect_hashtags(text)
                log.debug(f"Hashtags from keyword fallback: {hashtags or '(none)'}")

            if hashtags:
                data["formatted_text"] = clean_text + "\n\n" + hashtags
            else:
                data["formatted_text"] = clean_text

    if data.get("approved") and _signal_hit(data.get("formatted_text", "")):
        log.warning("Signal keyword in output — hard reject.")
        data["approved"] = False
        data["reason"] = "Signal keyword found in output."
        data["issues"].append("signal_content")
        data["formatted_text"] = ""

    return data


def _signal_hit(text: str) -> Optional[str]:
    if not text:
        return None
    _SIGNAL_RE = re.compile(
        r"\b(buy|sell|long|short|entry|tp|take[\s_-]?profit|sl|stop[\s_-]?loss|"
        r"stoploss|stop\s+at\s+\d|entry\s*[:\-]?\s*\d|target\s*[:\-]?\s*\d)\b",
        re.IGNORECASE,
    )
    m = _SIGNAL_RE.search(text)
    return m.group(0).strip() if m else None


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
            generation_config=genai.GenerationConfig(temperature=0.2, max_output_tokens=1200),
        )
        self._gemini_vision = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.GenerationConfig(
                temperature=0.1, max_output_tokens=800,
                response_mime_type="application/json"
            ),
        )

    async def analyse(self, text: str, image_data: Optional[bytes] = None,
                      image_mime: str = "image/jpeg") -> dict:
        prompt = self._build_moderation_prompt(text)
        try:
            verdict = await asyncio.wait_for(
                self._gemini_call(prompt, image_data, image_mime), timeout=40
            )
            verdict["engine"] = "gemini-2.5-flash"
            log.info(f"Gemini → approved={verdict['approved']} | {verdict.get('reason', '')} | markets={verdict.get('affects_markets', [])}")
            if verdict.get("approved") and verdict.get("formatted_text"):
                verdict["formatted_text"] = _build_post_body(
                    verdict["formatted_text"],
                    ai_markets=verdict.get("affects_markets", [])
                )
                if not verdict["formatted_text"].startswith("TODAY'S USD"):
                    verdict["formatted_text"] = _add_us_flag_emoji(verdict["formatted_text"])
            return verdict
        except Exception as exc:
            log.warning(f"Gemini failed ({exc}) — trying Groq …")
            
        try:
            verdict = await asyncio.wait_for(
                self._groq_call(prompt, image_data, image_mime), timeout=55
            )
            verdict["engine"] = "groq-llama4-scout"
            log.info(f"Groq → approved={verdict['approved']} | {verdict.get('reason', '')} | markets={verdict.get('affects_markets', [])}")
            if verdict.get("approved") and verdict.get("formatted_text"):
                verdict["formatted_text"] = _build_post_body(
                    verdict["formatted_text"],
                    ai_markets=verdict.get("affects_markets", [])
                )
                if not verdict["formatted_text"].startswith("TODAY'S USD"):
                    verdict["formatted_text"] = _add_us_flag_emoji(verdict["formatted_text"])
            return verdict
        except Exception as exc:
            log.error(f"Both engines failed — safe reject: {exc}")
            return {"approved": False, "reason": "Both AI engines unavailable.", "issues": ["engine_error"]}

    async def is_same_story(self, text_a: str, text_b: str,
                            image_a: Optional[bytes] = None,
                            image_b: Optional[bytes] = None) -> bool:
        if not text_a and not text_b and not image_a and not image_b:
            return False
        if image_a or image_b:
            prompt = _MULTIMODAL_SIMILARITY_PROMPT.format(
                text_a=(text_a[:400] if text_a else "(no text)"),
                text_b=(text_b[:400] if text_b else "(no text)"),
            )
        else:
            prompt = _SIMILARITY_PROMPT.format(
                story_a=(text_a[:500] if text_a else ""),
                story_b=(text_b[:500] if text_b else ""),
            )
        try:
            parts = []
            if image_a:
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": _b64(image_a)}})
            if image_b:
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": _b64(image_b)}})
            parts.append(prompt)
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._gemini_vision.generate_content(parts)),
                timeout=20
            )
            data = _parse_json(resp.text)
            same = bool(data.get("same_story", False))
            conf = data.get("confidence", 0)
            log.info(f"Gemini similarity → same={same} | conf={conf}")
            return same and conf >= 0.55
        except Exception as exc:
            log.warning(f"Gemini similarity failed ({exc}) — trying Groq …")
            
        try:
            content = []
            if image_a:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(image_a)}"}})
            if image_b:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(image_b)}"}})
            content.append({"type": "text", "text": prompt})
            
            resp = await asyncio.wait_for(
                self._groq.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": content}],
                    response_format={"type": "json_object"},
                    temperature=0.1, max_tokens=300,
                ),
                timeout=25,
            )
            data = _parse_json(resp.choices[0].message.content)
            same = bool(data.get("same_story", False))
            conf = data.get("confidence", 0)
            log.info(f"Groq similarity → same={same} | conf={conf}")
            return same and conf >= 0.55
        except Exception as exc:
            log.error(f"Both engines failed for similarity check: {exc}")
            return False

    async def analyse_ff_image(self, image_data: bytes, image_mime: str, today_date: str,
                               is_weekly: bool = False, week_range: str = "") -> dict:
        if is_weekly:
            prompt = _FF_WEEKLY_IMAGE_PROMPT.format(week_range=week_range)
        else:
            prompt = _FF_IMAGE_PROMPT.format(today_date=today_date)
        parts = [{"inline_data": {"mime_type": image_mime, "data": _b64(image_data)}}, prompt]
        try:
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._gemini_vision.generate_content(parts)),
                timeout=45
            )
            data = _parse_json(resp.text)
            log.info(f"FF image → approved={data.get('approved')} | {data.get('reason', '')}")
            if data.get("approved") and data.get("formatted_text"):
                data["formatted_text"] = _add_us_flag_emoji(data["formatted_text"])
            return data
        except Exception as exc:
            log.warning(f"Gemini FF failed ({exc}) — trying Groq …")
            
        try:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{_b64(image_data)}"}},
                {"type": "text", "text": prompt},
            ]
            resp = await asyncio.wait_for(
                self._groq.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": content}],
                    response_format={"type": "json_object"},
                    temperature=0.1, max_tokens=800,
                ),
                timeout=60,
            )
            data = _parse_json(resp.choices[0].message.content)
            log.info(f"Groq FF → approved={data.get('approved')}")
            if data.get("approved") and data.get("formatted_text"):
                data["formatted_text"] = _add_us_flag_emoji(data["formatted_text"])
            return data
        except Exception as exc:
            log.error(f"Both engines failed for FF image: {exc}")
            return {"approved": False, "reason": "AI engines unavailable for image analysis."}

    async def get_be_careful_line(self, event_name: str) -> str:
        return _get_be_careful_line(event_name)

    async def analyse_video(self, caption: str,
                            frames: Optional[List[bytes]] = None) -> dict:
        caption = (caption or "").strip()

        log.info(f"🎥 Video Stage 1 — caption analysis | caption={caption[:80]!r}")

        if caption:
            prompt = _VIDEO_CAPTION_PROMPT.format(caption=caption[:800])
            stage1 = await self._try_engines_text(prompt, timeout_gemini=30, timeout_groq=40)

            if stage1:
                conf = stage1.get("confidence", 0.5)
                approved = stage1.get("approved", False)

                if conf >= 0.80:
                    log.info(f"Stage 1 high-confidence → approved={approved} conf={conf:.2f} (skipping frames)")
                    if approved and stage1.get("formatted_text"):
                        stage1["formatted_text"] = _build_post_body(
                            stage1["formatted_text"],
                            ai_markets=stage1.get("affects_markets", [])
                        )
                    stage1["stage"] = "caption_only"
                    return stage1

                log.info(f"Stage 1 low-confidence (conf={conf:.2f}) → proceeding to visual frame analysis")
            else:
                log.warning("Stage 1 failed — proceeding to visual frame analysis")
        else:
            log.info("No caption — going straight to visual frame analysis")

        if not frames:
            log.info("No frames available for Stage 2 — rejecting (no caption + no frames)")
            return {"approved": False, "reason": "No caption and no frames to analyse.", "issues": ["no_content"], "confidence": 1.0, "engine": "pre_filter"}

        log.info(f"🎥 Video Stage 2 — visual analysis | frames={len(frames)}")
        prompt = _VIDEO_VISUAL_PROMPT.format(caption=caption or "(no caption)")

        parts = []
        for frame_bytes in frames[:4]:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": _b64(frame_bytes)
                }
            })
        parts.append(prompt)

        try:
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self._gemini_vision.generate_content(parts)
                ),
                timeout=45
            )
            data = _parse_json(resp.text)
            data["engine"] = "gemini-2.5-flash-vision"
            data["stage"] = "visual_frames"
            log.info(f"Stage 2 Gemini → approved={data.get('approved')} | "
                     f"visual_confirmed={data.get('visual_confirmed')} | "
                     f"conf={data.get('confidence', 0):.2f}")
            if data.get("approved") and data.get("formatted_text"):
                data["formatted_text"] = _build_post_body(
                    data["formatted_text"],
                    ai_markets=data.get("affects_markets", [])
                )
            return data
        except Exception as exc:
            log.warning(f"Stage 2 Gemini Vision failed ({exc}) — trying Groq …")

        try:
            content = []
            for frame_bytes in frames[:4]:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_b64(frame_bytes)}"}
                })
            content.append({"type": "text", "text": prompt})
            
            resp = await asyncio.wait_for(
                self._groq.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": content}],
                    response_format={"type": "json_object"},
                    temperature=0.1, max_tokens=600,
                ),
                timeout=60,
            )
            data = _parse_json(resp.choices[0].message.content)
            data["engine"] = "groq-llama4-scout-vision"
            data["stage"] = "visual_frames"
            log.info(f"Stage 2 Groq → approved={data.get('approved')} | conf={data.get('confidence', 0):.2f}")
            if data.get("approved") and data.get("formatted_text"):
                data["formatted_text"] = _build_post_body(
                    data["formatted_text"],
                    ai_markets=data.get("affects_markets", [])
                )
            return data
        except Exception as exc:
            log.error(f"Stage 2 both engines failed: {exc}")
            return {"approved": False, "reason": "AI engines unavailable for video analysis.", "engine": "engine_error", "confidence": 0.0}

    async def _try_engines_text(self, prompt: str,
                                timeout_gemini: int = 30,
                                timeout_groq: int = 40) -> Optional[dict]:
        try:
            result = await asyncio.wait_for(
                self._gemini_call(prompt, None, "image/jpeg"), timeout=timeout_gemini
            )
            result["engine"] = "gemini-2.5-flash"
            return result
        except Exception as exc:
            log.warning(f"_try_engines_text Gemini failed: {exc}")
            
        try:
            result = await asyncio.wait_for(
                self._groq_call(prompt, None, "image/jpeg"), timeout=timeout_groq
            )
            result["engine"] = "groq-llama4-scout"
            return result
        except Exception as exc:
            log.warning(f"_try_engines_text Groq failed: {exc}")
        return None

    def _build_moderation_prompt(self, text: str) -> str:
        return textwrap.dedent(f"""
            DATE (UTC): {_today_str()}
            CHANNEL FOCUS: {self._category}
            SOURCE CONTENT:
            \"\"\"
            {text.strip() if text else "(image only — no text)"}
            \"\"\"
            TASK: Analyse content. If relevant geopolitical/macro news OR actual released economic data, approve and format.
            If forecast/previous values, signal, TA, meme, off-topic, stale — reject.
            Format according to rules. Return JSON.
        """).strip()

    async def _gemini_call(self, prompt: str, image_data: Optional[bytes],
                           image_mime: str) -> dict:
        parts = []
        if image_data:
            parts.append({"inline_data": {"mime_type": image_mime, "data": _b64(image_data)}})
        parts.append(prompt)
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: self._gemini.generate_content(parts))
        return _parse_json(resp.text)

    async def _gemini_text_call(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: self._gemini_text.generate_content(prompt))
        return resp.text

    async def _groq_call(self, prompt: str, image_data: Optional[bytes],
                         image_mime: str) -> dict:
        content = []
        if image_data:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{image_mime};base64,{_b64(image_data)}"}})
        content.append({"type": "text", "text": prompt})
        
        resp = await self._groq.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                      {"role": "user", "content": content}],
            response_format={"type": "json_object"},
            temperature=0.15, max_tokens=600,
        )
        return _parse_json(resp.choices[0].message.content)

    async def _groq_text_call(self, prompt: str) -> str:
        resp = await self._groq.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=1200,
        )
        return resp.choices[0].message.content

    @staticmethod
    def _fallback_alert(event: dict, minutes_left: int) -> str:
        emoji = "🔴" if event.get("impact") == "red" else "🟠"
        event_name = event.get("name", "Unknown Event")
        be_careful = _get_be_careful_line(event_name)
        text = (
            f"🚨 ALERT: {minutes_left} MINUTES REMAINING\n\n"
            f"{emoji} {event_name}\n"
            f"🕒 {event.get('time_12h', '—')}\n\n"
            f"REQUIRED ACTION:\n"
            f"✅ Secure open profits now\n"
            f"✅ Move Stop-Loss to Break-even\n"
            f"✅ No new entries during the release\n\n"
            f"{be_careful}"
        )
        text = _add_us_flag_emoji(text)
        return _add_signature(text)


def _build_post_body(text: str, ai_markets: list = None) -> str:
    """
    Build final post body.
    ai_markets: list from AI affects_markets e.g. ["XAUUSD", "OIL"]
    """
    if not text:
        return ""
        
    text = text.replace("*", "")
    text = re.sub(r"📌\s*(NOTE|MARKET STATUS|STATUS)[^\n]*\n?", "", text).strip()
    text = _strip_be_careful(text)
    text = _strip_predictions(text)
    text = _strip_ai_filler(text)
    
    lines = text.split('\n')
    for i in range(max(0, len(lines) - 3), len(lines)):
        # አመተምህረቶችን ብቻ (20xx) ይሰርዛል እንጂ የዋጋ መጠንን አያጠፋም
        lines[i] = re.sub(r'\b20\d{2}\b', '', lines[i])
        
    text = '\n'.join(lines)
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()

    if "TODAY'S USD" in text or "WEEKLY HIGH IMPACT" in text:
        text = re.sub(r"#\w+", "", text).strip()
    else:
        valid = {"XAUUSD", "OIL", "DXY"}
        clean_markets = [m.upper() for m in (ai_markets or []) if m.upper() in valid]
        clean_text = re.sub(r"#\w+", "", text).strip()
        if clean_markets:
            hashtags = " ".join(f"#{m}" for m in clean_markets)
            log.debug(f"Hashtags from AI: {hashtags}")
        else:
            hashtags = _detect_hashtags(text)
            log.debug(f"Hashtags from keyword fallback: {hashtags or '(none)'}")
        text = clean_text + ("\n\n" + hashtags if hashtags else "")

    text = _add_signature(text)
    return text
