"""
ai_engine.py — AXIOM INTEL AI Engine.
- Gemini primary, Groq fallback with retry.
- Hard blocks: signals, sentiment, injection. Watermarks stripped silently.
- Hashtags: #XAUUSD #DXY #OIL only where relevant, no "HASHTAGS:" label.
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
    """Remove any 'HASHTAGS:' or 'HASHTAGS -' label the AI writes."""
    return re.sub(r"HASHTAGS?\s*[:\-]?\s*\n?", "", text, flags=re.IGNORECASE).strip()


def _strip_watermarks(text: str) -> str:
    """Silently remove any @username or t.me/channel links (except our own)."""
    # Remove t.me/links except Squad_4xx
    text = re.sub(r"t\.me/(?!Squad_4xx)[a-zA-Z0-9_]+", "", text, flags=re.IGNORECASE)
    # Remove @usernames except @Squad_4xx
    text = re.sub(r"@(?!Squad_4xx)[a-zA-Z0-9_]{4,}", "", text, flags=re.IGNORECASE)
    # Clean up any leftover double spaces or orphaned punctuation from removal
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
_WATERMARK_RE = re.compile(
    r"(t\.me/(?!Squad_4xx)[a-zA-Z0-9_]+|"
    r"@(?!Squad_4xx)[a-zA-Z0-9_]{4,})",
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
    """Returns block reason or None if clean."""
    if not text:
        return None
    if _SIGNAL_RE.search(text):
        return "signal_content"
    if _SENTIMENT_RE.search(text):
        return "sentiment_content"
    if _INJECTION_RE.search(text):
        return "injection_attempt"
    return None


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

NOTE: If content contains another channel username or link — IGNORE it, treat the news itself on its merits.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT (approved posts only):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[EMOJI] [SHORT FACTUAL HEADLINE]

[2-4 sentences. Real numbers allowed. No forecast. No previous.]

ADD HASHTAGS INLINE — no label, no "HASHTAGS:" word, just the tags on their own line:
- Affects Gold → #XAUUSD
- Affects USD/Dollar → #DXY
- Affects Oil → #OIL
- FOMC/Fed → always #DXY #XAUUSD
- Geopolitical/war → depends on impact
- Never add any other hashtag

CORRECT EXAMPLE:
🚨 Iranian drones hit UAE vessel in Hormuz.

The incident occurred in the Hormuz Strait, a critical oil waterway.
Iran's involvement may trigger further sanctions or retaliation.

#OIL #XAUUSD #DXY

WRONG — never write the word HASHTAGS:
HASHTAGS:
#OIL #XAUUSD

EMOJI: 🚨 🌍 📊 🏦 🛢️ 🏆 💵 ⚠️ 🗳️ 🇺🇸

DO NOT add signature. DO NOT add year. NO asterisks. NO bold.

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

EXACT FORMAT EXAMPLE (matches real FF calendar structure):
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
Compare these two news stories. Are they about the same real-world event?
Even if worded differently or from different sources.

Story A: {story_a}
Story B: {story_b}

Be aggressive — if any reasonable chance they are same, mark true.
JSON: {{"same_story": true/false, "confidence": 0.0-1.0, "reason": "..."}}
"""


# ── Be-careful lines for reminders ────────────────────────────────────────────
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

        # Calendar posts — strip ALL hashtags
        if text.startswith("TODAY'S USD") or text.startswith("WEEKLY HIGH IMPACT"):
            text = re.sub(r"#\w+", "", text).strip()
        else:
            # Regular news — keep only allowed hashtags
            found   = re.findall(r"#\w+", text)
            allowed = [h for h in found if h in ALLOWED_HASHTAGS]
            text    = re.sub(r"#\w+", "", text).strip()
            if allowed:
                text = text.rstrip() + "\n\n" + " ".join(allowed)

        data["formatted_text"] = text

    # Hard blocks — post-AI check
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


def _build_post_body(text: str) -> str:
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
    return _add_signature(text)


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
        # No response_mime_type on vision — causes conflicts with image prompts
        self._gemini_vision = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.GenerationConfig(
                temperature=0.1, max_output_tokens=1000
            ),
        )

    # ── Main analyse ──────────────────────────────────────────────────────────

    async def analyse(self, text: str, image_data: Optional[bytes] = None,
                      image_mime: str = "image/jpeg") -> dict:

        # Pre-filter source text before AI — saves quota
        if text:
            # Strip watermarks silently before AI sees the text
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
            Analyse. If real geopolitical/macro news OR actual released data OR price level hit → approve and format.
            If forecast/previous only/signal/TA/chart/meme/sentiment/stale → reject.
            Return JSON.
        """).strip()

        # Try Gemini with one retry
        for attempt in range(2):
            try:
                verdict = await asyncio.wait_for(
                    self._gemini_call(prompt, image_data, image_mime), timeout=40
                )
                verdict["engine"] = "gemini-2.5-flash"
                if verdict.get("approved") and verdict.get("formatted_text"):
                    verdict["formatted_text"] = _build_post_body(verdict["formatted_text"])
                    if not verdict["formatted_text"].startswith("TODAY'S USD"):
                        verdict["formatted_text"] = _add_us_flag(verdict["formatted_text"])
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
                verdict["formatted_text"] = _build_post_body(verdict["formatted_text"])
                if not verdict["formatted_text"].startswith("TODAY'S USD"):
                    verdict["formatted_text"] = _add_us_flag(verdict["formatted_text"])
            return verdict
        except Exception:
            return _reject("Both AI engines unavailable", "engine_error", confidence=0.0)

    # ── FF image detection ────────────────────────────────────────────────────

    async def detect_ff_image(self, image_data: bytes, image_mime: str) -> tuple:
        """
        Returns (is_ff: bool, is_weekly: bool).
        Single AI call — detects both FF calendar and weekly/daily.
        """
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

        # Gemini with retry
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
                    data["formatted_text"] = _add_us_flag(data["formatted_text"])
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
                data["formatted_text"] = _add_us_flag(data["formatted_text"])
            return data
        except Exception:
            return {"approved": False, "reason": "AI engines unavailable for image analysis."}

    # ── Similarity check ──────────────────────────────────────────────────────

    async def is_same_story(self, text_a: str, text_b: str,
                            image_a: Optional[bytes] = None,
                            image_b: Optional[bytes] = None) -> bool:
        if not text_a and not text_b:
            return False
        prompt = _SIMILARITY_PROMPT.format(
            story_a=(text_a[:500] if text_a else ""),
            story_b=(text_b[:500] if text_b else ""),
        )
        try:
            parts = []
            if image_a:
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": _b64(image_a)}})
            parts.append(prompt)
            loop = asyncio.get_running_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self._gemini_vision.generate_content(parts)
                ),
                timeout=20
            )
            data = _parse_json(resp.text)
            return bool(data.get("same_story", False)) and data.get("confidence", 0) >= 0.55
        except Exception:
            pass
        try:
            resp = await asyncio.wait_for(
                self._groq.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1, max_tokens=300,
                ),
                timeout=25,
            )
            data = _parse_json(resp.choices[0].message.content)
            return bool(data.get("same_story", False)) and data.get("confidence", 0) >= 0.55
        except Exception:
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
