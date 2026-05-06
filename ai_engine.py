"""
ai_engine.py — AXIOM INTEL (Groq-only, robust JSON, fixed FF detection)
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

from groq import AsyncGroq

log = logging.getLogger("ai_engine")

CHANNEL_SIGNATURE = "\n\n[Squad 4xx](https://t.me/Squad_4xx)"
ALLOWED_HASHTAGS_SET = {"#XAUUSD", "#DXY", "#OIL"}

def _add_signature(text: str) -> str:
    text = text.strip()
    if "[Squad 4xx]" not in text:
        if random.random() < 0.25:
            text += "\n\n💡 [Squad 4xx](https://t.me/Squad_4xx)"
        else:
            text += "\n\n[Squad 4xx](https://t.me/Squad_4xx)"
    return text

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
You are AXIOM INTEL — a Senior Institutional Macro & Geopolitical news editor.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 GEOPOLITICAL EXCEPTION (ALWAYS APPROVE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Any statement from a world leader (e.g., Trump, Biden, Putin, Xi) that affects:
- Oil supply (Hormuz, OPEC, embargo, sanctions)
- War / conflict escalation
- Tariffs / trade restrictions
- Central bank or financial policy changes
- Gold, USD, or energy markets
These are HIGH IMPACT geopolitical events, even if posted on social media.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 FOMC / CENTRAL BANK EXCEPTION (ALWAYS APPROVE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Any official announcement or news about:
- Federal Open Market Committee (FOMC)
- Federal Funds Rate / Interest Rate Decision
- Fed Chair Powell speech
- FOMC Statement or Minutes
These are HIGH IMPACT macroeconomic events. Always approve even if they contain numbers like "rate at 5.25%". Do NOT reject as "forecast" or "commentary".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR ONLY JOB:
Take the source content, clean it, and format it cleanly.
Do NOT change the meaning. Do NOT rewrite headlines. Do NOT alter numbers, prices, percentages, or dollar amounts.
Do NOT speculate. Do NOT add analysis beyond the facts.

CRITICAL FORMATTING RULES:
- DO NOT use asterisks (*) or any markdown bolding.
- Use ONLY plain text and emojis.
- NO NOTE line. NO MARKET STATUS. NO commentary line.
- **PRESERVE EXACT NUMBERS, PRICES, PERCENTAGES, AND DOLLAR AMOUNTS** – copy them exactly as they appear.
- **DO NOT summarise or paraphrase** – keep the original wording as close as possible.
- Actual released figures (e.g., "came at 2.5%") are ALLOWED and must be copied exactly.
- Forecast (expected) and previous values are FORBIDDEN. Never include them.
- Technical analysis, signals, predictions, opinions are FORBIDDEN.
- Hashtags: Only use #XAUUSD, #DXY, or #OIL – only those relevant to the story.
  - For gold (XAU) → #XAUUSD
  - For USD strength/weakness → #DXY
  - For crude oil/WTI/Brent → #OIL
- Do NOT add the current year at the end of posts.
- Do NOT add signature (added automatically).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REJECT IF ANY OF THESE APPLY:
1. SIGNALS       — Buy/Sell/Long/Short/Entry/TP/SL/price targets
2. CHART / TA    — Technical analysis, patterns, indicators
3. MEME          — Memes, jokes, informal content
4. ANALYSIS IMG  — Chart screenshots with TA annotations (drawn lines, arrows, circles, RSI, MACD)
5. WATERMARK     — Another channel logo or username
6. STALE         — Content older than 18 hours
7. OFF-TOPIC     — Not about geopolitics, central banks, macro data, Gold, Oil, USD
8. LOW VALUE     — Vague, no specific real-world event
9. DUPLICATE     — Same story already processed
10. PREDICTION   — "I think", "expect", "my analysis", forecasts
11. COMMENTARY   — Personal views, market opinions
12. FORECAST/PREVIOUS — Any mention of "forecast", "expected", "previous" values
13. TA CHART WITH MARKINGS — Any image containing drawn lines, arrows, circles, text annotations, RSI, MACD, or indicator overlay.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT (if approved):
[EMOJI] [SHORT ENGLISH HEADLINE — one line, factual, preserving original numbers]

[Source content lightly cleaned. 2-4 sentences max. Keep exact prices/data.]

[Relevant hashtags: #XAUUSD #DXY #OIL — only those that apply]

EMOJI: 🚨 🌍 📊 🏦 🛢️ 🏆 💵 ⚠️ 🗳️

RESPOND WITH VALID JSON ONLY — NO MARKDOWN FENCES — NO TRAILING COMMAS.
"""

_SIMILARITY_PROMPT = """
You are a duplicate news detector. Compare the two stories. If they describe the same real-world event – even if worded differently – respond with same_story=true.
Be aggressive. If any chance they are the same, mark true.

Story A: {story_a}
Story B: {story_b}

Respond in JSON: {{"same_story": true/false, "confidence": 0.0-1.0, "reason": "..."}}
"""

_MULTIMODAL_SIMILARITY_PROMPT = """
You are a duplicate news detector. Compare the two items (text + optional images). Decide if they are the SAME real-world event.

Item A text: {text_a}
Item B text: {text_b}
(Images compared visually if both exist)

Be aggressive: if any chance they are the same, mark same_story=true.

Respond in JSON: {{"same_story": true, "confidence": 0.0-1.0, "reason": "..."}}
"""

_FF_IMAGE_PROMPT = """
You are analysing a ForexFactory economic calendar screenshot.

EXPECTED DATE: {today_date} (example: "Friday, May 1" — no year).
The screenshot may show the same date with or without the year. Ignore the year.
Only approve if the date matches (day and month), regardless of year.

INSTRUCTIONS:
1. Check date in screenshot — must match {today_date} (year optional). If not → reject.
2. Extract ALL USD high-impact (🔴) and medium-impact (🟠) events.
   Write EACH event on its OWN separate line — even if same time.
3. Do NOT include the year in the date line (e.g. "Thursday, April 30").
4. Do NOT add any hashtags.
5. No forecast, no previous data, no NOTE, no commentary.
6. Do NOT add "Be careful" line — added automatically.
7. Keep original times. Convert to 12-hour AM/PM. No timezone. No leading zero.

EXACT OUTPUT FORMAT:

TODAY'S USD HIGH IMPACT NEWS
Thursday, April 30

🔴 3:30 PM | USD: Advance GDP q/q
🔴 3:30 PM | USD: Core PCE Price Index m/m
🔴 3:30 PM | USD: Employment Cost Index q/q
🟠 5:00 PM | USD: Unemployment Claims

RULES:
- Only USD events (🔴 and 🟠 only)
- Each event on its own line
- 12-hour AM/PM, no leading zero
- Plain text — no asterisks, no bold
- No signature or "Be careful" line

If not valid FF calendar for today: {"approved": false, "reason": "not valid FF today image"}
If valid: {"approved": true, "reason": "valid FF today image", "formatted_text": "..."}
"""

_FF_WEEKLY_IMAGE_PROMPT = """
You are analysing a ForexFactory calendar for the weekly outlook.
The image MUST show a date range (e.g., "Week of May 4 – May 11") or multiple day headers.
If only a single date appears, reject as not weekly.

INSTRUCTIONS:
- Identify the week range (e.g., "May 4 – May 10").
- Extract ALL USD high-impact (🔴) and medium-impact (🟠) events for the entire week.
- Group events by day. Write "Monday — May 4" (no year).
- Each event on its own line under the day.
- Use 12-hour AM/PM, no leading zero.
- No forecast, previous data, hashtags.
- Do NOT add "Be careful" line.

Output example:
WEEKLY HIGH IMPACT NEWS
Week of May 4 – May 8, 2026

Monday — May 4
🔴 3:30 PM | USD: ISM Manufacturing PMI

Tuesday — May 5
🟠 10:00 AM | USD: JOLTS Job Openings

If not a weekly calendar: {"approved": false, "reason": "not a weekly calendar (single date)"}
If valid: {"approved": true, "reason": "valid weekly calendar", "formatted_text": "..."}
"""

def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()

def _repair_json(raw: str) -> dict:
    """Repair Groq's malformed JSON."""
    raw = re.sub(r"```+(?:json|JSON)?", "", raw)
    raw = re.sub(r"```+", "", raw)
    raw = re.sub(r'//.*?(\n|$)', '\n', raw)  # remove comments
    raw = re.sub(r',\s*([}\]])', r'\1', raw)  # remove trailing commas
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found")
    candidate = m.group()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        # try single quotes
        candidate = re.sub(r"'([^']*)':", r'"\1":', candidate)
        candidate = re.sub(r": '([^']*)'", r': "\1"', candidate)
        data = json.loads(candidate)
    # Normalise keys
    if "content" in data and "formatted_text" not in data:
        data["formatted_text"] = data.pop("content")
    return data

def _validate_and_clean(data: dict) -> dict:
    data.setdefault("approved", False)
    data.setdefault("reason", "")
    data.setdefault("issues", [])
    data.setdefault("formatted_text", "")
    data.setdefault("confidence", 0.5)

    if data.get("formatted_text"):
        data["formatted_text"] = data["formatted_text"].replace("*", "")
        data["formatted_text"] = re.sub(r"📌\s*(NOTE|MARKET STATUS|STATUS)[^\n]*\n?", "", data["formatted_text"]).strip()
        data["formatted_text"] = _strip_be_careful(data["formatted_text"])
        text = data["formatted_text"]
        if "TODAY'S USD HIGH IMPACT" in text or "WEEKLY HIGH IMPACT" in text:
            text = re.sub(r"#\w+", "", text).strip()
            data["formatted_text"] = text
        else:
            hashtags = re.findall(r"#\w+", text)
            allowed = [h for h in hashtags if h in ALLOWED_HASHTAGS_SET]
            text = re.sub(r"#\w+", "", text).strip()
            if re.search(r'\bgold\b|\bxau\b', text, re.IGNORECASE) and "#XAUUSD" not in allowed:
                allowed.append("#XAUUSD")
            if re.search(r'\boil\b|\bwti\b|\bbrent\b', text, re.IGNORECASE) and "#OIL" not in allowed:
                allowed.append("#OIL")
            if re.search(r'\bdxy\b|\busd\b', text, re.IGNORECASE) and "#DXY" not in allowed:
                allowed.append("#DXY")
            if allowed:
                text = text + "\n\n" + " ".join(allowed)
            data["formatted_text"] = text
    return data

class AIEngine:
    def __init__(self, gemini_key: str, groq_key: str, channel_category: str):
        # gemini_key ignored – using only Groq
        self._groq = AsyncGroq(api_key=groq_key)
        self._category = channel_category

    async def analyse(self, text: str, image_data: Optional[bytes] = None,
                      image_mime: str = "image/jpeg") -> dict:
        prompt = self._build_moderation_prompt(text)
        try:
            content = []
            if image_data:
                content.append({"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{_b64(image_data)}"}})
            content.append({"type": "text", "text": prompt})
            resp = await self._groq.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": content}],
                temperature=0.15, max_tokens=600,
            )
            data = _repair_json(resp.choices[0].message.content)
            data = _validate_and_clean(data)
            data["engine"] = "groq"
            log.info(f"Groq analyse → approved={data['approved']} | {data.get('reason','')}")
            if data.get("approved") and data.get("formatted_text"):
                data["formatted_text"] = _build_post_body(data["formatted_text"])
                if not data["formatted_text"].startswith("TODAY'S USD HIGH IMPACT"):
                    data["formatted_text"] = _add_us_flag_emoji(data["formatted_text"])
            return data
        except Exception as e:
            log.error(f"Groq analyse failed: {e}")
            return {"approved": False, "reason": str(e), "issues": ["engine_error"], "formatted_text": "", "confidence": 0.0}

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
            content = []
            if image_a:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(image_a)}"}})
            if image_b:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(image_b)}"}})
            content.append({"type": "text", "text": prompt})
            resp = await self._groq.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": content}],
                temperature=0.1, max_tokens=300,
            )
            data = _repair_json(resp.choices[0].message.content)
            same = bool(data.get("same_story", False))
            conf = data.get("confidence", 0)
            log.info(f"Similarity → same={same} | conf={conf}")
            return same and conf >= 0.55
        except Exception as e:
            log.error(f"Similarity error: {e}")
            return False

    async def analyse_ff_image(self, image_data: bytes, image_mime: str, today_date: str,
                               is_weekly: bool = False, week_range: str = "") -> dict:
        if is_weekly:
            prompt = _FF_WEEKLY_IMAGE_PROMPT.format(week_range=week_range or "Week of ...")
        else:
            prompt = _FF_IMAGE_PROMPT.format(today_date=today_date)
        try:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{_b64(image_data)}"}},
                {"type": "text", "text": prompt},
            ]
            resp = await self._groq.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": content}],
                temperature=0.1, max_tokens=800,
            )
            data = _repair_json(resp.choices[0].message.content)
            data = _validate_and_clean(data)
            log.info(f"FF image → approved={data.get('approved')} | {data.get('reason','')}")
            if data.get("approved") and data.get("formatted_text"):
                data["formatted_text"] = _add_us_flag_emoji(data["formatted_text"])
            return data
        except Exception as e:
            log.error(f"FF image analysis failed: {e}")
            return {"approved": False, "reason": str(e)}

    async def get_be_careful_line(self, event_name: str) -> str:
        return _get_be_careful_line(event_name)

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

def _build_post_body(text: str) -> str:
    if not text:
        return ""
    text = text.replace("*", "")
    text = re.sub(r"📌\s*(NOTE|MARKET STATUS|STATUS)[^\n]*\n?", "", text).strip()
    text = _strip_be_careful(text)
    text = re.sub(r'\b\d{4}\b', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()
    text = _add_signature(text)
    return text
