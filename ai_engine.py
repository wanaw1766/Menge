"""
ai_engine.py — AXIOM INTEL Final
- Preserves exact numbers/prices.
- Rejects TA charts with markings.
- Hashtags only #XAUUSD, #OIL, #DXY.
- ForexFactory calendar detection (daily/weekly) with date‑range logic.
- Duplicate detection via AI similarity.
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
ALLOWED_HASHTAGS_SET = {"#XAUUSD", "#DXY", "#OIL"}


def _add_signature(text: str) -> str:
    """Append channel signature (with 25% chance of 💡 emoji)."""
    text = text.strip()
    if "[Squad 4xx]" not in text:
        if random.random() < 0.25:
            text += "\n\n💡 [Squad 4xx](https://t.me/Squad_4xx)"
        else:
            text += "\n\n[Squad 4xx](https://t.me/Squad_4xx)"
    return text


def _add_us_flag_emoji(text: str) -> str:
    """Add 🇺🇸 flag after first 'US' or 'USD' in the first line."""
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
    """Remove any AI-generated 'Be careful' line – we add our own controlled version."""
    return re.sub(r'\n?Be careful[^\n]*\n?', '', text, flags=re.IGNORECASE).strip()


def _get_be_careful_line(event_name: str) -> str:
    """Return a short, event‑specific warning for high‑impact releases."""
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
Take the source content, clean it (remove markdown, extra spaces, non-essential fluff), and format it cleanly.
Do NOT change the meaning. Do NOT rewrite headlines. Do NOT alter numbers, prices, percentages, or dollar amounts.
Do NOT speculate. Do NOT add analysis beyond the facts.

CRITICAL FORMATTING RULES:
- DO NOT use asterisks (*) or any markdown bolding.
- Use ONLY plain text and emojis.
- NO NOTE line. NO MARKET STATUS. NO commentary line.
- **PRESERVE EXACT NUMBERS, PRICES, PERCENTAGES, AND DOLLAR AMOUNTS** – copy them exactly as they appear.
- **DO NOT summarise or paraphrase** – keep the original wording as close as possible. Only remove obvious spam, emojis, or markdown.
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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. SIGNALS       — Buy/Sell/Long/Short/Entry/TP/SL/price targets
2. CHART / TA    — Technical analysis, patterns, indicators
3. MEME          — Memes, jokes, informal content
4. ANALYSIS IMG  — Chart screenshots with TA annotations (drawn lines, arrows, circles, RSI, MACD, support/resistance)
5. WATERMARK     — Another channel logo or username
6. STALE         — Content older than 18 hours
7. OFF-TOPIC     — Not about geopolitics, central banks, macro data, Gold, Oil, USD
8. LOW VALUE     — Vague, no specific real-world event
9. DUPLICATE     — Same story already processed
10. PREDICTION   — "I think", "expect", "my analysis", forecasts
11. COMMENTARY   — Personal views, market opinions
12. FORECAST/PREVIOUS — Any mention of "forecast", "expected", "previous" values
13. TA CHART WITH MARKINGS — Any image containing drawn lines, arrows, circles, text annotations, RSI, MACD, or any indicator overlay.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT (if approved):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[EMOJI] [SHORT ENGLISH HEADLINE — one line, factual, preserving original numbers]

[Source content lightly cleaned. 2-4 sentences max. Keep exact prices/data.]

[Relevant hashtags: #XAUUSD #DXY #OIL — only those that apply]

EMOJI: 🚨 🌍 📊 🏦 🛢️ 🏆 💵 ⚠️ 🗳️

RESPOND WITH VALID JSON ONLY — NO MARKDOWN FENCES — NO TRAILING COMMAS.
""".strip()  # No trailing comma in JSON example

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

EXPECTED DATE: {today_date} (example: "Friday, May 1" — no year).
The screenshot may show the same date with or without the year. Ignore the year if present.
Only approve if the date matches (day and month), regardless of year.

INSTRUCTIONS:
1. Check the date in the screenshot — must match {today_date} (year optional). If not → reject.
2. Extract ALL USD high-impact (🔴) and medium-impact (🟠) events.
   Write EACH event on its OWN separate line — even if same time.
3. Do NOT include the year in the date line (e.g. "Thursday, April 30").
4. Do NOT add any hashtags.
5. No forecast, no previous data, no NOTE, no commentary.
6. Do NOT add "Be careful" line — it is added automatically.
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
- 12-hour AM/PM, no leading zero (3:30 PM not 03:30 PM)
- Plain text only — no asterisks, no bold
- Do NOT add signature or "Be careful" line

If not valid FF calendar for today → {{"approved": false, "reason": "not valid FF today image"}}
If valid → {{"approved": true, "reason": "valid FF today image", "formatted_text": "..."}}
RESPOND WITH VALID JSON ONLY.
"""

_FF_WEEKLY_IMAGE_PROMPT = """
You are analysing a ForexFactory calendar for the weekly outlook.
The image MUST show a date range (e.g., "Week of May 4 – May 11, 2026") or multiple day headers (e.g., "Monday — May 4", "Tuesday — May 5").
If the image shows only a single date (e.g., "Friday, May 1"), reject it as not weekly.

INSTRUCTIONS:
- Identify the week range (e.g., "May 4 – May 10, 2026").
- Extract ALL USD high-impact (🔴) and medium-impact (🟠) events for the entire week.
- Group events by day. For each day, write "Monday — May 4" (no year).
- Each event on its own line under the day.
- Use 12-hour AM/PM, no leading zero.
- No forecast, no previous data, no hashtags.
- Do NOT add "Be careful" line (added automatically).

Output format example:
WEEKLY HIGH IMPACT NEWS
Week of May 4 – May 8, 2026

Monday — May 4
🔴 3:30 PM | USD: ISM Manufacturing PMI

Tuesday — May 5
🟠 10:00 AM | USD: JOLTS Job Openings

If the image is not a weekly calendar (single date only or no date range), respond with {"approved": false, "reason": "not a weekly calendar (single date)"}.

If valid:
{"approved": true, "reason": "valid weekly calendar", "formatted_text": "..."}
"""


def _b64(data: bytes) -> str:
    """Base64 encode bytes for inclusion in Gemini API."""
    return base64.b64encode(data).decode()


def _today_str() -> str:
    """Current UTC date/time for logging."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _parse_json(raw: str) -> dict:
    """Robust JSON extraction from AI response (removes markdown fences, fixes trailing commas)."""
    if not raw:
        raise ValueError("Empty response from AI engine.")
    raw = re.sub(r"```+(?:json|JSON)?", "", raw)
    raw = re.sub(r"```+", "", raw)
    raw = raw.strip().strip("`").strip()
    raw = re.sub(r",\s*([}\]])", r"\1", raw)   # remove trailing commas
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
    """Apply post‑processing to AI output: strip markdown, add hashtags, remove forbidden lines."""
    data.setdefault("approved", False)
    data.setdefault("reason", "")
    data.setdefault("issues", [])
    data.setdefault("formatted_text", "")
    data.setdefault("confidence", 0.5)

    if data.get("formatted_text"):
        # Remove markdown asterisks
        data["formatted_text"] = data["formatted_text"].replace("*", "")
        # Remove any NOTE/MARKET STATUS lines
        data["formatted_text"] = re.sub(
            r"📌\s*(NOTE|MARKET STATUS|STATUS)[^\n]*\n?", "", data["formatted_text"]
        ).strip()
        # Remove any AI‑generated "Be careful" line
        data["formatted_text"] = _strip_be_careful(data["formatted_text"])
        text = data["formatted_text"]

        # For calendar posts: remove any stray hashtags (they are not wanted)
        if "TODAY'S USD HIGH IMPACT" in text or "WEEKLY HIGH IMPACT" in text:
            text = re.sub(r"#\w+", "", text).strip()
            data["formatted_text"] = text
        else:
            # For news posts: extract and filter allowed hashtags
            hashtags = re.findall(r"#\w+", text)
            allowed_hashtags = [h for h in hashtags if h in ALLOWED_HASHTAGS_SET]
            text = re.sub(r"#\w+", "", text).strip()
            # Auto‑add hashtags based on content if missing
            if re.search(r'\bgold\b|\bxau\b', text, re.IGNORECASE):
                if "#XAUUSD" not in allowed_hashtags:
                    allowed_hashtags.append("#XAUUSD")
            if re.search(r'\boil\b|\bwti\b|\bbrent\b', text, re.IGNORECASE):
                if "#OIL" not in allowed_hashtags:
                    allowed_hashtags.append("#OIL")
            if re.search(r'\bdxy\b|\busd\b', text, re.IGNORECASE):
                if "#DXY" not in allowed_hashtags:
                    allowed_hashtags.append("#DXY")
            if allowed_hashtags:
                text = text + "\n\n" + " ".join(allowed_hashtags)
            data["formatted_text"] = text

    # Hard reject if any signal word remains
    if data.get("approved") and _signal_hit(data.get("formatted_text", "")):
        log.warning("Signal keyword in output — hard reject.")
        data["approved"] = False
        data["reason"] = "Signal keyword found in output."
        data["issues"].append("signal_content")
        data["formatted_text"] = ""

    return data


def _signal_hit(text: str) -> Optional[str]:
    """Detect trading signal keywords."""
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
    """Main AI interface – uses Gemini as primary, Groq as fallback."""

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
        """Analyse a news message (text + optional image) and return approval verdict."""
        prompt = self._build_moderation_prompt(text)
        try:
            verdict = await asyncio.wait_for(
                self._gemini_call(prompt, image_data, image_mime), timeout=40
            )
            verdict["engine"] = "gemini-2.5-flash"
            log.info(f"Gemini → approved={verdict['approved']} | {verdict.get('reason', '')}")
            if verdict.get("approved") and verdict.get("formatted_text"):
                verdict["formatted_text"] = _build_post_body(verdict["formatted_text"])
                if not verdict["formatted_text"].startswith("TODAY'S USD HIGH IMPACT"):
                    verdict["formatted_text"] = _add_us_flag_emoji(verdict["formatted_text"])
            return verdict
        except Exception as exc:
            log.warning(f"Gemini failed ({exc}) — trying Groq …")

        try:
            verdict = await asyncio.wait_for(
                self._groq_call(prompt, image_data, image_mime), timeout=55
            )
            verdict["engine"] = "groq-llama4-scout"
            log.info(f"Groq → approved={verdict['approved']} | {verdict.get('reason', '')}")
            if verdict.get("approved") and verdict.get("formatted_text"):
                verdict["formatted_text"] = _build_post_body(verdict["formatted_text"])
                if not verdict["formatted_text"].startswith("TODAY'S USD HIGH IMPACT"):
                    verdict["formatted_text"] = _add_us_flag_emoji(verdict["formatted_text"])
            return verdict
        except Exception as exc:
            log.error(f"Both AI engines failed — safe reject.")
            return {
                "approved": False,
                "reason": "Both AI engines unavailable.",
                "issues": ["engine_error"],
                "formatted_text": "",
                "confidence": 0.0,
                "engine": "none"
            }

    async def is_same_story(self, text_a: str, text_b: str,
                            image_a: Optional[bytes] = None,
                            image_b: Optional[bytes] = None) -> bool:
        """Return True if two stories (text+image) describe the same real‑world event."""
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
        """Analyse a ForexFactory calendar image (daily or weekly)."""
        if is_weekly:
            prompt = _FF_WEEKLY_IMAGE_PROMPT.format(week_range=week_range)
        else:
            prompt = _FF_IMAGE_PROMPT.format(today_date=today_date)
        parts = [{"inline_data": {"mime_type": image_mime, "data": _b64(image_data)}}, prompt]
        try:
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._gemini_vision.generate_content(parts)),
                timeout=60
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
                    temperature=0.1, max_tokens=800,
                ),
                timeout=75,
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
        """Public wrapper to get an event‑specific caution line."""
        return _get_be_careful_line(event_name)

    def _build_moderation_prompt(self, text: str) -> str:
        """Build the prompt for moderation (with current date and channel focus)."""
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
        """Call Gemini with optional image."""
        parts = []
        if image_data:
            parts.append({"inline_data": {"mime_type": image_mime, "data": _b64(image_data)}})
        parts.append(prompt)
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: self._gemini.generate_content(parts))
        return _parse_json(resp.text)

    async def _groq_call(self, prompt: str, image_data: Optional[bytes],
                         image_mime: str) -> dict:
        """Call Groq with optional image."""
        content = []
        if image_data:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{image_mime};base64,{_b64(image_data)}"}})
        content.append({"type": "text", "text": prompt})
        resp = await self._groq.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                      {"role": "user", "content": content}],
            temperature=0.15, max_tokens=600,
        )
        return _parse_json(resp.choices[0].message.content)


def _build_post_body(text: str) -> str:
    """Final formatting before sending to Telegram: remove residual markdown, add signature."""
    if not text:
        return ""
    text = text.replace("*", "")
    text = re.sub(r"📌\s*(NOTE|MARKET STATUS|STATUS)[^\n]*\n?", "", text).strip()
    text = _strip_be_careful(text)
    lines = text.split('\n')
    # Remove stray year numbers from the last few lines
    for i in range(max(0, len(lines) - 3), len(lines)):
        lines[i] = re.sub(r'\b\d{4}\b', '', lines[i])
    text = '\n'.join(lines)
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()
    text = _add_signature(text)
    return text
