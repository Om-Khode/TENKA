"""IP-based region detection for browser planning prompts."""

import logging
import os
import re

logger = logging.getLogger("geolocation")

_SAFE_RE = re.compile(r"[^\w ,/+\-().]", re.UNICODE)

_cached_region: dict | None = None


def _sanitize(val: str) -> str:
    return _SAFE_RE.sub("", val)[:80]


def _parse_region_response(data: dict) -> dict:
    return {
        "country": _sanitize(str(data.get("country", ""))),
        "country_code": _sanitize(str(data.get("countryCode", ""))),
        "city": _sanitize(str(data.get("city", ""))),
        "timezone": _sanitize(str(data.get("timezone", ""))),
    }


def _env_fallback() -> dict:
    code = os.getenv("USER_REGION", "")
    tz = os.getenv("USER_TIMEZONE", "")
    return {
        "country": "",
        "country_code": code,
        "city": "",
        "timezone": tz,
    }


async def detect_region() -> dict:
    global _cached_region
    if _cached_region is not None:
        return _cached_region

    # Preference override: /set user_region JP takes precedence
    try:
        from assistant.core.runtime_config import _get_db_value
        pref_code = _get_db_value("user_region")
        pref_tz = _get_db_value("user_timezone")
        if pref_code:
            _cached_region = {
                "country": "", "country_code": str(pref_code),
                "city": "", "timezone": str(pref_tz or ""),
            }
            logger.info(f"[GEO] Using preference override: {_cached_region}")
            return _cached_region
    except Exception:
        pass

    try:
        import asyncio
        import json
        import urllib.request

        def _fetch():
            req = urllib.request.Request(
                "http://ip-api.com/json/?fields=country,countryCode,city,timezone",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                return json.loads(resp.read().decode())

        data = await asyncio.to_thread(_fetch)
        _cached_region = _parse_region_response(data)
        logger.info(f"[GEO] Detected: {_cached_region}")
        return _cached_region
    except Exception as e:
        logger.info(f"[GEO] IP detection failed ({e}), using env fallback")

    _cached_region = _env_fallback()
    return _cached_region


def get_cached_region() -> dict:
    if _cached_region is not None:
        return _cached_region
    return _env_fallback()


def format_region_hint(region: dict) -> str:
    country = region.get("country", "")
    city = region.get("city", "")
    tz = region.get("timezone", "")
    if not country and not tz:
        return ""
    loc = f"{city}, {country}" if city else country
    return f"[User location: {loc} | Timezone: {tz}]"
