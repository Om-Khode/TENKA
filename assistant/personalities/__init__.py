"""Personality loading and switching.

Each personality is a subfolder with prompt.txt, traits.json,
and optionally modifiers.json and responses.json.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

_logger = logging.getLogger("personalities")

_PERSONALITIES_DIR = Path(__file__).parent


class PersonalityLoader:
    BUILTIN = ("warm_honest", "tsundere", "minimal")
    DEFAULT = "warm_honest"

    def __init__(self, name: str) -> None:
        if name not in self.BUILTIN:
            raise ValueError(
                f"Unknown personality '{name}'. Must be one of: {self.BUILTIN}"
            )
        self._name = name
        self._dir = _PERSONALITIES_DIR / name
        self._prompt: Optional[str] = None
        self._traits_config: Optional[dict] = None
        self._modifiers: Optional[dict] = None
        self._responses: Optional[dict] = None
        self._load()

    @property
    def name(self) -> str:
        return self._name

    def _load(self) -> None:
        from assistant import config

        prompt_path = self._dir / "prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Missing {prompt_path}")
        raw = prompt_path.read_text(encoding="utf-8").strip()
        self._prompt = raw.replace("{ASSISTANT_NAME}", config.ASSISTANT_NAME_DISPLAY)

        traits_path = self._dir / "traits.json"
        with open(traits_path, encoding="utf-8") as f:
            self._traits_config = json.load(f)

        modifiers_path = self._dir / "modifiers.json"
        if self._traits_config.get("has_modifiers", True) and modifiers_path.exists():
            with open(modifiers_path, encoding="utf-8") as f:
                self._modifiers = json.load(f)
        else:
            self._modifiers = {}

        responses_path = self._dir / "responses.json"
        if responses_path.exists():
            with open(responses_path, encoding="utf-8") as f:
                self._responses = json.load(f)
            if self._name == "tsundere" and os.getenv("VOCAL_CASUAL_LANGUAGE", "false").lower() == "true":
                extras = self._responses.pop("_casual_extras", {})
                for key, variants in extras.items():
                    if key in self._responses:
                        self._responses[key].extend(variants)
        else:
            self._responses = {}

        _logger.info(f"[PERSONALITY] Loaded '{self._name}' personality")

    def get_prompt_base(self) -> str:
        return self._prompt

    def get_trait_defaults(self) -> dict[str, dict[str, float]]:
        return self._traits_config["defaults"]

    def get_emotion_mode(self) -> str:
        return self._traits_config.get("emotion_mode", "neutral")

    def get_reflection_hints(self) -> dict[str, str]:
        return self._traits_config.get("reflection_hints", {})

    def get_modifiers(self) -> dict[str, dict[str, str]]:
        return self._modifiers

    def get_feature_flags(self) -> dict[str, bool]:
        return {
            "sycophancy_filter": self._traits_config.get("sycophancy_filter", False),
            "wellbeing_checkin": self._traits_config.get("wellbeing_checkin", False),
        }

    def get_responses(self) -> dict[str, list[str]]:
        return self._responses


# ─── Module-level active personality ────────────────────────────────────────

_active_loader: Optional[PersonalityLoader] = None


def get_active_loader() -> PersonalityLoader:
    global _active_loader
    if _active_loader is None:
        _active_loader = PersonalityLoader(PersonalityLoader.DEFAULT)
    return _active_loader


_just_switched: bool = False


def set_active_personality(name: str) -> None:
    global _active_loader, _just_switched
    _active_loader = PersonalityLoader(name)
    _just_switched = True


def consume_switch_flag() -> bool:
    """Return True (once) if personality was switched since last check."""
    global _just_switched
    was = _just_switched
    _just_switched = False
    return was


def get_active_personality_id() -> str:
    return get_active_loader().name
