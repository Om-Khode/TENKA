"""Tests for T-items batch — THE-rule cleanup."""

import re


class TestT4TextEditorFromRegistry:
    def test_router_text_editor_regex_matches_known_apps(self):
        from assistant.core.known_apps import get_apps_by_category
        from assistant.automation.router import _TEXT_EDITOR_NAMES
        editors = get_apps_by_category("text_editor")
        for editor in editors:
            assert _TEXT_EDITOR_NAMES.search(editor), f"router regex doesn't match '{editor}'"

    def test_text_editor_regex_no_substring_match(self):
        from assistant.automation.router import _TEXT_EDITOR_NAMES
        assert not _TEXT_EDITOR_NAMES.search("encode"), "'code' should not match 'encode'"
        assert not _TEXT_EDITOR_NAMES.search("decode"), "'code' should not match 'decode'"

    def test_browser_regex_no_substring_match(self):
        from assistant.automation.router import _BROWSER_NAMES
        assert not _BROWSER_NAMES.search("wedge"), "'edge' should not match 'wedge'"
        assert not _BROWSER_NAMES.search("Fileopera"), "'opera' should not match 'Fileopera'"

    def test_no_hardcoded_text_editors_in_agent(self):
        import ast
        import pathlib
        source = pathlib.Path("assistant/automation/vision/agent.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "TEXT_EDITORS":
                        raise AssertionError("agent.py still has hardcoded TEXT_EDITORS list")


# --- T8: _APP_NOT_RUNNING_RE is generic ---

class TestT8AppNotRunningGeneric:
    def test_no_brand_names_in_regex_source(self):
        from assistant.code_executor._utils import _APP_NOT_RUNNING_RE
        pattern = _APP_NOT_RUNNING_RE.pattern
        for brand in ("spotify", "Spotify", "gmail", "whatsapp"):
            assert brand.lower() not in pattern.lower(), f"brand '{brand}' found in _APP_NOT_RUNNING_RE pattern"

    def test_matches_generic_patterns(self):
        from assistant.code_executor._utils import _APP_NOT_RUNNING_RE
        assert _APP_NOT_RUNNING_RE.search("no active devices")
        assert _APP_NOT_RUNNING_RE.search("player command failed")
        assert _APP_NOT_RUNNING_RE.search("app is not running")
        assert _APP_NOT_RUNNING_RE.search("no active playback")


# --- T9: gmail/email slug alias lookup ---

class TestT9SlugAliasLookup:
    def test_gmail_has_email_alias(self):
        from assistant.core.known_apps import KNOWN_APPS
        gmail_entry = KNOWN_APPS.get("gmail")
        assert gmail_entry is not None
        assert "email" in gmail_entry.aliases

    def test_resolve_email_to_gmail(self):
        from assistant.core.known_apps import resolve_app
        result = resolve_app("email")
        assert result == ("gmail", "email_app")


# --- T1: _DEVELOPER_URLS in service_registry ---

class TestT1DeveloperUrls:
    def test_developer_urls_in_service_registry(self):
        from assistant.service_registry import DEVELOPER_URLS
        assert "spotify" in DEVELOPER_URLS
        assert "gmail" in DEVELOPER_URLS
        assert all(url.startswith("https://") for url in DEVELOPER_URLS.values())

    def test_pending_handlers_no_local_urls(self):
        import pathlib
        source = pathlib.Path("assistant/actions/pending_handlers.py").read_text()
        assert "_DEVELOPER_URLS = {" not in source, "pending_handlers still defines its own _DEVELOPER_URLS"


# --- T16: No hardcoded "whatsapp" default ---

class TestT16NoHardcodedDefault:
    def test_no_literal_whatsapp_service_default(self):
        import pathlib
        source = pathlib.Path("assistant/main.py").read_text()
        assert 'get("service", "whatsapp")' not in source


# --- T6: _ACTION_NOUNS includes known_apps dynamically ---

class TestT6ActionNounsDynamic:
    def test_action_nouns_contains_known_apps(self):
        from assistant.actions.planner.planner import _ACTION_NOUNS
        from assistant.core.known_apps import KNOWN_APPS
        for app_name in KNOWN_APPS:
            for word in app_name.split():
                assert word in _ACTION_NOUNS, f"'{word}' from app '{app_name}' missing from _ACTION_NOUNS"

    def test_action_nouns_has_generic_terms(self):
        from assistant.actions.planner.planner import _ACTION_NOUNS
        for term in ("weather", "email", "music", "message", "file", "camera"):
            assert term in _ACTION_NOUNS


# --- T3: Browser names from known_apps ---

class TestT3BrowserNamesFromRegistry:
    def test_config_browser_names_matches_registry(self):
        from assistant.core.known_apps import get_apps_by_category
        from assistant import config
        registry_browsers = set(get_apps_by_category("browser"))
        for browser in registry_browsers:
            assert browser in config.BROWSER_NAMES, f"'{browser}' in registry but not in config.BROWSER_NAMES"

    def test_browser_generic_term_included(self):
        from assistant import config
        assert "browser" in config.BROWSER_NAMES


# --- T12: WHATSAPP_* config values renamed to MESSAGING_* ---

class TestT12MessagingConfigRename:
    def test_messaging_notify_debounce_exists(self):
        from assistant import config
        assert hasattr(config, "MESSAGING_NOTIFY_DEBOUNCE")
        assert isinstance(config.MESSAGING_NOTIFY_DEBOUNCE, float)

    def test_messaging_suppress_window_exists(self):
        from assistant import config
        assert hasattr(config, "MESSAGING_SUPPRESS_WINDOW")
        assert isinstance(config.MESSAGING_SUPPRESS_WINDOW, float)

    def test_old_names_removed(self):
        from assistant import config
        assert not hasattr(config, "WHATSAPP_NOTIFY_DEBOUNCE")
        assert not hasattr(config, "WHATSAPP_SUPPRESS_WINDOW")


# --- T11: chrome_setup alias removed; browser_cdp_setup is the canonical name ---

class TestT11BrowserCdpSetupRename:
    def test_browser_cdp_setup_in_intents(self):
        from assistant.config import INTENTS
        assert "browser_cdp_setup" in INTENTS

    def test_chrome_setup_no_longer_an_intent(self):
        from assistant.config import INTENTS
        assert "chrome_setup" not in INTENTS

    def test_handler_importable(self):
        from assistant.actions.browser_cdp_setup import handle_browser_cdp_setup
        assert callable(handle_browser_cdp_setup)


# --- T15: No hardcoded brand stop-words in verifier ---

class TestT15VerifierStopwords:
    def test_music_stopwords_contains_app_names(self):
        from assistant.automation.vision.verifier import _MUSIC_STOPWORDS
        assert "spotify" in _MUSIC_STOPWORDS
        assert "chrome" in _MUSIC_STOPWORDS
        assert "whatsapp" in _MUSIC_STOPWORDS

    def test_music_stopwords_contains_generic_terms(self):
        from assistant.automation.vision.verifier import _MUSIC_STOPWORDS
        for word in ("play", "open", "music", "song", "from", "with"):
            assert word in _MUSIC_STOPWORDS

    def test_no_hardcoded_spotify_in_stopword_sets(self):
        import pathlib
        source = pathlib.Path("assistant/automation/vision/verifier.py").read_text()
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            if '"spotify"' in line and ("{" in line or "not in" in line):
                if "_MUSIC_STOPWORDS" not in line and "KNOWN_APPS" not in line:
                    raise AssertionError(f"Line {i}: possible hardcoded 'spotify' in stopword/filter set")


# --- T5: De-spotify verifier result strings ---

class TestT5VerifierGenericResults:
    def test_no_hardcoded_on_spotify_result(self):
        import pathlib
        source = pathlib.Path("assistant/automation/vision/verifier.py").read_text()
        assert 'on Spotify"' not in source

    def test_no_spotify_in_verification_prompt(self):
        import pathlib
        source = pathlib.Path("assistant/automation/vision/verifier.py").read_text()
        prompt_start = source.find("VERIFICATION_SYSTEM_PROMPT")
        prompt_section = source[prompt_start:source.find('"""', prompt_start + 30)]
        assert "Spotify" not in prompt_section, \
            "VERIFICATION_SYSTEM_PROMPT still references 'Spotify'"

    def test_music_keywords_use_registry(self):
        import pathlib
        source = pathlib.Path("assistant/automation/vision/verifier.py").read_text()
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            if "music_keywords" in line and "=" in line and "[" in line:
                if '"spotify"' in line.lower() or '"youtube music"' in line.lower():
                    raise AssertionError(f"Line {i}: brand name in music_keywords list")


# --- T13+T14: No brand names in prompt examples ---

class TestT13T14GenericPromptExamples:
    # ROUTER_EXAMPLES constant was deleted as part of CE-DYN (2026-05-30).
    # The slug-content guard now lives in tests/test_ce_dyn.py via
    # test_no_legacy_router_examples_string_in_prompt (which asserts the
    # legacy slugs do not leak into the dynamically-built router prompt).

    def test_intent_prompt_examples_no_whatsapp(self):
        from assistant.config import INTENT_SYSTEM_PROMPT
        examples_start = INTENT_SYSTEM_PROMPT.find("Few-shot examples")
        assert examples_start != -1, "INTENT_SYSTEM_PROMPT missing 'Few-shot examples' section"
        examples_section = INTENT_SYSTEM_PROMPT[examples_start:]
        assert "whatsapp" not in examples_section.lower(), \
            "Few-shot examples still reference 'whatsapp'"

    def test_intent_prompt_disambiguators_no_brands(self):
        from assistant.config import INTENT_SYSTEM_PROMPT
        disambig_start = INTENT_SYSTEM_PROMPT.find("Quick disambiguators")
        assert disambig_start != -1, "INTENT_SYSTEM_PROMPT missing 'Quick disambiguators' section"
        disambig_section = INTENT_SYSTEM_PROMPT[disambig_start:INTENT_SYSTEM_PROMPT.find("Param rules")]
        assert "spotify" not in disambig_section.lower(), \
            "Quick disambiguators still reference 'spotify'"
        assert "whatsapp" not in disambig_section.lower(), \
            "Quick disambiguators still reference 'whatsapp'"


# --- Cleanup sweep: remaining brand references in logic ---

class TestCleanupSweepNoBrandsInLogic:
    def test_intent_app_regex_uses_known_apps(self):
        from assistant.core.known_apps import KNOWN_APPS
        from assistant.intent import _APP_TARGETED_RE
        for app_name in KNOWN_APPS:
            assert _APP_TARGETED_RE.search(f"on {app_name}"), \
                f"_APP_TARGETED_RE doesn't match 'on {app_name}'"

    def test_intent_browser_guard_uses_known_apps(self):
        from assistant.core.known_apps import get_apps_by_category
        import re
        browsers = get_apps_by_category("browser")
        for browser in browsers:
            assert re.search(
                rf'\b(?:on|in|using)\s+(?:the\s+)?{re.escape(browser)}\b',
                f"on {browser}", re.IGNORECASE
            ), f"Guard 3 regex should match 'on {browser}'"

    def test_agent_app_keywords_uses_known_apps(self):
        from assistant.core.known_apps import KNOWN_APPS
        app_words = frozenset(
            word for name in KNOWN_APPS for word in name.split()
        ) | {"explorer", "premium", "word", "excel"}
        assert "spotify" in app_words
        assert "chrome" in app_words
        assert "whatsapp" in app_words

    def test_actions_messaging_keywords_from_registry(self):
        from assistant.core.known_apps import get_apps_by_category
        msg_apps = get_apps_by_category("messaging_default")
        assert "whatsapp" in msg_apps
        assert "telegram" in msg_apps
        assert "discord" in msg_apps

    def test_no_spotify_in_agent_system_prompt(self):
        import pathlib
        source = pathlib.Path("assistant/automation/vision/agent.py").read_text()
        prompt_start = source.find("AGENT_SYSTEM_PROMPT")
        prompt_end = source.find('"""', source.find('"""', prompt_start) + 3)
        prompt_section = source[prompt_start:prompt_end]
        assert "Spotify shows" not in prompt_section, \
            "AGENT_SYSTEM_PROMPT still has 'Spotify shows'"
        assert "Spotify IS already" not in prompt_section, \
            "AGENT_SYSTEM_PROMPT still has 'Spotify IS already'"

    def test_no_whatsapp_in_planner_prompt(self):
        import pathlib
        source = pathlib.Path("assistant/actions/planner/planner.py").read_text()
        prompt_start = source.find("_PLAN_SYSTEM_PROMPT")
        prompt_end = source.find('"""', source.find('"""', prompt_start) + 3)
        prompt_section = source[prompt_start:prompt_end]
        assert "WhatsApp" not in prompt_section, \
            "_PLAN_SYSTEM_PROMPT still references 'WhatsApp'"
        assert "whatsapp" not in prompt_section.lower(), \
            "_PLAN_SYSTEM_PROMPT still references 'whatsapp'"

    def test_no_spotify_in_planner_examples(self):
        import pathlib
        source = pathlib.Path("assistant/actions/planner/planner.py").read_text()
        examples_start = source.find("EXAMPLES:")
        if examples_start != -1:
            examples_section = source[examples_start:examples_start + 500]
            assert "spotify" not in examples_section.lower(), \
                "Planner EXAMPLES still references 'spotify'"

    def test_no_brands_in_reflection_prompt(self):
        import pathlib
        source = pathlib.Path("assistant/reflection.py").read_text()
        assert '"play on Spotify"' not in source, \
            "reflection.py still references 'play on Spotify'"
        assert "via WhatsApp" not in source, \
            "reflection.py still references 'via WhatsApp'"

    def test_pending_handlers_no_whatsapp_default(self):
        import pathlib
        source = pathlib.Path("assistant/actions/pending_handlers.py").read_text()
        assert '"whatsapp")' not in source, \
            "pending_handlers still has 'whatsapp' as default"

    def test_code_executor_prompts_no_brands_in_examples(self):
        import pathlib
        source = pathlib.Path("assistant/code_executor/prompts.py").read_text()
        assert "e.g. Spotify" not in source, \
            "prompts.py still references 'e.g. Spotify'"
        assert "For Gmail:" not in source, \
            "prompts.py still references 'For Gmail:'"
        assert "WhatsApp, Telegram, Discord) that" not in source, \
            "prompts.py still lists specific messaging services"
