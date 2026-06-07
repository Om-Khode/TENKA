"""Tests for PACKAGE_ENV_ALIASES injection logic."""

from assistant.service_registry import PACKAGE_ENV_ALIASES


class TestAliasInjection:
    def _inject_aliases(self, env_vars: dict, requires: list[str]) -> dict:
        env = dict(env_vars)
        for pkg in requires:
            for alias_key, alias_src in PACKAGE_ENV_ALIASES.get(pkg, {}).items():
                if alias_key not in env:
                    if alias_src.startswith("$"):
                        src = alias_src[1:]
                        if src in env:
                            env[alias_key] = env[src]
                    else:
                        env[alias_key] = alias_src
        return env

    def test_spotipy_aliases_injected(self):
        env = self._inject_aliases(
            {
                "SPOTIFY_ACCESS_TOKEN": "tok_abc",
                "SPOTIFY_CLIENT_ID": "cid_123",
                "SPOTIFY_CLIENT_SECRET": "sec_456",
            },
            ["spotipy"],
        )
        assert env["SPOTIPY_CLIENT_ID"] == "cid_123"
        assert env["SPOTIPY_CLIENT_SECRET"] == "sec_456"
        assert env["SPOTIPY_ACCESS_TOKEN"] == "tok_abc"
        assert env["SPOTIPY_REDIRECT_URI"] == "http://localhost:8888/callback"

    def test_no_aliases_for_unknown_package(self):
        env = self._inject_aliases({"FOO": "bar"}, ["unknown-pkg"])
        assert env == {"FOO": "bar"}

    def test_existing_keys_not_overwritten(self):
        env = self._inject_aliases(
            {
                "SPOTIFY_CLIENT_ID": "cid_123",
                "SPOTIPY_CLIENT_ID": "already_set",
            },
            ["spotipy"],
        )
        assert env["SPOTIPY_CLIENT_ID"] == "already_set"

    def test_oauth_not_needed_after_injection(self):
        env = self._inject_aliases(
            {
                "SPOTIFY_ACCESS_TOKEN": "tok",
                "SPOTIFY_CLIENT_ID": "cid",
                "SPOTIFY_CLIENT_SECRET": "sec",
            },
            ["spotipy"],
        )
        needs_oauth = not all([
            env.get("SPOTIPY_CLIENT_ID"),
            env.get("SPOTIPY_CLIENT_SECRET"),
            env.get("SPOTIPY_ACCESS_TOKEN"),
        ])
        assert needs_oauth is False
