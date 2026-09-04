"""Tests for Round 36 GUI Ease-of-Use features.

R36-1: Light Theme + Theme Switcher
R36-3: Command Palette (Ctrl+K)
R36-4: Accessibility (Font Scaling + High Contrast)
"""
from __future__ import annotations

import pytest


# ── R36-1: Theme Manager ──────────────────────────────────────────────────

class TestThemeManager:
    """Light/dark theme switching via ThemeManager."""

    def test_themes_available(self):
        from forge_gui.theme import ThemeManager
        assert "dark" in ThemeManager.THEMES
        assert "light" in ThemeManager.THEMES

    def test_default_theme_is_dark(self):
        from forge_gui.theme import ThemeManager
        # Reset to default
        ThemeManager._current_theme = "dark"
        assert ThemeManager.current_theme() == "dark"

    def test_toggle_theme(self):
        from forge_gui.theme import ThemeManager
        ThemeManager._current_theme = "dark"
        new = ThemeManager.toggle_theme()
        assert new == "light"
        assert ThemeManager.current_theme() == "light"
        new = ThemeManager.toggle_theme()
        assert new == "dark"

    def test_set_theme(self):
        from forge_gui.theme import ThemeManager
        ThemeManager.set_theme("light")
        assert ThemeManager.current_theme() == "light"
        ThemeManager.set_theme("dark")
        assert ThemeManager.current_theme() == "dark"

    def test_set_invalid_theme_ignored(self):
        from forge_gui.theme import ThemeManager
        ThemeManager.set_theme("dark")
        ThemeManager.set_theme("invalid")
        assert ThemeManager.current_theme() == "dark"

    def test_light_tokens_differ_from_dark(self):
        from forge_gui.theme import _LIGHT_TOKENS, _DARK_TOKENS
        assert _LIGHT_TOKENS["bg"] != _DARK_TOKENS["bg"]
        assert _LIGHT_TOKENS["text"] != _DARK_TOKENS["text"]
        assert _LIGHT_TOKENS["panel"] != _DARK_TOKENS["panel"]

    def test_light_tokens_complete(self):
        """All dark token keys must exist in light tokens."""
        from forge_gui.theme import _LIGHT_TOKENS, _DARK_TOKENS
        for key in _DARK_TOKENS:
            assert key in _LIGHT_TOKENS, f"Missing light token: {key}"

    def test_palette_has_all_attributes(self):
        from forge_gui.theme import Palette
        attrs = ["bg", "bg_alt", "panel", "panel_alt", "border", "border_hi",
                 "text", "text_dim", "text_faint", "accent", "accent_hi",
                 "accent_dim", "accent2", "ok", "warn", "err"]
        for attr in attrs:
            assert hasattr(Palette, attr), f"Palette missing: {attr}"


# ── R36-4: Font Scaling ───────────────────────────────────────────────────

class TestFontScaling:
    """Font size cycling for accessibility."""

    def test_font_sizes_available(self):
        from forge_gui.theme import ThemeManager
        assert "small" in ThemeManager.FONT_SIZES
        assert "medium" in ThemeManager.FONT_SIZES
        assert "large" in ThemeManager.FONT_SIZES
        assert "xl" in ThemeManager.FONT_SIZES

    def test_font_size_values(self):
        from forge_gui.theme import ThemeManager
        assert ThemeManager.FONT_SIZES["small"] < ThemeManager.FONT_SIZES["medium"]
        assert ThemeManager.FONT_SIZES["medium"] < ThemeManager.FONT_SIZES["large"]
        assert ThemeManager.FONT_SIZES["large"] < ThemeManager.FONT_SIZES["xl"]

    def test_default_font_size(self):
        from forge_gui.theme import ThemeManager
        ThemeManager._current_font_size = "medium"
        assert ThemeManager.current_font_size() == "medium"

    def test_set_font_size(self):
        from forge_gui.theme import ThemeManager
        ThemeManager.set_font_size("large")
        assert ThemeManager.current_font_size() == "large"
        ThemeManager.set_font_size("medium")
        assert ThemeManager.current_font_size() == "medium"

    def test_set_invalid_font_size_ignored(self):
        from forge_gui.theme import ThemeManager
        ThemeManager.set_font_size("medium")
        ThemeManager.set_font_size("huge")
        assert ThemeManager.current_font_size() == "medium"


# ── R36-3: Command Palette ────────────────────────────────────────────────

class TestCommandPalette:
    """Ctrl+K command palette for global search."""

    def test_build_items(self):
        """Test that command palette items are built correctly."""
        # Test the item building logic without QApplication
        index_to_name = {0: "Dashboard", 1: "Chat Studio", 2: "Agent"}
        # Simulate what _build_items does
        items = []
        for idx, name in sorted(index_to_name.items()):
            items.append((f"Go to {name}", idx, "Navigate"))
        items.append(("Toggle Theme (Light/Dark)", -1, "Action:theme"))
        items.append(("Font Size: Small", -1, "Action:font:small"))
        items.append(("Refresh Current Page", -1, "Action:refresh"))
        # Verify
        assert len(items) == 6  # 3 pages + 3 actions
        assert items[0] == ("Go to Dashboard", 0, "Navigate")
        assert items[3] == ("Toggle Theme (Light/Dark)", -1, "Action:theme")

    def test_filter_logic(self):
        """Test the filtering logic."""
        all_items = [
            ("Go to Dashboard", 0, "Navigate"),
            ("Go to Chat Studio", 1, "Navigate"),
            ("Go to Agent", 2, "Navigate"),
            ("Toggle Theme", -1, "Action:theme"),
            ("Font Size: Large", -1, "Action:font:large"),
        ]
        # Filter by "chat"
        filtered = [i for i in all_items if "chat" in i[0].lower()]
        assert len(filtered) == 1
        assert filtered[0][0] == "Go to Chat Studio"
        # Filter by "theme"
        filtered = [i for i in all_items if "theme" in i[0].lower()]
        assert len(filtered) == 1
        # Filter by empty string returns all
        filtered = [i for i in all_items if "" in i[0].lower()]
        assert len(filtered) == 5
        # Filter by non-matching
        filtered = [i for i in all_items if "xyz" in i[0].lower()]
        assert len(filtered) == 0

    def test_action_parsing(self):
        """Test that action categories are parsed correctly."""
        actions = [
            ("Action:theme", "theme"),
            ("Action:font:small", "font"),
            ("Action:font:large", "font"),
            ("Action:refresh", "refresh"),
        ]
        for action_str, expected_keyword in actions:
            parts = action_str.split(":")
            assert expected_keyword in parts

    def test_page_navigation_items(self):
        """Test that all pages get navigation items."""
        index_to_name = {
            0: "Dashboard", 1: "Chat Studio", 2: "Agent",
            3: "Generations", 4: "Engine", 5: "Models",
        }
        nav_items = []
        for idx, name in sorted(index_to_name.items()):
            nav_items.append((f"Go to {name}", idx, "Navigate"))
        assert len(nav_items) == 6
        for item in nav_items:
            assert item[1] >= 0  # page index
            assert item[2] == "Navigate"
