# Round 36 — GUI Ease-of-Use

## Overview
Make ForgeAI approachable for non-experts: light theme, empty states,
command palette, accessibility, activation wizard, model download manager.

## Features

### R36-1: Light Theme + Theme Switcher
- **File**: `forge_gui/theme.py` (ThemeManager, _LIGHT_TOKENS, _DARK_TOKENS)
- **What**: Added light theme alongside dark. ThemeManager class manages
  switching, persists choice to QSettings. Toggle via topbar button (◐)
  or Ctrl+Shift+T.
- **Impact**: Users can switch themes instantly. All Palette tokens
  update dynamically.

### R36-2: Empty-State Guidance + First-Run Onboarding
- **Files**: `forge_gui/widgets/empty_state.py`, `forge_gui/widgets/onboarding.py`
- **What**: Reusable EmptyState widget (icon + title + description + action
  button). OnboardingDialog for first-run (trust agent, model path, get
  started). Integrated into Models and Engine pages — shows empty state
  when no checkpoints instead of blank list.
- **Impact**: No more confusing empty lists. First-run users get guided.

### R36-3: Command Palette (Ctrl+K)
- **File**: `forge_gui/widgets/command_palette.py`
- **What**: VS Code-style command palette. Searches across all 14 pages
  + quick actions (toggle theme, font size, refresh). Keyboard navigation.
- **Impact**: Power users can navigate without mouse.

### R36-4: Accessibility (Font Scaling + High Contrast)
- **File**: `forge_gui/theme.py` (ThemeManager.FONT_SIZES)
- **What**: Font sizes: small (9pt), medium (10pt), large (12pt), xl (14pt).
  Cycle via topbar button (A) or Ctrl+Shift+F. Light theme provides
  high-contrast option for visually impaired.
- **Impact**: Accessibility compliance. Users can adjust text size.

### R36-5: One-Click Activation Preset Wizard
- **File**: `forge_gui/widgets/activation_wizard.py`
- **What**: 3-step wizard: select use-case (Chat/Coding/Agent/Long-Context/
  Max-Speed) → auto-selects best quantization + KV cache + decoding →
  apply. Integrated into Engine page.
- **Impact**: Surfaces 79+ ForgeEngine features in user-friendly way.

### R36-6: Model Download Manager
- **File**: `forge_gui/widgets/download_manager.py`
- **What**: Search HuggingFace models, download with progress bar, VRAM
  budget validation (warns if model won't fit 12GB after quantization).
  3 warning levels: green (≤8GB), yellow (>8GB), red (>11GB).
- **Impact**: No more manual model downloads. Prevents OOM downloads.

## Test Results
- R36-1/3/4 (theme + palette + font): 17 tests
- R36-2 (empty states + onboarding): 20 tests
- R36-5 (activation wizard): 26 tests
- R36-6 (download manager): 33 tests
- **Total R36: 96 tests, 0 failures**

## Files Created
- `forge_gui/widgets/empty_state.py`
- `forge_gui/widgets/onboarding.py`
- `forge_gui/widgets/command_palette.py`
- `forge_gui/widgets/activation_wizard.py`
- `forge_gui/widgets/download_manager.py`
- `tests/unit/test_r36_gui.py`
- `tests/unit/test_r36_empty_states.py`
- `tests/unit/test_r36_wizard.py`
- `tests/unit/test_r36_download.py`

## Files Modified
- `forge_gui/theme.py` (ThemeManager, light tokens, font scaling)
- `forge_gui/app.py` (theme toggle, font cycle, command palette, onboarding)
- `forge_gui/pages/models.py` (empty state integration)
- `forge_gui/pages/engine.py` (empty state + wizard button)
