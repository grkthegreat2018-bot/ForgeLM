// Small UI state — sidebar collapse, command palette.

import { create } from 'zustand'

interface UiState {
  collapsed: boolean
  paletteOpen: boolean
  toggleCollapsed: () => void
  setPalette: (v: boolean) => void
}

export const useUi = create<UiState>((set) => ({
  collapsed: false,
  paletteOpen: false,
  toggleCollapsed: () => set((s) => ({ collapsed: !s.collapsed })),
  setPalette: (v) => set({ paletteOpen: v }),
}))
