import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { Language } from '@/lib/i18n'

export type Theme = 'light' | 'dark' | 'system'

export type SettingsState = {
  theme: Theme
  language: Language
  apiBaseUrl: string
  mockMode: boolean
  setTheme: (theme: Theme) => void
  setLanguage: (language: Language) => void
  setApiBaseUrl: (url: string) => void
  setMockMode: (enabled: boolean) => void
  reset: () => void
}

const defaultSettings = {
  theme: 'system' as Theme,
  language: 'en' as Language,
  apiBaseUrl: import.meta.env.VITE_API_BASE || 'http://localhost:8000',
  mockMode: false,
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      ...defaultSettings,

      setTheme: (theme) =>
        set(() => ({
          theme,
        })),

      setLanguage: (language) =>
        set(() => ({
          language,
        })),

      setApiBaseUrl: (url) =>
        set(() => ({
          apiBaseUrl: url,
        })),

      setMockMode: (enabled) =>
        set(() => ({
          mockMode: enabled,
        })),

      reset: () =>
        set(() => ({
          ...defaultSettings,
        })),
    }),
    {
      name: 'settings-storage',
    }
  )
)
