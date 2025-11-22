import React from 'react'
import { Globe } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useSettingsStore } from '@/store/settings'
import { getAvailableLanguages, getLanguageName } from '@/lib/i18n'
import { Button } from '@/components/ui/button'

export function LanguageSwitch() {
  const { language, setLanguage } = useSettingsStore()
  const [isOpen, setIsOpen] = React.useState(false)

  const availableLanguages = getAvailableLanguages()

  const cycleLanguage = () => {
    const currentIndex = availableLanguages.indexOf(language)
    const nextIndex = (currentIndex + 1) % availableLanguages.length
    setLanguage(availableLanguages[nextIndex])
  }

  const getLanguageCode = () => {
    return language.toUpperCase()
  }

  const getLanguageName = () => {
    switch (language) {
      case 'en':
        return 'English'
      case 'tr':
        return 'Türkçe'
      default:
        return 'English'
    }
  }

  return (
    <div className="relative">
      <Button
        variant="ghost"
        size="sm"
        onClick={cycleLanguage}
        className="h-9 px-3"
        aria-label={`Current language: ${getLanguageName()}`}
        title={`Current language: ${getLanguageName()}`}
      >
        <Globe className="h-4 w-4 mr-2" />
        <span className="text-sm font-medium">{getLanguageCode()}</span>
      </Button>
    </div>
  )
}
