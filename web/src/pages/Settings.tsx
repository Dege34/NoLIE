import React, { useState } from 'react'
import { ArrowLeft, Save, RotateCcw, Globe, Monitor, Sun, Moon } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { t, getAvailableLanguages, getLanguageName } from '@/lib/i18n'
import { useSettingsStore } from '@/store/settings'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useToast } from '@/hooks/use-toast'

export function Settings() {
  const navigate = useNavigate()
  const { toast } = useToast()
  const { language } = useSettingsStore()
  
  const {
    theme,
    language: currentLanguage,
    apiBaseUrl,
    mockMode,
    setTheme,
    setLanguage,
    setApiBaseUrl,
    setMockMode,
    reset
  } = useSettingsStore()

  const [localApiUrl, setLocalApiUrl] = useState(apiBaseUrl)
  const [localMockMode, setLocalMockMode] = useState(mockMode)
  const [localTheme, setLocalTheme] = useState(theme)
  const [localLanguage, setLocalLanguage] = useState(currentLanguage)

  const handleSave = () => {
    setApiBaseUrl(localApiUrl)
    setMockMode(localMockMode)
    setTheme(localTheme)
    setLanguage(localLanguage)
    
    toast({
      title: 'Settings saved',
      description: 'Your preferences have been updated'
    })
  }

  const handleReset = () => {
    reset()
    setLocalApiUrl(import.meta.env.VITE_API_BASE || 'http://localhost:8000')
    setLocalMockMode(false)
    setLocalTheme('system')
    setLocalLanguage('en')
    
    toast({
      title: 'Settings reset',
      description: 'All settings have been restored to defaults'
    })
  }

  const themes = [
    { value: 'light', label: 'Light', icon: Sun },
    { value: 'dark', label: 'Dark', icon: Moon },
    { value: 'system', label: 'System', icon: Monitor }
  ]

  const languages = getAvailableLanguages()

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex items-center space-x-4 mb-8">
          <Button
            variant="ghost"
            onClick={() => navigate(-1)}
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
          <div>
            <h1 className="text-3xl font-bold">{t('settings', language)}</h1>
            <p className="text-muted-foreground">
              Customize your experience and preferences
            </p>
          </div>
        </div>

        <div className="space-y-6">
          {/* Appearance */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Monitor className="h-5 w-5" />
                <span>Appearance</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Theme */}
              <div className="space-y-3">
                <label className="text-sm font-medium">{t('theme', language)}</label>
                <div className="grid grid-cols-3 gap-3">
                  {themes.map((themeOption) => {
                    const Icon = themeOption.icon
                    return (
                      <button
                        key={themeOption.value}
                        onClick={() => setLocalTheme(themeOption.value as any)}
                        className={cn(
                          'flex items-center space-x-2 p-3 border rounded-lg text-left transition-colors',
                          localTheme === themeOption.value
                            ? 'border-primary bg-primary/5'
                            : 'border-border hover:bg-accent'
                        )}
                      >
                        <Icon className="h-4 w-4" />
                        <span className="text-sm font-medium">{themeOption.label}</span>
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Language */}
              <div className="space-y-3">
                <label className="text-sm font-medium">{t('language', language)}</label>
                <div className="grid grid-cols-2 gap-3">
                  {languages.map((lang) => (
                    <button
                      key={lang}
                      onClick={() => setLocalLanguage(lang)}
                      className={cn(
                        'flex items-center space-x-2 p-3 border rounded-lg text-left transition-colors',
                        localLanguage === lang
                          ? 'border-primary bg-primary/5'
                          : 'border-border hover:bg-accent'
                      )}
                    >
                      <Globe className="h-4 w-4" />
                      <span className="text-sm font-medium">{getLanguageName(lang)}</span>
                    </button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* API Settings */}
          <Card>
            <CardHeader>
              <CardTitle>API Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* API Base URL */}
              <div className="space-y-3">
                <label htmlFor="api-url" className="text-sm font-medium">
                  {t('api_url', language)}
                </label>
                <input
                  id="api-url"
                  type="url"
                  value={localApiUrl}
                  onChange={(e) => setLocalApiUrl(e.target.value)}
                  className="w-full px-3 py-2 border border-input rounded-md bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                  placeholder="http://localhost:8000"
                />
                <p className="text-xs text-muted-foreground">
                  Base URL for the deepfake detection API
                </p>
              </div>

              {/* Mock Mode */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium">{t('mock_mode', language)}</label>
                    <p className="text-xs text-muted-foreground">
                      Use simulated results for demonstration purposes
                    </p>
                  </div>
                  <button
                    onClick={() => setLocalMockMode(!localMockMode)}
                    className={cn(
                      'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
                      localMockMode ? 'bg-primary' : 'bg-muted'
                    )}
                  >
                    <span
                      className={cn(
                        'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
                        localMockMode ? 'translate-x-6' : 'translate-x-1'
                      )}
                    />
                  </button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Upload Settings */}
          <Card>
            <CardHeader>
              <CardTitle>Upload Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-3">
                <label className="text-sm font-medium">File Size Limits</label>
                <div className="grid grid-cols-2 gap-4 text-sm text-muted-foreground">
                  <div>
                    <strong>Images:</strong> 25MB maximum
                  </div>
                  <div>
                    <strong>Videos:</strong> 100MB maximum
                  </div>
                </div>
              </div>
              
              <div className="space-y-3">
                <label className="text-sm font-medium">Supported Formats</label>
                <div className="text-sm text-muted-foreground">
                  <strong>Images:</strong> JPG, JPEG, PNG<br />
                  <strong>Videos:</strong> MP4, MOV, AVI
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Privacy & Security */}
          <Card>
            <CardHeader>
              <CardTitle>Privacy & Security</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h4 className="font-medium">Data Handling</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Files are processed locally when possible</li>
                  <li>• No personal data is collected or stored</li>
                  <li>• Results are not shared with third parties</li>
                  <li>• All processing is done securely</li>
                </ul>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-medium">Security Features</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Client-side file validation</li>
                  <li>• Secure API communication</li>
                  <li>• No persistent storage of uploaded files</li>
                  <li>• Regular security updates</li>
                </ul>
              </div>
            </CardContent>
          </Card>

          {/* Actions */}
          <Card>
            <CardHeader>
              <CardTitle>Actions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex space-x-4">
                <Button onClick={handleSave}>
                  <Save className="h-4 w-4 mr-2" />
                  Save Settings
                </Button>
                <Button variant="outline" onClick={handleReset}>
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Reset to Defaults
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
