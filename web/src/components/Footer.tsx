import React from 'react'
import { Github, ExternalLink } from 'lucide-react'
import { cn } from '@/lib/utils'
import { t } from '@/lib/i18n'
import { useSettingsStore } from '@/store/settings'

export function Footer() {
  const { language } = useSettingsStore()

  return (
    <footer className="bg-muted/50 border-t">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Brand */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 bg-primary rounded flex items-center justify-center">
                <span className="text-primary-foreground font-bold text-xs">DF</span>
              </div>
              <span className="text-lg font-bold">{t('app_title', language)}</span>
            </div>
            <p className="text-sm text-muted-foreground">
              {t('app_description', language)}
            </p>
          </div>

          {/* Links */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold">Resources</h4>
            <div className="space-y-2">
              <a
                href="https://github.com/deepfake-forensics/deepfake-forensics"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                <Github className="h-4 w-4" />
                <span>GitHub</span>
                <ExternalLink className="h-3 w-3" />
              </a>
              <a
                href="https://deepfake-forensics.readthedocs.io"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                <span>Documentation</span>
                <ExternalLink className="h-3 w-3" />
              </a>
            </div>
          </div>

          {/* Legal */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold">Legal</h4>
            <div className="space-y-2">
              <a
                href="#privacy"
                className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                {t('privacy', language)}
              </a>
              <a
                href="#ethics"
                className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                {t('ethics', language)}
              </a>
              <a
                href="#limitations"
                className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                {t('limitations', language)}
              </a>
            </div>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <p className="text-sm text-muted-foreground">
              © 2024 Deepfake Forensics. All rights reserved.
            </p>
            <p className="text-sm text-muted-foreground">
              Built with React, TypeScript, and Tailwind CSS
            </p>
          </div>
        </div>
      </div>
    </footer>
  )
}
