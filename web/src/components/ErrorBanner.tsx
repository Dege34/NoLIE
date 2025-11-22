import React from 'react'
import { AlertCircle, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

interface ErrorBannerProps {
  title: string
  message: string
  onDismiss?: () => void
  className?: string
  variant?: 'error' | 'warning' | 'info'
}

export function ErrorBanner({ 
  title, 
  message, 
  onDismiss, 
  className,
  variant = 'error'
}: ErrorBannerProps) {
  const getVariantStyles = () => {
    switch (variant) {
      case 'error':
        return 'bg-destructive/10 border-destructive/20 text-destructive'
      case 'warning':
        return 'bg-yellow-500/10 border-yellow-500/20 text-yellow-600'
      case 'info':
        return 'bg-blue-500/10 border-blue-500/20 text-blue-600'
      default:
        return 'bg-destructive/10 border-destructive/20 text-destructive'
    }
  }

  return (
    <div className={cn(
      'flex items-start space-x-3 p-4 border rounded-lg',
      getVariantStyles(),
      className
    )}>
      <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
      <div className="flex-1 min-w-0">
        <h4 className="text-sm font-medium">{title}</h4>
        <p className="text-sm mt-1">{message}</p>
      </div>
      {onDismiss && (
        <Button
          variant="ghost"
          size="sm"
          onClick={onDismiss}
          className="flex-shrink-0 h-6 w-6 p-0"
        >
          <X className="h-4 w-4" />
        </Button>
      )}
    </div>
  )
}
