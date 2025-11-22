import React from 'react'
import { Play, Pause, X, RotateCcw, CheckCircle, AlertCircle, Clock } from 'lucide-react'
import { cn, formatFileSize, getFileType } from '@/lib/utils'
import { t } from '@/lib/i18n'
import { useSettingsStore } from '@/store/settings'
import { useUploadsStore, type UploadItem as UploadItemType } from '@/store/uploads'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'

interface UploadItemProps {
  item: UploadItemType
  onStart: (id: string) => void
  onCancel: (id: string) => void
  onRetry: (id: string) => void
  onRemove: (id: string) => void
}

export function UploadItem({ 
  item, 
  onStart, 
  onCancel, 
  onRetry, 
  onRemove 
}: UploadItemProps) {
  const { language } = useSettingsStore()
  const { isProcessing } = useUploadsStore()

  const getStatusIcon = () => {
    switch (item.status) {
      case 'idle':
        return <Play className="h-4 w-4" />
      case 'uploading':
      case 'predicting':
        return <Clock className="h-4 w-4 animate-spin" />
      case 'done':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-destructive" />
      case 'canceled':
        return <X className="h-4 w-4 text-muted-foreground" />
      default:
        return null
    }
  }

  const getStatusText = () => {
    switch (item.status) {
      case 'idle':
        return t('idle', language)
      case 'uploading':
        return t('uploading', language)
      case 'predicting':
        return t('predicting', language)
      case 'done':
        return t('done', language)
      case 'error':
        return t('error', language)
      case 'canceled':
        return t('canceled', language)
      default:
        return ''
    }
  }

  const getStatusColor = () => {
    switch (item.status) {
      case 'idle':
        return 'text-muted-foreground'
      case 'uploading':
      case 'predicting':
        return 'text-primary'
      case 'done':
        return 'text-green-500'
      case 'error':
        return 'text-destructive'
      case 'canceled':
        return 'text-muted-foreground'
      default:
        return 'text-muted-foreground'
    }
  }

  const canStart = item.status === 'idle' && !isProcessing
  const canCancel = item.status === 'uploading' || item.status === 'predicting'
  const canRetry = item.status === 'error'
  const canRemove = item.status === 'done' || item.status === 'canceled' || item.status === 'error'

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {item.thumbnail && (
              <div className="w-12 h-12 rounded-lg overflow-hidden bg-muted">
                <img 
                  src={item.thumbnail} 
                  alt={item.file.name}
                  className="w-full h-full object-cover"
                />
              </div>
            )}
            <div className="flex-1 min-w-0">
              <h4 className="text-sm font-medium break-all">{item.file.name}</h4>
              <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                <span>{formatFileSize(item.file.size)}</span>
                <span>•</span>
                <span className="capitalize">{getFileType(item.file)}</span>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {getStatusIcon()}
            <span className={cn('text-sm font-medium', getStatusColor())}>
              {getStatusText()}
            </span>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        {(item.status === 'uploading' || item.status === 'predicting') && (
          <div className="space-y-2">
            <Progress value={item.progress} className="h-2" />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{Math.round(item.progress)}%</span>
              <span>{getStatusText()}</span>
            </div>
          </div>
        )}

        {item.error && (
          <div className="mt-2 p-2 bg-destructive/10 border border-destructive/20 rounded text-sm text-destructive">
            {item.error}
          </div>
        )}

        <div className="flex items-center justify-between mt-4">
          <div className="flex space-x-2">
            {canStart && (
              <Button
                size="sm"
                onClick={() => onStart(item.id)}
                disabled={isProcessing}
              >
                <Play className="h-3 w-3 mr-1" />
                {t('start', language)}
              </Button>
            )}

            {canCancel && (
              <Button
                size="sm"
                variant="outline"
                onClick={() => onCancel(item.id)}
              >
                <Pause className="h-3 w-3 mr-1" />
                {t('cancel', language)}
              </Button>
            )}

            {canRetry && (
              <Button
                size="sm"
                variant="outline"
                onClick={() => onRetry(item.id)}
              >
                <RotateCcw className="h-3 w-3 mr-1" />
                {t('retry', language)}
              </Button>
            )}
          </div>

          {canRemove && (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => onRemove(item.id)}
            >
              <X className="h-3 w-3" />
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
