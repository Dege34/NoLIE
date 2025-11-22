import React, { useState } from 'react'
import { Download, Eye, ChevronDown, ChevronUp, FileText, Image, Video } from 'lucide-react'
import { cn, formatFileSize, getFileType, downloadJSON } from '@/lib/utils'
import { t } from '@/lib/i18n'
import { useSettingsStore } from '@/store/settings'
import { type UploadItem } from '@/store/uploads'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScoreBadge } from '@/components/ScoreBadge'
import { VideoTimeline } from '@/components/VideoTimeline'
import { HeatmapGallery } from '@/components/HeatmapGallery'

interface ResultCardProps {
  item: UploadItem
  onDownloadReport: (item: UploadItem) => void
  onSaveFrames: (item: UploadItem) => void
}

export function ResultCard({ item, onDownloadReport, onSaveFrames }: ResultCardProps) {
  const { language } = useSettingsStore()
  const [showDetails, setShowDetails] = useState(false)

  if (!item.result) return null

  const { result } = item
  const fileType = getFileType(item.file)
  const confidence = Math.round(result.score * 100)

  const getFileIcon = () => {
    switch (fileType) {
      case 'image':
        return <Image className="h-4 w-4" />
      case 'video':
        return <Video className="h-4 w-4" />
      default:
        return <FileText className="h-4 w-4" />
    }
  }

  const getLabelColor = () => {
    if (result.score < 0.35) return 'text-green-500 bg-green-500/10'
    if (result.score > 0.65) return 'text-red-500 bg-red-500/10'
    return 'text-yellow-500 bg-yellow-500/10'
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            {item.thumbnail && (
              <div className="w-16 h-16 rounded-lg overflow-hidden bg-muted">
                <img 
                  src={item.thumbnail} 
                  alt={item.file.name}
                  className="w-full h-full object-cover"
                />
              </div>
            )}
            <div className="flex-1 min-w-0">
              <CardTitle className="text-lg break-all">{item.file.name}</CardTitle>
              <div className="flex items-center space-x-2 text-sm text-muted-foreground mt-1">
                {getFileIcon()}
                <span>{formatFileSize(item.file.size)}</span>
                <span>•</span>
                <span className="capitalize">{fileType}</span>
                {item.completedAt && (
                  <>
                    <span>•</span>
                    <span>
                      {new Date(item.completedAt).toLocaleString()}
                    </span>
                  </>
                )}
              </div>
            </div>
          </div>
          <ScoreBadge score={result.score} size="lg" />
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Score and Label */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="text-center">
              <div className="text-2xl font-bold">{confidence}%</div>
              <div className="text-sm text-muted-foreground">{t('confidence', language)}</div>
            </div>
            <div className="text-center">
              <div className={cn(
                'px-3 py-1 rounded-full text-sm font-medium',
                getLabelColor()
              )}>
                {result.label === 'real' ? t('result_real', language) : t('result_deepfake', language)}
              </div>
              <div className="text-sm text-muted-foreground mt-1">{t('label', language)}</div>
            </div>
          </div>
        </div>

        {/* Video Timeline */}
        {result.per_frame_scores && result.per_frame_scores.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium">{t('timeline', language)}</h4>
            <VideoTimeline scores={result.per_frame_scores} />
          </div>
        )}

        {/* Heatmaps */}
        {result.explanation_assets?.heatmaps && result.explanation_assets.heatmaps.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium">{t('heatmaps', language)}</h4>
            <HeatmapGallery 
              heatmaps={result.explanation_assets.heatmaps}
              title={item.file.name}
            />
          </div>
        )}

        {/* Key Frames */}
        {result.explanation_assets?.key_frames && result.explanation_assets.key_frames.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium">{t('key_frames', language)}</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {result.explanation_assets.key_frames.map((frame, index) => (
                <div key={index} className="aspect-video rounded-lg overflow-hidden bg-muted">
                  <img 
                    src={frame} 
                    alt={`Key frame ${index + 1}`}
                    className="w-full h-full object-cover"
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Details Accordion */}
        <div className="space-y-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowDetails(!showDetails)}
            className="w-full justify-between"
          >
            <span>{t('details', language)}</span>
            {showDetails ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </Button>
          
          {showDetails && (
            <div className="p-4 bg-muted rounded-lg space-y-2">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium">{t('file_name', language)}:</span>
                  <span className="ml-2 text-muted-foreground">{item.file.name}</span>
                </div>
                <div>
                  <span className="font-medium">{t('file_size', language)}:</span>
                  <span className="ml-2 text-muted-foreground">{formatFileSize(item.file.size)}</span>
                </div>
                <div>
                  <span className="font-medium">{t('file_type', language)}:</span>
                  <span className="ml-2 text-muted-foreground capitalize">{fileType}</span>
                </div>
                <div>
                  <span className="font-medium">{t('score', language)}:</span>
                  <span className="ml-2 text-muted-foreground">{result.score.toFixed(4)}</span>
                </div>
                {result.meta && (
                  <>
                    <div>
                      <span className="font-medium">{t('model_info', language)}:</span>
                      <span className="ml-2 text-muted-foreground">
                        {result.meta.model || 'Unknown'}
                      </span>
                    </div>
                    <div>
                      <span className="font-medium">{t('version', language)}:</span>
                      <span className="ml-2 text-muted-foreground">
                        {result.meta.version || 'Unknown'}
                      </span>
                    </div>
                  </>
                )}
              </div>
              
              {result.per_frame_scores && (
                <div className="mt-4">
                  <h5 className="text-sm font-medium mb-2">{t('per_frame_scores', language)}</h5>
                  <div className="text-xs text-muted-foreground font-mono bg-background p-2 rounded">
                    {JSON.stringify(result.per_frame_scores, null, 2)}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-2 pt-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onDownloadReport(item)}
            className="flex-1"
          >
            <Download className="h-4 w-4 mr-2" />
            {t('download_report', language)}
          </Button>
          
          {fileType === 'video' && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onSaveFrames(item)}
              className="flex-1"
            >
              <Image className="h-4 w-4 mr-2" />
              {t('save_frames', language)}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
