import React, { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, AlertCircle, CheckCircle } from 'lucide-react'
import { cn, formatFileSize, isValidFileType, getFileType, createImageThumbnail, createVideoThumbnail, generateFileId } from '@/lib/utils'
import { t } from '@/lib/i18n'
import { useSettingsStore } from '@/store/settings'
import { useUploadsStore, type UploadItem } from '@/store/uploads'
import { predict, healthcheck, APIError } from '@/lib/api'
import { FileDropzone } from '@/components/FileDropzone'
import { UploadItem as UploadItemComponent } from '@/components/UploadItem'
import { ErrorBanner } from '@/components/ErrorBanner'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useToast } from '@/hooks/use-toast'

export function Home() {
  const navigate = useNavigate()
  const { language, mockMode } = useSettingsStore()
  const { 
    items, 
    addItem, 
    updateItem, 
    removeItem, 
    getActiveItems, 
    getCompletedItems,
    maxConcurrent,
    isProcessing,
    setIsProcessing
  } = useUploadsStore()
  const { toast } = useToast()
  
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const healthy = await healthcheck()
        setIsHealthy(healthy)
        if (!healthy && !mockMode) {
          setError(t('error_network', language))
        }
      } catch (err) {
        setIsHealthy(false)
        if (!mockMode) {
          setError(t('error_network', language))
        }
      }
    }
    
    checkHealth()
  }, [language, mockMode])

  const handleFilesSelected = useCallback(async (files: File[]) => {
    if (files.length === 0) return

    // Validate files
    const validFiles = files.filter(file => {
      if (!isValidFileType(file)) {
        toast({
          title: t('error_invalid_type', language),
          description: `${file.name} - ${t('file_types', language)}`,
          variant: 'destructive'
        })
        return false
      }
      return true
    })

    if (validFiles.length === 0) return

    // Create upload items
    const uploadItems: UploadItem[] = []
    
    for (const file of validFiles) {
      const id = generateFileId()
      let thumbnail: string | undefined

      try {
        if (getFileType(file) === 'image') {
          thumbnail = await createImageThumbnail(file)
        } else if (getFileType(file) === 'video') {
          thumbnail = await createVideoThumbnail(file)
        }
      } catch (err) {
        console.warn('Failed to create thumbnail:', err)
      }

      const item: UploadItem = {
        id,
        file,
        thumbnail,
        status: 'idle',
        progress: 0,
        createdAt: Date.now()
      }

      uploadItems.push(item)
      addItem(item)
    }

    toast({
      title: t('success_uploaded', language),
      description: `${validFiles.length} files added to queue`
    })
  }, [addItem, language, toast])

  const handleStart = useCallback(async (id: string) => {
    const item = items.find(i => i.id === id)
    if (!item) return

    if (isProcessing) return

    setIsProcessing(true)
    updateItem(id, { status: 'uploading', progress: 0 })

    try {
      // Simulate upload progress
      for (let progress = 0; progress <= 100; progress += 10) {
        updateItem(id, { progress })
        await new Promise(resolve => setTimeout(resolve, 100))
      }

      updateItem(id, { status: 'predicting', progress: 0 })

      let result
      if (mockMode) {
        // Mock prediction
        await new Promise(resolve => setTimeout(resolve, 2000))
        result = {
          score: Math.random(),
          label: Math.random() > 0.5 ? 'real' : 'fake' as const,
          per_frame_scores: getFileType(item.file) === 'video' ? 
            Array.from({ length: 30 }, () => Math.random()) : undefined,
          explanation_assets: {
            heatmaps: Math.random() > 0.5 ? 
              Array.from({ length: 3 }, (_, i) => `mock_heatmap_${i + 1}.png`) : undefined,
            key_frames: getFileType(item.file) === 'video' && Math.random() > 0.5 ? 
              Array.from({ length: 5 }, (_, i) => `mock_keyframe_${i + 1}.jpg`) : undefined
          },
          meta: {
            model: 'Mock Model',
            version: '1.0.0'
          }
        }
      } else {
        // Real prediction
        result = await predict(item.file)
      }

      updateItem(id, { 
        status: 'done', 
        progress: 100, 
        result,
        completedAt: Date.now()
      })

      toast({
        title: t('success_predicted', language),
        description: `${item.file.name} analysis complete`
      })

      // Navigate to results if this is the first completed item
      const completedItems = getCompletedItems()
      if (completedItems.length === 1) {
        navigate('/results')
      }

    } catch (err) {
      console.error('Prediction failed:', err)
      
      let errorMessage = t('error_prediction_failed', language)
      if (err instanceof APIError) {
        if (err.status === 408) {
          errorMessage = t('error_timeout', language)
        } else if (err.status >= 500) {
          errorMessage = t('error_server', language)
        } else if (err.status >= 400) {
          errorMessage = t('error_upload_failed', language)
        }
      }

      updateItem(id, { 
        status: 'error', 
        error: errorMessage 
      })

      toast({
        title: t('error', language),
        description: errorMessage,
        variant: 'destructive'
      })
    } finally {
      setIsProcessing(false)
    }
  }, [items, isProcessing, mockMode, updateItem, getCompletedItems, language, toast, navigate])

  const handleCancel = useCallback((id: string) => {
    updateItem(id, { status: 'canceled' })
  }, [updateItem])

  const handleRetry = useCallback((id: string) => {
    updateItem(id, { status: 'idle', error: undefined, progress: 0 })
  }, [updateItem])

  const handleRemove = useCallback((id: string) => {
    removeItem(id)
  }, [removeItem])

  const activeItems = getActiveItems()
  const completedItems = getCompletedItems()
  const errorItems = items.filter(item => item.status === 'error')

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold tracking-tight text-gray-900 dark:text-white mb-4">
          NoLIE          
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Production-grade deepfake detection with explainability and robustness
          </p>
        </div>

        {/* Health Status */}
        {isHealthy === false && !mockMode && (
          <div className="mb-6">
            <ErrorBanner
              title="API Unavailable"
              message={t('error_network', language)}
              variant="warning"
            />
          </div>
        )}

        {/* Mock Mode Banner */}
        {mockMode && (
          <div className="mb-6">
            <ErrorBanner
              title={t('info_mock_mode', language)}
              message="Using simulated results for demonstration purposes"
              variant="info"
            />
          </div>
        )}

        {/* Upload Section */}
        <div className="mb-8">
          <FileDropzone
            onFilesSelected={handleFilesSelected}
            maxSize={100 * 1024 * 1024} // 100MB
          />
        </div>

        {/* Upload Queue */}
        {items.length > 0 && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-semibold">Upload Queue</h2>
              <div className="text-sm text-muted-foreground">
                {items.length} files • {completedItems.length} complete
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {items.map((item) => (
                <UploadItemComponent
                  key={item.id}
                  item={item}
                  onStart={handleStart}
                  onCancel={handleCancel}
                  onRetry={handleRetry}
                  onRemove={handleRemove}
                />
              ))}
            </div>

            {/* Results CTA */}
            {completedItems.length > 0 && (
              <div className="text-center">
                <Button
                  onClick={() => navigate('/results')}
                  size="lg"
                  className="px-8"
                >
                  <CheckCircle className="h-5 w-5 mr-2" />
                  View Results ({completedItems.length})
                </Button>
              </div>
            )}
          </div>
        )}

        {/* Empty State */}
        {items.length === 0 && (
          <div className="text-center py-12">
            <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium text-muted-foreground mb-2">
              {t('info_upload_files', language)}
            </h3>
            <p className="text-muted-foreground">
              Drag and drop files or click to select
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
