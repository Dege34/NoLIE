import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, File, AlertCircle } from 'lucide-react'
import { cn, formatFileSize, isValidFileType, getFileType } from '@/lib/utils'
import { t } from '@/lib/i18n'
import { useSettingsStore } from '@/store/settings'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'

interface FileDropzoneProps {
  onFilesSelected: (files: File[]) => void
  maxSize?: number
  className?: string
}

export function FileDropzone({ 
  onFilesSelected, 
  maxSize = 100 * 1024 * 1024, // 100MB default
  className 
}: FileDropzoneProps) {
  const { language } = useSettingsStore()
  const [dragActive, setDragActive] = useState(false)
  const [dragReject, setDragReject] = useState(false)

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    setDragActive(false)
    setDragReject(false)
    
    if (rejectedFiles.length > 0) {
      console.warn('Some files were rejected:', rejectedFiles)
    }
    
    if (acceptedFiles.length > 0) {
      onFilesSelected(acceptedFiles)
    }
  }, [onFilesSelected])

  const onDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(true)
    setDragReject(false)
  }, [])

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    setDragReject(false)
  }, [])

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    
    const files = Array.from(e.dataTransfer.items).map(item => item.getAsFile())
    const hasInvalidFiles = files.some(file => file && !isValidFileType(file))
    
    if (hasInvalidFiles) {
      setDragReject(true)
    } else {
      setDragReject(false)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png'],
      'video/*': ['.mp4', '.mov', '.avi']
    },
    maxSize,
    multiple: true,
    onDragEnter,
    onDragLeave,
    onDragOver,
  })

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    const validFiles = files.filter(file => isValidFileType(file) && file.size <= maxSize)
    onFilesSelected(validFiles)
  }

  return (
    <Card className={cn('w-full', className)}>
      <CardContent className="p-0">
        <div
          {...getRootProps()}
          className={cn(
            'file-drop-zone',
            isDragActive && 'drag-over',
            dragReject && 'drag-reject'
          )}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className="rounded-full bg-primary/10 p-4">
              {dragReject ? (
                <AlertCircle className="h-8 w-8 text-destructive" />
              ) : (
                <Upload className="h-8 w-8 text-primary" />
              )}
            </div>
            
            <div className="text-center">
              <h3 className="text-lg font-semibold">
                {dragReject 
                  ? t('error_invalid_type', language)
                  : t('upload_title', language)
                }
              </h3>
              <p className="text-sm text-muted-foreground mt-1">
                {dragReject 
                  ? t('file_types', language)
                  : t('drop_here', language)
                }
              </p>
            </div>

            <div className="text-xs text-muted-foreground space-y-1">
              <p>{t('file_types', language)}</p>
              <p>{t('max_size', language)}</p>
            </div>

            <Button 
              type="button" 
              variant="outline"
              onClick={(e) => {
                e.stopPropagation()
                document.getElementById('file-input')?.click()
              }}
            >
              <File className="h-4 w-4 mr-2" />
              {t('select_files', language)}
            </Button>

            <input
              id="file-input"
              type="file"
              multiple
              accept="image/*,video/*"
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
