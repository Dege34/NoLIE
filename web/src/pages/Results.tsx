import React, { useCallback, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Download, FileText, Image, AlertCircle } from 'lucide-react'
import { cn, downloadJSON, createZip } from '@/lib/utils'
import { t } from '@/lib/i18n'
import { useSettingsStore } from '@/store/settings'
import { useUploadsStore } from '@/store/uploads'
import { ResultCard } from '@/components/ResultCard'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useToast } from '@/hooks/use-toast'

export function Results() {
  const navigate = useNavigate()
  const { language } = useSettingsStore()
  const { getCompletedItems, clearItems } = useUploadsStore()
  const { toast } = useToast()
  
  const [isExporting, setIsExporting] = useState(false)
  
  const completedItems = getCompletedItems()

  const handleDownloadReport = useCallback(async (item: any) => {
    if (!item.result) return

    try {
      const report = {
        file: {
          name: item.file.name,
          size: item.file.size,
          type: item.file.type,
          lastModified: item.file.lastModified
        },
        result: item.result,
        metadata: {
          createdAt: item.createdAt,
          completedAt: item.completedAt,
          processingTime: item.completedAt ? item.completedAt - item.createdAt : null
        }
      }

      downloadJSON(report, `deepfake_report_${item.file.name}.json`)
      
      toast({
        title: t('success_uploaded', language),
        description: 'Report downloaded successfully'
      })
    } catch (err) {
      console.error('Failed to download report:', err)
      toast({
        title: t('error', language),
        description: 'Failed to download report',
        variant: 'destructive'
      })
    }
  }, [language, toast])

  const handleSaveFrames = useCallback(async (item: any) => {
    if (!item.result?.explanation_assets?.key_frames) return

    try {
      setIsExporting(true)
      
      // Create a ZIP file with the key frames
      const files = item.result.explanation_assets.key_frames.map((frame: string, index: number) => ({
        name: `frame_${index + 1}.jpg`,
        data: new Blob([frame], { type: 'image/jpeg' })
      }))

      const zipBlob = await createZip(files)
      
      // Download the ZIP file
      const url = URL.createObjectURL(zipBlob)
      const a = document.createElement('a')
      a.href = url
      a.download = `frames_${item.file.name}.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      
      toast({
        title: t('success_uploaded', language),
        description: 'Frames saved successfully'
      })
    } catch (err) {
      console.error('Failed to save frames:', err)
      toast({
        title: t('error', language),
        description: 'Failed to save frames',
        variant: 'destructive'
      })
    } finally {
      setIsExporting(false)
    }
  }, [language, toast])

  const handleDownloadAllReports = useCallback(async () => {
    try {
      setIsExporting(true)
      
      const allReports = completedItems.map(item => ({
        file: {
          name: item.file.name,
          size: item.file.size,
          type: item.file.type,
          lastModified: item.file.lastModified
        },
        result: item.result,
        metadata: {
          createdAt: item.createdAt,
          completedAt: item.completedAt,
          processingTime: item.completedAt ? item.completedAt - item.createdAt : null
        }
      }))

      const summary = {
        totalFiles: completedItems.length,
        realCount: completedItems.filter(item => item.result?.label === 'real').length,
        fakeCount: completedItems.filter(item => item.result?.label === 'fake').length,
        averageScore: completedItems.reduce((sum, item) => sum + (item.result?.score || 0), 0) / completedItems.length,
        generatedAt: new Date().toISOString(),
        reports: allReports
      }

      downloadJSON(summary, 'deepfake_analysis_summary.json')
      
      toast({
        title: t('success_uploaded', language),
        description: 'All reports downloaded successfully'
      })
    } catch (err) {
      console.error('Failed to download all reports:', err)
      toast({
        title: t('error', language),
        description: 'Failed to download all reports',
        variant: 'destructive'
      })
    } finally {
      setIsExporting(false)
    }
  }, [completedItems, language, toast])

  const handleClearResults = useCallback(() => {
    clearItems()
    navigate('/')
  }, [clearItems, navigate])

  if (completedItems.length === 0) {
    return (
      <div className="min-h-screen bg-background">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center py-12">
            <AlertCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium text-muted-foreground mb-2">
              {t('info_no_results', language)}
            </h3>
            <p className="text-muted-foreground mb-6">
              Upload some files to get started with deepfake detection
            </p>
            <Button onClick={() => navigate('/')}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Go to Upload
            </Button>
          </div>
        </div>
      </div>
    )
  }

  const realCount = completedItems.filter(item => item.result?.label === 'real').length
  const fakeCount = completedItems.filter(item => item.result?.label === 'fake').length
  const averageScore = completedItems.reduce((sum, item) => sum + (item.result?.score || 0), 0) / completedItems.length

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <Button
              variant="ghost"
              onClick={() => navigate('/')}
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Upload
            </Button>
            <div>
              <h1 className="text-3xl font-bold">Analysis Results</h1>
              <p className="text-muted-foreground">
                {completedItems.length} files analyzed
              </p>
            </div>
          </div>
          
          <div className="flex space-x-2">
            <Button
              variant="outline"
              onClick={handleDownloadAllReports}
              disabled={isExporting}
            >
              <Download className="h-4 w-4 mr-2" />
              Download All Reports
            </Button>
            <Button
              variant="outline"
              onClick={handleClearResults}
            >
              Clear Results
            </Button>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Total Files</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{completedItems.length}</div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Real</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-500">{realCount}</div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Deepfake</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-500">{fakeCount}</div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Avg Score</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {averageScore.toFixed(3)}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Results Grid */}
        <div className="space-y-6">
          {completedItems.map((item) => (
            <ResultCard
              key={item.id}
              item={item}
              onDownloadReport={handleDownloadReport}
              onSaveFrames={handleSaveFrames}
            />
          ))}
        </div>

        {/* Bottom CTA */}
        <div className="text-center mt-12">
          <Button
            onClick={() => navigate('/')}
            size="lg"
            className="px-8"
          >
            <FileText className="h-5 w-5 mr-2" />
            {t('analyze_more', language)}
          </Button>
        </div>
      </div>
    </div>
  )
}
