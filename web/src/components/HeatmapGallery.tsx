import React, { useState } from 'react'
import { X, ZoomIn, Download } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { getAssetUrl } from '@/lib/api'

interface HeatmapGalleryProps {
  heatmaps: string[]
  title?: string
  className?: string
}

export function HeatmapGallery({ heatmaps, title, className }: HeatmapGalleryProps) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null)

  if (!heatmaps || heatmaps.length === 0) return null

  const openLightbox = (index: number) => {
    setSelectedIndex(index)
  }

  const closeLightbox = () => {
    setSelectedIndex(null)
  }

  const nextImage = () => {
    if (selectedIndex !== null) {
      setSelectedIndex((selectedIndex + 1) % heatmaps.length)
    }
  }

  const prevImage = () => {
    if (selectedIndex !== null) {
      setSelectedIndex(selectedIndex === 0 ? heatmaps.length - 1 : selectedIndex - 1)
    }
  }

  const downloadImage = (url: string, index: number) => {
    const link = document.createElement('a')
    link.href = getAssetUrl(url)
    link.download = `heatmap_${index + 1}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className={cn('space-y-4', className)}>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {heatmaps.map((heatmap, index) => (
          <div
            key={index}
            className="relative aspect-square rounded-lg overflow-hidden bg-muted cursor-pointer group hover:scale-105 transition-transform"
            onClick={() => openLightbox(index)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                openLightbox(index)
              }
            }}
            aria-label={`Heatmap ${index + 1}`}
          >
            <img
              src={getAssetUrl(heatmap)}
              alt={`Heatmap ${index + 1}`}
              className="w-full h-full object-cover"
              loading="lazy"
            />
            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
              <ZoomIn className="h-6 w-6 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
            </div>
            <div className="absolute top-2 right-2">
              <span className="bg-black/50 text-white text-xs px-2 py-1 rounded">
                {index + 1}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Lightbox Modal */}
      {selectedIndex !== null && (
        <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
          <div className="relative max-w-4xl max-h-full">
            {/* Close button */}
            <Button
              variant="ghost"
              size="icon"
              className="absolute top-4 right-4 z-10 bg-black/50 text-white hover:bg-black/70"
              onClick={closeLightbox}
            >
              <X className="h-6 w-6" />
            </Button>

            {/* Navigation buttons */}
            {heatmaps.length > 1 && (
              <>
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute left-4 top-1/2 -translate-y-1/2 z-10 bg-black/50 text-white hover:bg-black/70"
                  onClick={prevImage}
                >
                  <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute right-4 top-1/2 -translate-y-1/2 z-10 bg-black/50 text-white hover:bg-black/70"
                  onClick={nextImage}
                >
                  <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </Button>
              </>
            )}

            {/* Image */}
            <img
              src={getAssetUrl(heatmaps[selectedIndex])}
              alt={`Heatmap ${selectedIndex + 1}`}
              className="max-w-full max-h-full object-contain"
            />

            {/* Image info and controls */}
            <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between">
              <div className="bg-black/50 text-white px-3 py-2 rounded">
                <div className="text-sm font-medium">
                  {title && `${title} - `}Heatmap {selectedIndex + 1} of {heatmaps.length}
                </div>
              </div>
              
              <Button
                variant="ghost"
                size="sm"
                className="bg-black/50 text-white hover:bg-black/70"
                onClick={() => downloadImage(heatmaps[selectedIndex], selectedIndex)}
              >
                <Download className="h-4 w-4 mr-2" />
                Download
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Keyboard navigation */}
      {selectedIndex !== null && (
        <div
          className="fixed inset-0 z-40"
          onKeyDown={(e) => {
            switch (e.key) {
              case 'Escape':
                closeLightbox()
                break
              case 'ArrowLeft':
                if (heatmaps.length > 1) prevImage()
                break
              case 'ArrowRight':
                if (heatmaps.length > 1) nextImage()
                break
            }
          }}
          tabIndex={0}
        />
      )}
    </div>
  )
}
