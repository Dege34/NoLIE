import React, { useState, useRef } from 'react'
import { cn } from '@/lib/utils'

interface VideoTimelineProps {
  scores: number[]
  className?: string
}

export function VideoTimeline({ scores, className }: VideoTimelineProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)
  const [focusedIndex, setFocusedIndex] = useState<number | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  if (!scores || scores.length === 0) return null

  const width = 400
  const height = 60
  const padding = 10
  const chartWidth = width - padding * 2
  const chartHeight = height - padding * 2

  // Normalize scores to 0-1 range
  const minScore = Math.min(...scores)
  const maxScore = Math.max(...scores)
  const normalizedScores = scores.map(score => 
    (score - minScore) / (maxScore - minScore || 1)
  )

  // Create path data for the polyline
  const points = normalizedScores.map((score, index) => {
    const x = (index / (scores.length - 1)) * chartWidth + padding
    const y = height - padding - (score * chartHeight)
    return `${x},${y}`
  }).join(' ')

  // Create area path for fill
  const areaPoints = [
    `${padding},${height - padding}`,
    ...normalizedScores.map((score, index) => {
      const x = (index / (scores.length - 1)) * chartWidth + padding
      const y = height - padding - (score * chartHeight)
      return `${x},${y}`
    }),
    `${width - padding},${height - padding}`
  ].join(' ')

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return
    
    const rect = svgRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const index = Math.round(((x - padding) / chartWidth) * (scores.length - 1))
    
    if (index >= 0 && index < scores.length) {
      setHoveredIndex(index)
    } else {
      setHoveredIndex(null)
    }
  }

  const handleMouseLeave = () => {
    setHoveredIndex(null)
  }

  const handleKeyDown = (e: React.KeyboardEvent<SVGSVGElement>) => {
    if (focusedIndex === null) {
      setFocusedIndex(0)
      return
    }

    switch (e.key) {
      case 'ArrowLeft':
        e.preventDefault()
        setFocusedIndex(Math.max(0, focusedIndex - 1))
        break
      case 'ArrowRight':
        e.preventDefault()
        setFocusedIndex(Math.min(scores.length - 1, focusedIndex + 1))
        break
      case 'Escape':
        setFocusedIndex(null)
        break
    }
  }

  const getScoreColor = (score: number) => {
    if (score < 0.35) return '#10b981' // green
    if (score > 0.65) return '#ef4444' // red
    return '#f59e0b' // yellow
  }

  const getScoreLabel = (score: number) => {
    if (score < 0.35) return 'Real'
    if (score > 0.65) return 'Deepfake'
    return 'Uncertain'
  }

  return (
    <div className={cn('w-full', className)}>
      <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
        <span>Frame 0</span>
        <span>Frame {scores.length - 1}</span>
      </div>
      
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="w-full h-16 cursor-crosshair"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onKeyDown={handleKeyDown}
        tabIndex={0}
        role="img"
        aria-label={`Video timeline with ${scores.length} frames`}
      >
        {/* Grid lines */}
        <defs>
          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="currentColor" strokeWidth="0.5" opacity="0.1"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
        
        {/* Area fill */}
        <path
          d={`M ${areaPoints} Z`}
          fill="currentColor"
          fillOpacity="0.1"
        />
        
        {/* Line */}
        <polyline
          points={points}
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        
        {/* Data points */}
        {normalizedScores.map((score, index) => {
          const x = (index / (scores.length - 1)) * chartWidth + padding
          const y = height - padding - (score * chartHeight)
          const isHovered = hoveredIndex === index
          const isFocused = focusedIndex === index
          const isActive = isHovered || isFocused
          
          return (
            <circle
              key={index}
              cx={x}
              cy={y}
              r={isActive ? 4 : 2}
              fill={getScoreColor(scores[index])}
              stroke="white"
              strokeWidth={isActive ? 2 : 1}
              className={cn(
                'transition-all duration-200',
                isActive && 'drop-shadow-md'
              )}
            />
          )
        })}
        
        {/* Hover indicator */}
        {hoveredIndex !== null && (
          <line
            x1={(hoveredIndex / (scores.length - 1)) * chartWidth + padding}
            y1={padding}
            x2={(hoveredIndex / (scores.length - 1)) * chartWidth + padding}
            y2={height - padding}
            stroke="currentColor"
            strokeWidth="1"
            strokeDasharray="2,2"
            opacity="0.5"
          />
        )}
      </svg>
      
      {/* Tooltip */}
      {hoveredIndex !== null && (
        <div className="mt-2 p-2 bg-background border rounded-lg shadow-lg">
          <div className="text-sm">
            <div className="font-medium">Frame {hoveredIndex}</div>
            <div className="text-muted-foreground">
              Score: {scores[hoveredIndex].toFixed(4)} ({getScoreLabel(scores[hoveredIndex])})
            </div>
          </div>
        </div>
      )}
      
      {/* Focus indicator */}
      {focusedIndex !== null && (
        <div className="mt-2 p-2 bg-primary/10 border border-primary rounded-lg">
          <div className="text-sm">
            <div className="font-medium">Frame {focusedIndex} (Focused)</div>
            <div className="text-muted-foreground">
              Score: {scores[focusedIndex].toFixed(4)} ({getScoreLabel(scores[focusedIndex])})
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              Use arrow keys to navigate, Escape to exit
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
