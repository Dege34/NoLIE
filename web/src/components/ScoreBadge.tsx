import React from 'react'
import { cn } from '@/lib/utils'
import { t } from '@/lib/i18n'
import { useSettingsStore } from '@/store/settings'

interface ScoreBadgeProps {
  score: number
  className?: string
  size?: 'sm' | 'md' | 'lg'
}

export function ScoreBadge({ score, className, size = 'md' }: ScoreBadgeProps) {
  const { language } = useSettingsStore()

  const getScoreInfo = (score: number) => {
    if (score < 0.35) {
      return {
        label: t('result_real', language),
        color: 'text-green-500',
        bgColor: 'bg-green-500/10',
        ringColor: 'stroke-green-500',
        threshold: 0.35
      }
    } else if (score > 0.65) {
      return {
        label: t('result_deepfake', language),
        color: 'text-red-500',
        bgColor: 'bg-red-500/10',
        ringColor: 'stroke-red-500',
        threshold: 0.65
      }
    } else {
      return {
        label: t('result_uncertain', language),
        color: 'text-yellow-500',
        bgColor: 'bg-yellow-500/10',
        ringColor: 'stroke-yellow-500',
        threshold: 0.5
      }
    }
  }

  const scoreInfo = getScoreInfo(score)
  const percentage = Math.round(score * 100)
  const circumference = 2 * Math.PI * 40 // radius = 40
  const strokeDashoffset = circumference - (score * circumference)

  const sizeClasses = {
    sm: 'w-16 h-16 text-xs',
    md: 'w-24 h-24 text-sm',
    lg: 'w-32 h-32 text-base'
  }

  const radius = {
    sm: 24,
    md: 40,
    lg: 56
  }

  const currentRadius = radius[size]
  const currentCircumference = 2 * Math.PI * currentRadius
  const currentStrokeDashoffset = currentCircumference - (score * currentCircumference)

  return (
    <div className={cn('relative inline-flex items-center justify-center', sizeClasses[size], className)}>
      <svg
        className="absolute inset-0 -rotate-90"
        width={currentRadius * 2}
        height={currentRadius * 2}
      >
        {/* Background circle */}
        <circle
          cx={currentRadius}
          cy={currentRadius}
          r={currentRadius - 4}
          stroke="currentColor"
          strokeWidth="4"
          fill="none"
          className="text-muted-foreground/20"
        />
        {/* Progress circle */}
        <circle
          cx={currentRadius}
          cy={currentRadius}
          r={currentRadius - 4}
          stroke="currentColor"
          strokeWidth="4"
          fill="none"
          strokeLinecap="round"
          className={scoreInfo.ringColor}
          style={{
            strokeDasharray: currentCircumference,
            strokeDashoffset: currentStrokeDashoffset,
            transition: 'stroke-dashoffset 0.5s ease-in-out'
          }}
        />
      </svg>
      
      <div className="flex flex-col items-center justify-center text-center">
        <div className={cn('font-bold', scoreInfo.color)}>
          {percentage}%
        </div>
        <div className={cn('text-xs font-medium', scoreInfo.color)}>
          {scoreInfo.label}
        </div>
      </div>
    </div>
  )
}

export function scoreToLabel(score: number) {
  if (score < 0.35) return { label: 'Real', tone: 'success' }
  if (score > 0.65) return { label: 'Deepfake', tone: 'destructive' }
  return { label: 'Uncertain', tone: 'warning' }
}
