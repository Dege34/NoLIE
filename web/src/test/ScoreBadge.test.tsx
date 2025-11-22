import React from 'react'
import { render, screen } from '@testing-library/react'
import { ScoreBadge, scoreToLabel } from '@/components/ScoreBadge'

// Mock the stores
jest.mock('@/store/settings', () => ({
  useSettingsStore: () => ({
    language: 'en'
  })
}))

describe('ScoreBadge', () => {
  it('renders with correct score', () => {
    render(<ScoreBadge score={0.8} />)
    
    expect(screen.getByText('80%')).toBeInTheDocument()
  })

  it('shows correct label for high score (deepfake)', () => {
    render(<ScoreBadge score={0.8} />)
    
    expect(screen.getByText('Deepfake')).toBeInTheDocument()
  })

  it('shows correct label for low score (real)', () => {
    render(<ScoreBadge score={0.2} />)
    
    expect(screen.getByText('Real')).toBeInTheDocument()
  })

  it('shows correct label for medium score (uncertain)', () => {
    render(<ScoreBadge score={0.5} />)
    
    expect(screen.getByText('Uncertain')).toBeInTheDocument()
  })

  it('applies custom className', () => {
    const { container } = render(<ScoreBadge score={0.5} className="custom-class" />)
    
    expect(container.firstChild).toHaveClass('custom-class')
  })

  it('renders different sizes', () => {
    const { container: small } = render(<ScoreBadge score={0.5} size="sm" />)
    const { container: medium } = render(<ScoreBadge score={0.5} size="md" />)
    const { container: large } = render(<ScoreBadge score={0.5} size="lg" />)
    
    expect(small.firstChild).toHaveClass('w-16', 'h-16')
    expect(medium.firstChild).toHaveClass('w-24', 'h-24')
    expect(large.firstChild).toHaveClass('w-32', 'h-32')
  })
})

describe('scoreToLabel', () => {
  it('returns correct label for real score', () => {
    expect(scoreToLabel(0.2)).toEqual({ label: 'Real', tone: 'success' })
    expect(scoreToLabel(0.3)).toEqual({ label: 'Real', tone: 'success' })
  })

  it('returns correct label for deepfake score', () => {
    expect(scoreToLabel(0.7)).toEqual({ label: 'Deepfake', tone: 'destructive' })
    expect(scoreToLabel(0.9)).toEqual({ label: 'Deepfake', tone: 'destructive' })
  })

  it('returns correct label for uncertain score', () => {
    expect(scoreToLabel(0.4)).toEqual({ label: 'Uncertain', tone: 'warning' })
    expect(scoreToLabel(0.6)).toEqual({ label: 'Uncertain', tone: 'warning' })
  })
})
