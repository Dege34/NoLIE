import { describe, it, expect, vi, beforeEach } from 'vitest'
import { predict, healthcheck, APIError } from '@/lib/api'

// Mock fetch
global.fetch = vi.fn()

describe('API', () => {
  beforeEach(() => {
    vi.resetAllMocks()
  })

  describe('healthcheck', () => {
    it('returns true when API is healthy', async () => {
      ;(global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200
      })

      const result = await healthcheck()
      expect(result).toBe(true)
    })

    it('returns false when API is unhealthy', async () => {
      ;(global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 500
      })

      const result = await healthcheck()
      expect(result).toBe(false)
    })

    it('returns false when network error occurs', async () => {
      ;(global.fetch as any).mockRejectedValueOnce(new Error('Network error'))

      const result = await healthcheck()
      expect(result).toBe(false)
    })
  })

  describe('predict', () => {
    it('successfully predicts with valid response', async () => {
      const mockResponse = {
        score: 0.8,
        label: 'fake',
        per_frame_scores: [0.7, 0.8, 0.9],
        explanation_assets: {
          heatmaps: ['heatmap1.png', 'heatmap2.png']
        },
        meta: {
          model: 'test-model',
          version: '1.0.0'
        }
      }

      ;(global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      })

      const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' })
      const result = await predict(file)

      expect(result).toEqual(mockResponse)
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/predict'),
        expect.objectContaining({
          method: 'POST',
          headers: {
            'X-Client': 'deepfake-webui'
          }
        })
      )
    })

    it('throws APIError when response is not ok', async () => {
      ;(global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        text: () => Promise.resolve('Invalid file type')
      })

      const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' })
      
      await expect(predict(file)).rejects.toThrow(APIError)
    })

    it('throws APIError when network error occurs', async () => {
      ;(global.fetch as any).mockRejectedValueOnce(new Error('Network error'))

      const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' })
      
      await expect(predict(file)).rejects.toThrow(APIError)
    })

    it('handles timeout correctly', async () => {
      ;(global.fetch as any).mockImplementationOnce(() => 
        new Promise((_, reject) => {
          setTimeout(() => reject(new Error('Timeout')), 100)
        })
      )

      const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' })
      
      await expect(predict(file)).rejects.toThrow(APIError)
    })
  })
})
