import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type UploadStatus = 'idle' | 'uploading' | 'predicting' | 'done' | 'error' | 'canceled'

export type UploadItem = {
  id: string
  file: File
  thumbnail?: string
  status: UploadStatus
  progress: number
  error?: string
  result?: {
    score: number
    label: 'real' | 'fake'
    per_frame_scores?: number[]
    explanation_assets?: {
      heatmaps?: string[]
      key_frames?: string[]
    }
    meta?: Record<string, unknown>
  }
  createdAt: number
  completedAt?: number
}

export type UploadsState = {
  items: UploadItem[]
  maxConcurrent: number
  isProcessing: boolean
  addItem: (item: UploadItem) => void
  removeItem: (id: string) => void
  updateItem: (id: string, updates: Partial<UploadItem>) => void
  clearItems: () => void
  setMaxConcurrent: (max: number) => void
  setIsProcessing: (processing: boolean) => void
  getActiveItems: () => UploadItem[]
  getCompletedItems: () => UploadItem[]
  getErrorItems: () => UploadItem[]
}

export const useUploadsStore = create<UploadsState>()(
  persist(
    (set, get) => ({
      items: [],
      maxConcurrent: 2,
      isProcessing: false,

      addItem: (item) =>
        set((state) => ({
          items: [...state.items, item],
        })),

      removeItem: (id) =>
        set((state) => ({
          items: state.items.filter((item) => item.id !== id),
        })),

      updateItem: (id, updates) =>
        set((state) => ({
          items: state.items.map((item) =>
            item.id === id ? { ...item, ...updates } : item
          ),
        })),

      clearItems: () =>
        set(() => ({
          items: [],
        })),

      setMaxConcurrent: (max) =>
        set(() => ({
          maxConcurrent: max,
        })),

      setIsProcessing: (processing) =>
        set(() => ({
          isProcessing: processing,
        })),

      getActiveItems: () => {
        const state = get()
        return state.items.filter(
          (item) => item.status === 'uploading' || item.status === 'predicting'
        )
      },

      getCompletedItems: () => {
        const state = get()
        return state.items.filter((item) => item.status === 'done')
      },

      getErrorItems: () => {
        const state = get()
        return state.items.filter((item) => item.status === 'error')
      },
    }),
    {
      name: 'uploads-storage',
      partialize: (state) => ({
        items: state.items.filter((item) => item.status === 'done'),
        maxConcurrent: state.maxConcurrent,
      }),
    }
  )
)
