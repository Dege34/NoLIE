import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { FileDropzone } from '@/components/FileDropzone'

// Mock the stores
jest.mock('@/store/settings', () => ({
  useSettingsStore: () => ({
    language: 'en'
  })
}))

describe('FileDropzone', () => {
  const mockOnFilesSelected = jest.fn()

  beforeEach(() => {
    mockOnFilesSelected.mockClear()
  })

  it('renders upload interface', () => {
    render(<FileDropzone onFilesSelected={mockOnFilesSelected} />)
    
    expect(screen.getByText('Upload Files')).toBeInTheDocument()
    expect(screen.getByText('Drop files here or click to select')).toBeInTheDocument()
    expect(screen.getByText('Select Files')).toBeInTheDocument()
  })

  it('shows supported file types', () => {
    render(<FileDropzone onFilesSelected={mockOnFilesSelected} />)
    
    expect(screen.getByText('Supported: JPG, PNG, MP4, MOV, AVI')).toBeInTheDocument()
  })

  it('shows file size limits', () => {
    render(<FileDropzone onFilesSelected={mockOnFilesSelected} />)
    
    expect(screen.getByText('Max size: 25MB images, 100MB videos')).toBeInTheDocument()
  })

  it('calls onFilesSelected when files are selected', () => {
    render(<FileDropzone onFilesSelected={mockOnFilesSelected} />)
    
    const fileInput = screen.getByLabelText('file-input')
    const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' })
    
    fireEvent.change(fileInput, { target: { files: [file] } })
    
    expect(mockOnFilesSelected).toHaveBeenCalledWith([file])
  })

  it('handles drag and drop events', () => {
    render(<FileDropzone onFilesSelected={mockOnFilesSelected} />)
    
    const dropzone = screen.getByText('Drop files here or click to select').closest('div')
    
    const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' })
    
    fireEvent.dragEnter(dropzone!, { dataTransfer: { files: [file] } })
    fireEvent.dragOver(dropzone!, { dataTransfer: { files: [file] } })
    fireEvent.drop(dropzone!, { dataTransfer: { files: [file] } })
    
    expect(mockOnFilesSelected).toHaveBeenCalledWith([file])
  })
})
