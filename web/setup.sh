#!/bin/bash

# Deepfake Forensics Web UI Setup Script

echo "🚀 Setting up Deepfake Forensics Web UI..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version 18+ is required. Current version: $(node -v)"
    exit 1
fi

echo "✅ Node.js $(node -v) detected"

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Create environment file if it doesn't exist
if [ ! -f .env.local ]; then
    echo "📝 Creating .env.local file..."
    echo "VITE_API_BASE=http://localhost:8000" > .env.local
    echo "✅ Created .env.local with default API URL"
fi

# Run type checking
echo "🔍 Running type checking..."
npm run type-check

# Run linting
echo "🧹 Running linting..."
npm run lint

# Run tests
echo "🧪 Running tests..."
npm run test

echo "✅ Setup complete!"
echo ""
echo "To start the development server:"
echo "  npm run dev"
echo ""
echo "To build for production:"
echo "  npm run build"
echo ""
echo "To run tests:"
echo "  npm run test"
echo ""
echo "Web UI will be available at: http://localhost:5173"
echo "Make sure the API server is running at: http://localhost:8000"
