#!/bin/bash
# Deepfake Forensics - Unix/Linux/Mac Startup Script

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Banner
echo -e "${BOLD}${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Deepfake Forensics                        ║"
echo "║              Production-Grade Detection System               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed or not in PATH${NC}"
    echo -e "${YELLOW}💡 Please install Python 3.11+ and try again${NC}"
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed or not in PATH${NC}"
    echo -e "${YELLOW}💡 Please install Node.js 18+ and try again${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python and Node.js detected${NC}"

# Install Python dependencies if needed
echo -e "${YELLOW}📦 Checking Python dependencies...${NC}"
if ! python3 -c "import torch" &> /dev/null; then
    echo -e "${YELLOW}📦 Installing Python dependencies...${NC}"
    pip3 install -e .
fi

# Install web dependencies if needed
if [ ! -d "web/node_modules" ]; then
    echo -e "${YELLOW}📦 Installing web dependencies...${NC}"
    cd web
    npm install
    cd ..
fi

# Create environment files
if [ ! -f "web/.env.local" ]; then
    echo -e "${YELLOW}📝 Creating environment files...${NC}"
    echo "VITE_API_BASE=http://localhost:8000" > web/.env.local
fi

# Create data directories
mkdir -p data/raw data/interim data/processed checkpoints outputs logs

echo -e "${GREEN}✅ Environment setup complete${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Stopping services...${NC}"
    kill $API_PID 2>/dev/null || true
    kill $WEB_PID 2>/dev/null || true
    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start services
echo -e "\n${BLUE}🚀 Starting Deepfake Forensics services...${NC}"
echo -e "${BLUE}🌐 API will be available at: http://localhost:8000${NC}"
echo -e "${BLUE}🌐 Web UI will be available at: http://localhost:5173${NC}"
echo -e "${YELLOW}💡 Press Ctrl+C to stop all services${NC}"
echo

# Start API server in background
python3 -m deepfake_forensics.cli serve --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait a moment for API to start
sleep 3

# Start web server in background
cd web
npm run dev &
WEB_PID=$!
cd ..

echo -e "\n${GREEN}🎉 Services started successfully!${NC}"
echo -e "${BLUE}📖 Check the README.md for usage instructions${NC}"

# Wait for services
wait

