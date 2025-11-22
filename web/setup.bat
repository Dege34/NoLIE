@echo off
REM Deepfake Forensics Web UI Setup Script

echo 🚀 Setting up Deepfake Forensics Web UI...

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js 18+ first.
    pause
    exit /b 1
)

echo ✅ Node.js detected
node --version

REM Install dependencies
echo 📦 Installing dependencies...
npm install
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Create environment file if it doesn't exist
if not exist .env.local (
    echo 📝 Creating .env.local file...
    echo VITE_API_BASE=http://localhost:8000 > .env.local
    echo ✅ Created .env.local with default API URL
)

REM Run type checking
echo 🔍 Running type checking...
npm run type-check
if %errorlevel% neq 0 (
    echo ⚠️  Type checking failed, but continuing...
)

REM Run linting
echo 🧹 Running linting...
npm run lint
if %errorlevel% neq 0 (
    echo ⚠️  Linting failed, but continuing...
)

REM Run tests
echo 🧪 Running tests...
npm run test
if %errorlevel% neq 0 (
    echo ⚠️  Tests failed, but continuing...
)

echo ✅ Setup complete!
echo.
echo To start the development server:
echo   npm run dev
echo.
echo To build for production:
echo   npm run build
echo.
echo To run tests:
echo   npm run test
echo.
echo Web UI will be available at: http://localhost:5173
echo Make sure the API server is running at: http://localhost:8000
pause
