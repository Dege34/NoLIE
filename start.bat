@echo off
REM NOLIE ULTIMATE DEEPFAKE DETECTOR - Windows Startup Script

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                        NOLIE                              ║
echo ║              ULTIMATE DEEPFAKE DETECTOR                    ║
echo ║                                                              ║
echo ║  🧠 12 AI Models • 🎯 Ultra-High Accuracy                  ║
echo ║  🔍 Advanced Analysis • 📊 Professional Results             ║
echo ║                                                              ║
echo ║  Created by: Dogan Ege BULTE                        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo 💡 Please install Python 3.11+ and try again
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed or not in PATH
    echo 💡 Please install Node.js 18+ and try again
    echo.
    echo 🔧 Attempting to fix Node.js PATH...
    set "PATH=%PATH%;C:\Program Files\nodejs\;C:\Program Files (x86)\nodejs\;%APPDATA%\npm\"
    node --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Node.js still not found
        echo 💡 Please install Node.js from https://nodejs.org/
        pause
        exit /b 1
    ) else (
        echo ✅ Node.js found after PATH fix
    )
)

REM Check if npm is available
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm is not found
    echo 🔧 Attempting to fix npm PATH...
    set "PATH=%PATH%;C:\Program Files\nodejs\;C:\Program Files (x86)\nodejs\;%APPDATA%\npm\"
    npm --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ npm still not found
        echo 💡 Please reinstall Node.js with npm included
        pause
        exit /b 1
    ) else (
        echo ✅ npm found after PATH fix
    )
)

echo ✅ Python and Node.js detected

REM Install Python dependencies if needed
echo 📦 Checking Python dependencies...
pip show fastapi >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing Python dependencies...
    pip install fastapi uvicorn requests
    if %errorlevel% neq 0 (
        echo ❌ Failed to install Python dependencies
        pause
        exit /b 1
    )
)

REM Install web dependencies if needed
if not exist "web\node_modules" (
    echo 📦 Installing web dependencies...
    cd web
    npm install
    if %errorlevel% neq 0 (
        echo ❌ Failed to install web dependencies
        pause
        exit /b 1
    )
    cd ..
)

REM Create environment files
if not exist "web\.env.local" (
    echo 📝 Creating environment files...
    echo VITE_API_BASE=http://localhost:8000 > web\.env.local
)

REM Create data directories
if not exist "data\raw" mkdir data\raw
if not exist "data\interim" mkdir data\interim
if not exist "data\processed" mkdir data\processed
if not exist "checkpoints" mkdir checkpoints
if not exist "outputs" mkdir outputs
if not exist "logs" mkdir logs

echo ✅ Environment setup complete

REM Start services
echo.
echo 🚀 Starting Deepfake Forensics services...
echo.
echo 🌐 API will be available at: http://localhost:8000
echo 🌐 Web UI will be available at: http://localhost:5173
echo.
echo 💡 Press Ctrl+C to stop all services
echo.

REM Start ULTIMATE DETECTOR API server in background
start "NOLIE ULTIMATE DETECTOR API" cmd /c "python ultimate_detector.py"

REM Wait a moment for API to start
timeout /t 3 /nobreak >nul

REM Start React web server
cd web
start "NOLIE React Web" cmd /c "npm run dev"
cd ..

echo.
echo 🎉 Services started successfully!
echo.
echo 📖 Check the README.md for usage instructions
echo 💡 Press any key to stop all services
pause >nul

REM Stop services
echo.
echo 🛑 Stopping services...
taskkill /f /im "python.exe" >nul 2>&1
taskkill /f /im "node.exe" >nul 2>&1
echo ✅ All services stopped
