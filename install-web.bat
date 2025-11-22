@echo off
REM Simple Web Dependencies Installation Script

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                Installing Web Dependencies                   ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Check if web directory exists
if not exist "web" (
    echo ❌ Web directory not found!
    echo 💡 Make sure you're running this from the project root directory
    pause
    exit /b 1
)

REM Navigate to web directory
echo 📁 Navigating to web directory...
cd web

REM Check if package.json exists
if not exist "package.json" (
    echo ❌ package.json not found in web directory!
    pause
    exit /b 1
)

echo 📦 Installing all dependencies...
echo This may take a few minutes...
echo.

npm install

if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    echo 💡 Try running: npm cache clean --force
    pause
    exit /b 1
)

echo.
echo ✅ Dependencies installed successfully!
echo.
echo 🚀 Starting the development server...
echo 🌐 Web UI will be available at: http://localhost:5173
echo 💡 Press Ctrl+C to stop the server
echo.

npm run dev

echo.
echo 🛑 Development server stopped
pause
