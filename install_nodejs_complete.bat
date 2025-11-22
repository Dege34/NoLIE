@echo off
title Complete Node.js Installation for NOLIE
color 0A

echo.
echo ========================================
echo    Complete Node.js Installation
echo    For NOLIE Deepfake Detection System
echo    Created by Dogan Ege BULTE
echo ========================================
echo.

echo 🔍 Checking current Node.js installation...

REM Check if Node.js is already working
where node >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Node.js found in PATH
    node --version
) else (
    echo ❌ Node.js not found in PATH
)

where npm >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ npm found in PATH
    npm --version
) else (
    echo ❌ npm not found in PATH
)

echo.
echo 🔧 Attempting to fix PATH issues...

REM Add common Node.js paths to current session
set "PATH=%PATH%;C:\Program Files\nodejs\;C:\Program Files (x86)\nodejs\;%APPDATA%\npm\;%USERPROFILE%\AppData\Roaming\npm\"

echo Testing Node.js after PATH fix...
where node >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Node.js now working!
    node --version
) else (
    echo ❌ Node.js still not found
)

echo Testing npm after PATH fix...
where npm >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ npm now working!
    npm --version
    echo.
    echo 🎉 Node.js and npm are working!
    echo.
    echo Now you can run the full NOLIE React system:
    echo   start_nolie_full.bat
    echo.
    pause
    exit /b 0
) else (
    echo ❌ npm still not found
)

echo.
echo 📝 Node.js Installation Required
echo.
echo Please follow these steps:
echo.
echo 1. Download Node.js from: https://nodejs.org/
echo 2. Choose the LTS version (recommended)
echo 3. Run the installer and follow these steps:
echo    - Accept the license agreement
echo    - Choose installation directory (default is fine)
echo    - IMPORTANT: Check 'Add to PATH' option
echo    - Complete the installation
echo 4. Restart your computer
echo 5. Run this script again
echo.
echo Alternative installation methods:
echo.
echo Method 1 - Chocolatey (if installed):
echo   choco install nodejs
echo.
echo Method 2 - Winget (Windows 10/11):
echo   winget install OpenJS.NodeJS
echo.
echo Method 3 - Manual download:
echo   Go to https://nodejs.org/ and download the Windows installer
echo.

echo 🔗 Opening Node.js download page...
start https://nodejs.org/

echo.
echo After installing Node.js, run this script again to verify the installation.
echo.
pause
