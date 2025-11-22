@echo off
title Fix Node.js and npm for NOLIE React Website
color 0A

echo.
echo ========================================
echo    Fix Node.js and npm for NOLIE
echo    React Website Installation
echo ========================================
echo.

echo 🔍 Checking current Node.js installation...

REM Check Node.js
node --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Node.js found
    node --version
) else (
    echo ❌ Node.js not found
)

REM Check npm
npm --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ npm found
    npm --version
) else (
    echo ❌ npm not found
)

echo.
echo 🔧 Attempting to fix PATH issues...

REM Add common Node.js paths
set "PATH=%PATH%;C:\Program Files\nodejs\;C:\Program Files (x86)\nodejs\;%APPDATA%\npm\;%USERPROFILE%\AppData\Roaming\npm\"

echo Testing Node.js after PATH fix...
node --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Node.js now working!
    node --version
) else (
    echo ❌ Node.js still not found
)

echo Testing npm after PATH fix...
npm --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ npm now working!
    npm --version
    echo.
    echo 🎉 Node.js and npm are working!
    echo.
    echo Now you can run the React website:
    echo   start.bat
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

echo 🔗 Opening Node.js download page...
start https://nodejs.org/

echo.
echo After installing Node.js, run this script again to verify the installation.
echo.
pause
