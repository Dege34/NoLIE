@echo off
title Fix Node.js and npm for NOLIE
color 0A

echo.
echo ========================================
echo    Fixing Node.js and npm for NOLIE
echo ========================================
echo.

echo Checking Node.js installation...
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo Node.js not found in PATH. Let's fix this...
    
    echo.
    echo Common Node.js installation paths:
    echo 1. C:\Program Files\nodejs\
    echo 2. C:\Program Files (x86)\nodejs\
    echo 3. %APPDATA%\npm\
    echo.
    
    echo Adding common Node.js paths to current session...
    set "PATH=%PATH%;C:\Program Files\nodejs\;C:\Program Files (x86)\nodejs\;%APPDATA%\npm\"
    
    echo Testing Node.js...
    node --version
    if %errorlevel% neq 0 (
        echo Node.js still not found. Please install Node.js from https://nodejs.org/
        echo Download the LTS version and make sure to check "Add to PATH" during installation.
        pause
        exit /b 1
    )
) else (
    echo Node.js found!
    node --version
)

echo.
echo Testing npm...
npm --version
if %errorlevel% neq 0 (
    echo npm not found. Let's try to fix this...
    
    echo Adding npm to PATH...
    set "PATH=%PATH%;C:\Program Files\nodejs\;C:\Program Files (x86)\nodejs\;%APPDATA%\npm\"
    
    echo Testing npm again...
    npm --version
    if %errorlevel% neq 0 (
        echo npm still not found. Trying to reinstall npm...
        node -e "console.log('Node.js is working')"
        if %errorlevel% equ 0 (
            echo Installing npm globally...
            node -e "const npm = require('npm'); npm.load(function(err) { if (err) console.error(err); else console.log('npm loaded'); });"
        )
    )
)

echo.
echo Final test...
node --version
npm --version

if %errorlevel% equ 0 (
    echo.
    echo ✅ Node.js and npm are working!
    echo.
    echo Now you can run the full NOLIE React website:
    echo   start_nolie.bat
    echo.
) else (
    echo.
    echo ❌ Node.js/npm still not working properly.
    echo.
    echo Please:
    echo 1. Download Node.js from https://nodejs.org/
    echo 2. Install the LTS version
    echo 3. Make sure to check "Add to PATH" during installation
    echo 4. Restart your computer
    echo 5. Run this script again
    echo.
)

pause
