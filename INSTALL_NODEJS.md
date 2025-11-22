# Installing Node.js for NOLIE Full Experience

## 🚀 Quick Solution (Current)

**You can use NOLIE right now without Node.js!**

1. **Run the simple version:**
   ```bash
   start_nolie_simple.bat
   ```
   or
   ```bash
   python start_nolie_simple.py
   ```

2. **This gives you:**
   - ✅ Enhanced AI with multi-model ensemble
   - ✅ Beautiful web interface
   - ✅ All deepfake detection features
   - ✅ Detailed analysis reports

## 🌟 For Full React Experience

To get the complete React-based NOLIE interface, install Node.js:

### Method 1: Official Website (Recommended)

1. **Go to:** https://nodejs.org/
2. **Download:** The LTS version (Long Term Support)
3. **Install:** Run the installer and follow the instructions
4. **Important:** Make sure to check "Add to PATH" during installation
5. **Restart:** Your computer after installation

### Method 2: Using Chocolatey (Windows)

```powershell
# Install Chocolatey first (if not installed)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Node.js
choco install nodejs
```

### Method 3: Using Winget (Windows 10/11)

```powershell
winget install OpenJS.NodeJS
```

### Method 4: Using Scoop (Windows)

```powershell
# Install Scoop first (if not installed)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex

# Install Node.js
scoop install nodejs
```

## ✅ Verify Installation

After installation, open a new Command Prompt and run:

```bash
node --version
npm --version
```

You should see version numbers like:
```
v18.17.0
9.6.7
```

## 🎯 After Node.js Installation

Once Node.js is installed, you can use the full NOLIE experience:

```bash
start_nolie.bat
```

This will give you:
- 🌐 Full React-based web interface
- 🎨 Advanced UI components
- 📱 Better mobile experience
- 🔧 Advanced settings and configuration

## 🆘 Troubleshooting

### Node.js not found after installation

1. **Restart your computer** (this is important!)
2. **Check PATH:** Make sure Node.js is in your system PATH
3. **Try different terminal:** Close and reopen Command Prompt/PowerShell
4. **Manual PATH:** Add Node.js installation directory to PATH manually

### npm not found

1. **Reinstall Node.js** - npm comes with Node.js
2. **Check installation:** Make sure you downloaded the full Node.js installer
3. **Try alternative:** Use `yarn` instead of `npm` if available

### Permission errors

1. **Run as Administrator** when installing Node.js
2. **Check antivirus** - some antivirus software blocks Node.js installation

## 📞 Support

If you're having trouble with Node.js installation:

1. **Use the simple version** - it works perfectly without Node.js
2. **Check the official Node.js documentation:** https://nodejs.org/en/docs/
3. **Try a different installation method** from the list above

## 🎉 Current Status

**Your NOLIE system is working perfectly right now!**

- ✅ Enhanced AI with 4-model ensemble
- ✅ Beautiful web interface
- ✅ High accuracy deepfake detection
- ✅ Detailed analysis reports
- ✅ All core features working

The React version is just an enhancement - the core functionality is already excellent!

