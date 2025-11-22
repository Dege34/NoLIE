# 🚀 Start Here - Deepfake Forensics

Welcome to Deepfake Forensics! This is your quick start guide to get everything running.

## ⚡ Super Quick Start

### Windows Users
```cmd
start.bat
```

### Mac/Linux Users
```bash
./start.sh
```

### Python Users
```bash
python start.py
```

**That's it!** The system will automatically:
- ✅ Check and install dependencies
- ✅ Set up the environment
- ✅ Start the API server
- ✅ Start the web interface
- ✅ Open your browser to http://localhost:5173

## 🎯 What You'll Get

After running the startup script, you'll have:

- **Web Interface**: http://localhost:5173
  - Drag & drop file upload
  - Real-time deepfake detection
  - Interactive visualizations
  - Results export

- **API Server**: http://localhost:8000
  - REST API for programmatic access
  - Health check endpoint
  - File upload and prediction

## 🎭 Try It Out

1. **Upload a Test File**: Drag any image or video to the web interface
2. **View Results**: See the confidence score and analysis
3. **Explore Features**: Check out heatmaps, timelines, and detailed results
4. **Export Data**: Download reports and annotated frames

## 🔧 Troubleshooting

### If Something Goes Wrong

1. **Check Prerequisites**:
   - Python 3.11+ installed
   - Node.js 18+ installed
   - Git installed

2. **Manual Setup**:
   ```bash
   # Install Python dependencies
   pip install -e .
   
   # Install web dependencies
   cd web && npm install && cd ..
   
   # Start manually
   python start.py
   ```

3. **Docker Alternative**:
   ```bash
   docker-compose up --build
   ```

### Common Issues

- **Port already in use**: The script will try different ports automatically
- **Missing dependencies**: The script will install them automatically
- **Permission denied**: Make sure you have write permissions in the directory

## 📚 Next Steps

1. **Read the Full Guide**: [GETTING_STARTED.md](GETTING_STARTED.md)
2. **Explore the API**: Check out the API documentation
3. **Train Models**: Follow the training guide
4. **Customize**: Modify settings and configurations

## 🆘 Need Help?

- **Documentation**: [README.md](README.md)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

## 🎉 You're Ready!

Run the startup script and start detecting deepfakes! 🕵️‍♂️

---

**Happy detecting!** 🚀

