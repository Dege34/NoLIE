# Getting Started with Deepfake Forensics

This guide will help you get the Deepfake Forensics system up and running quickly.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

**For Windows:**
```cmd
start.bat
```

**For Unix/Linux/Mac:**
```bash
./start.sh
```

**For Python users:**
```bash
python start.py
```

### Option 2: Manual Setup

1. **Install Python Dependencies**
   ```bash
   pip install -e .
   ```

2. **Install Web Dependencies**
   ```bash
   cd web
   npm install
   cd ..
   ```

3. **Start the API Server**
   ```bash
   python -m deepfake_forensics.cli serve --host 0.0.0.0 --port 8000
   ```

4. **Start the Web UI** (in a new terminal)
   ```bash
   cd web
   npm run dev
   ```

5. **Access the Application**
   - Web UI: http://localhost:5173
   - API: http://localhost:8000

## 📋 Prerequisites

### Required Software

- **Python 3.11+** with pip
- **Node.js 18+** with npm
- **Git** (for cloning the repository)

### Optional Software

- **Docker** and **Docker Compose** (for containerized deployment)
- **CUDA** (for GPU acceleration)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/deepfake-forensics/deepfake-forensics.git
cd deepfake-forensics
```

### 2. Install Python Dependencies

```bash
# Install the package in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### 3. Install Web Dependencies

```bash
cd web
npm install
cd ..
```

### 4. Set Up Environment

The startup scripts will automatically create the necessary environment files and directories. If you're setting up manually:

```bash
# Create environment file for web UI
echo "VITE_API_BASE=http://localhost:8000" > web/.env.local

# Create data directories
mkdir -p data/raw data/interim data/processed checkpoints outputs logs
```

## 🎯 Usage

### Web Interface

1. **Upload Files**: Drag and drop images or videos onto the upload area
2. **View Results**: See real-time analysis results with confidence scores
3. **Explore Visualizations**: View heatmaps, video timelines, and detailed explanations
4. **Export Reports**: Download JSON reports and annotated frames

### API Usage

```python
import requests

# Upload a file for analysis
with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    result = response.json()
    print(f"Score: {result['score']}, Label: {result['label']}")
```

### CLI Usage

```bash
# Train a model
python -m deepfake_forensics.cli train --config configs/train_small.yaml

# Run inference
python -m deepfake_forensics.cli predict --input test.jpg --model checkpoints/best.ckpt

# Start API server
python -m deepfake_forensics.cli serve --model checkpoints/best.ckpt
```

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop services
docker-compose down
```

### Using Individual Containers

```bash
# Build API container
docker build -t deepfake-forensics-api .

# Build Web container
docker build -t deepfake-forensics-web ./web

# Run API container
docker run -p 8000:8000 deepfake-forensics-api

# Run Web container
docker run -p 5173:80 deepfake-forensics-web
```

## 🧪 Testing

### Run Tests

```bash
# Python tests
python -m pytest tests/

# Web UI tests
cd web
npm run test
cd ..

# All tests
python start.py --mode test
```

### Test Coverage

```bash
# Python coverage
python -m pytest tests/ --cov=deepfake_forensics

# Web UI coverage
cd web
npm run test -- --coverage
cd ..
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill processes using the ports
   lsof -ti:8000 | xargs kill -9
   lsof -ti:5173 | xargs kill -9
   ```

2. **Python Dependencies Not Found**
   ```bash
   # Reinstall dependencies
   pip install -e .
   ```

3. **Node.js Dependencies Not Found**
   ```bash
   # Reinstall web dependencies
   cd web
   npm install
   cd ..
   ```

4. **Permission Denied (Unix/Linux)**
   ```bash
   # Make scripts executable
   chmod +x start.sh
   chmod +x web/setup.sh
   ```

### Logs and Debugging

- **API Logs**: Check the terminal where the API server is running
- **Web Logs**: Check the terminal where the web server is running
- **Docker Logs**: `docker-compose logs`

### Getting Help

- **Documentation**: Check the main README.md
- **Issues**: Report issues on GitHub
- **Discussions**: Join the community discussions

## 📚 Next Steps

1. **Explore the Web UI**: Upload some test files and explore the interface
2. **Read the Documentation**: Check out the detailed README.md
3. **Try the API**: Experiment with the REST API endpoints
4. **Customize Settings**: Adjust settings in the web interface
5. **Train Models**: Follow the training guide to train your own models

## 🎭 Mock Mode

For demonstration purposes, you can enable mock mode:

```bash
# Using the startup script
python start.py --mock

# Or set environment variable
export VITE_MOCK_MODE=true
cd web && npm run dev
```

Mock mode provides simulated results without requiring a trained model.

## 🔒 Security Notes

- Files are processed locally and not stored permanently
- No personal data is collected or transmitted
- Results are not shared with third parties
- All processing is done securely

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Happy detecting! 🕵️‍♂️**
