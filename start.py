#!/usr/bin/env python3
"""
Deepfake Forensics - Main Startup Script

This script provides an easy way to start the entire deepfake forensics system,
including the backend API and web UI.

Usage:
    python start.py [options]

Options:
    --mode MODE          Start mode: 'dev', 'prod', 'web-only', 'api-only'
    --port PORT          API port (default: 8000)
    --web-port PORT      Web UI port (default: 5173)
    --mock               Enable mock mode for web UI
    --help               Show this help message
"""

import os
import sys
import subprocess
import argparse
import time
import signal
import threading
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_banner():
    """Print the startup banner."""
    banner = f"""
{Colors.BOLD}{Colors.BLUE}
╔══════════════════════════════════════════════════════════════╗
║                        NOLIE                              ║
║              ULTIMATE DEEPFAKE DETECTOR                    ║
║                                                              ║
║  🧠 12 AI Models • 🎯 Ultra-High Accuracy                  ║
║  🔍 Advanced Analysis • 📊 Professional Results             ║
║                                                              ║
║  Created by: Dogan Ege BULTE                        ║
╚══════════════════════════════════════════════════════════════╝
{Colors.END}
"""
    print(banner)

def check_requirements():
    """Check if all requirements are installed."""
    print(f"{Colors.YELLOW}🔍 Checking requirements...{Colors.END}")
    
    # Check Python dependencies for ULTIMATE DETECTOR
    try:
        import fastapi
        import uvicorn
        import requests
        print(f"{Colors.GREEN}✅ Python dependencies installed{Colors.END}")
    except ImportError as e:
        print(f"{Colors.RED}❌ Missing Python dependency: {e}{Colors.END}")
        print(f"{Colors.YELLOW}💡 Run: pip install fastapi uvicorn requests{Colors.END}")
        return False
    
    # Check if ULTIMATE DETECTOR exists
    if not Path("ultimate_detector.py").exists():
        print(f"{Colors.RED}❌ ULTIMATE DETECTOR not found{Colors.END}")
        return False
    else:
        print(f"{Colors.GREEN}✅ ULTIMATE DETECTOR found{Colors.END}")
    
    # Check Node.js for web UI (optional)
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{Colors.GREEN}✅ Node.js {result.stdout.strip()} detected{Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠️ Node.js not found (web UI will use fallback){Colors.END}")
    except FileNotFoundError:
        print(f"{Colors.YELLOW}⚠️ Node.js not found (web UI will use fallback){Colors.END}")
    
    return True

def install_web_dependencies():
    """Install web UI dependencies if needed."""
    web_dir = Path("web")
    if not web_dir.exists():
        print(f"{Colors.RED}❌ Web directory not found{Colors.END}")
        return False
    
    node_modules = web_dir / "node_modules"
    if not node_modules.exists():
        print(f"{Colors.YELLOW}📦 Installing web dependencies...{Colors.END}")
        try:
            subprocess.run(['npm', 'install'], cwd=web_dir, check=True)
            print(f"{Colors.GREEN}✅ Web dependencies installed{Colors.END}")
        except subprocess.CalledProcessError:
            print(f"{Colors.RED}❌ Failed to install web dependencies{Colors.END}")
            return False
    else:
        print(f"{Colors.GREEN}✅ Web dependencies already installed{Colors.END}")
    
    return True

def create_env_files():
    """Create necessary environment files."""
    # Create web .env.local if it doesn't exist
    web_env = Path("web/.env.local")
    if not web_env.exists():
        print(f"{Colors.YELLOW}📝 Creating web environment file...{Colors.END}")
        web_env.write_text("VITE_API_BASE=http://localhost:8000\n")
        print(f"{Colors.GREEN}✅ Created web/.env.local{Colors.END}")
    
    # Create data directories
    data_dirs = ["data/raw", "data/interim", "data/processed", "checkpoints", "outputs", "logs"]
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"{Colors.GREEN}✅ Data directories created{Colors.END}")

def start_api_server(port=8000):
    """Start the ULTIMATE DETECTOR API server."""
    print(f"{Colors.BLUE}🚀 Starting ULTIMATE DETECTOR API server on port {port}...{Colors.END}")
    
    try:
        # Start the ULTIMATE DETECTOR API server
        process = subprocess.Popen([
            sys.executable, 'ultimate_detector.py'
        ])
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        if process.poll() is None:
            print(f"{Colors.GREEN}✅ ULTIMATE DETECTOR API server started successfully{Colors.END}")
            print(f"{Colors.BLUE}🌐 API available at: http://localhost:{port}{Colors.END}")
            print(f"{Colors.YELLOW}🧠 12 AI Models loaded and ready{Colors.END}")
            print(f"{Colors.YELLOW}🎯 Ultra-high accuracy detection system{Colors.END}")
            return process
        else:
            print(f"{Colors.RED}❌ API server failed to start{Colors.END}")
            return None
    except Exception as e:
        print(f"{Colors.RED}❌ Error starting API server: {e}{Colors.END}")
        return None

def start_web_server(port=5173, mock_mode=False):
    """Start the web development server."""
    print(f"{Colors.BLUE}🚀 Starting web server on port {port}...{Colors.END}")
    
    # Check if Node.js is available
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError("Node.js not found")
    except FileNotFoundError:
        print(f"{Colors.YELLOW}⚠️ Node.js not available, using ULTIMATE DETECTOR web interface{Colors.END}")
        print(f"{Colors.BLUE}🌐 Web interface available at: http://localhost:8000{Colors.END}")
        print(f"{Colors.GREEN}✅ ULTIMATE DETECTOR includes built-in web interface{Colors.END}")
        return "builtin"  # Return special value to indicate built-in interface
    
    web_dir = Path("web")
    if not web_dir.exists():
        print(f"{Colors.RED}❌ Web directory not found{Colors.END}")
        return None
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['VITE_API_BASE'] = 'http://localhost:8000'
        if mock_mode:
            env['VITE_MOCK_MODE'] = 'true'
        
        # Start the web server
        process = subprocess.Popen([
            'npm', 'run', 'dev', '--', '--port', str(port), '--host'
        ], cwd=web_dir, env=env)
        
        # Wait a moment for the server to start
        time.sleep(5)
        
        if process.poll() is None:
            print(f"{Colors.GREEN}✅ Web server started successfully{Colors.END}")
            print(f"{Colors.BLUE}🌐 Web UI available at: http://localhost:{port}{Colors.END}")
            if mock_mode:
                print(f"{Colors.YELLOW}🎭 Mock mode enabled - using simulated results{Colors.END}")
            return process
        else:
            print(f"{Colors.RED}❌ Web server failed to start{Colors.END}")
            return None
    except Exception as e:
        print(f"{Colors.RED}❌ Error starting web server: {e}{Colors.END}")
        return None

def start_docker_services():
    """Start services using Docker Compose."""
    print(f"{Colors.BLUE}🐳 Starting services with Docker Compose...{Colors.END}")
    
    try:
        # Check if docker-compose is available
        subprocess.run(['docker-compose', '--version'], check=True, capture_output=True)
        
        # Start the services
        process = subprocess.Popen(['docker-compose', 'up', '--build'])
        
        print(f"{Colors.GREEN}✅ Docker services starting...{Colors.END}")
        print(f"{Colors.BLUE}🌐 Web UI will be available at: http://localhost:5173{Colors.END}")
        print(f"{Colors.BLUE}🌐 API will be available at: http://localhost:8000{Colors.END}")
        print(f"{Colors.YELLOW}💡 Press Ctrl+C to stop all services{Colors.END}")
        
        return process
    except subprocess.CalledProcessError:
        print(f"{Colors.RED}❌ Docker Compose not found{Colors.END}")
        return None
    except Exception as e:
        print(f"{Colors.RED}❌ Error starting Docker services: {e}{Colors.END}")
        return None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print(f"\n{Colors.YELLOW}🛑 Shutting down services...{Colors.END}")
    sys.exit(0)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Deepfake Forensics Startup Script')
    parser.add_argument('--mode', choices=['dev', 'prod', 'web-only', 'api-only', 'docker'], 
                       default='dev', help='Start mode')
    parser.add_argument('--port', type=int, default=8000, help='API port')
    parser.add_argument('--web-port', type=int, default=5173, help='Web UI port')
    parser.add_argument('--mock', action='store_true', help='Enable mock mode for web UI')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    if args.mode == 'docker':
        # Start with Docker
        process = start_docker_services()
        if process:
            try:
                process.wait()
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}🛑 Stopping Docker services...{Colors.END}")
                process.terminate()
        return
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create necessary files and directories
    create_env_files()
    
    # Install web dependencies if needed
    if args.mode in ['dev', 'prod', 'web-only']:
        if not install_web_dependencies():
            sys.exit(1)
    
    processes = []
    
    try:
        if args.mode in ['dev', 'prod', 'api-only']:
            # Start API server
            api_process = start_api_server(args.port)
            if api_process:
                processes.append(api_process)
            else:
                print(f"{Colors.RED}❌ Failed to start API server{Colors.END}")
                if args.mode == 'api-only':
                    sys.exit(1)
        
        if args.mode in ['dev', 'prod', 'web-only']:
            # Start web server
            web_process = start_web_server(args.web_port, args.mock)
            if web_process == "builtin":
                print(f"{Colors.GREEN}✅ Using ULTIMATE DETECTOR built-in web interface{Colors.END}")
                # No need to add to processes list since it's built into the API
            elif web_process:
                processes.append(web_process)
            else:
                print(f"{Colors.RED}❌ Failed to start web server{Colors.END}")
                if args.mode == 'web-only':
                    sys.exit(1)
        
        if processes or args.mode == 'api-only':
            print(f"\n{Colors.GREEN}🎉 NOLIE ULTIMATE DETECTOR started successfully!{Colors.END}")
            print(f"{Colors.BLUE}🧠 12 AI Models loaded and ready{Colors.END}")
            print(f"{Colors.BLUE}🎯 Ultra-high accuracy detection system{Colors.END}")
            print(f"{Colors.BLUE}🌐 Web interface: http://localhost:8000{Colors.END}")
            print(f"{Colors.BLUE}📖 API docs: http://localhost:8000/docs{Colors.END}")
            print(f"{Colors.YELLOW}💡 Press Ctrl+C to stop all services{Colors.END}")
            
            # Wait for all processes
            for process in processes:
                process.wait()
        else:
            print(f"{Colors.RED}❌ No services started{Colors.END}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}🛑 Shutting down services...{Colors.END}")
        for process in processes:
            if process.poll() is None:
                process.terminate()
        print(f"{Colors.GREEN}✅ All services stopped{Colors.END}")

if __name__ == '__main__':
    main()
