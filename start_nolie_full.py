#!/usr/bin/env python3
"""
NOLIE Full System Startup - Fixes Node.js and starts React website
Created by Dogan Ege BULTE
"""

import sys
import os
import subprocess
import time
import webbrowser
import signal
import threading
import json
import requests
from pathlib import Path
from typing import Optional, List

class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

class NOLIEFull:
    """Full NOLIE system with React website."""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.api_port = 8000
        self.web_port = 5173
        self.api_url = f"http://localhost:{self.api_port}"
        self.web_url = f"http://localhost:{self.web_port}"
        
    def print_banner(self):
        """Print the NOLIE banner."""
        banner = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗
║                        {Colors.BOLD}NOLIE{Colors.END}{Colors.CYAN}                              ║
║              {Colors.BOLD}Advanced Deepfake Detection System{Colors.END}{Colors.CYAN}           ║
║                                                              ║
║  🔍 AI-Powered Deepfake Detection                            ║
║  🎯 Ultra-High Accuracy & Reliability                        ║
║  🌐 Full React Web Interface                                 ║
║  📊 Detailed Analysis & Reports                              ║
║                                                              ║
║  {Colors.BOLD}Created by: Dogan Ege BULTE{Colors.END}{Colors.CYAN}                        ║
║  {Colors.GREEN}Full Mode - React + Enhanced AI{Colors.END}{Colors.CYAN}                      ║
╚══════════════════════════════════════════════════════════════╝{Colors.END}
"""
        print(banner)
    
    def fix_nodejs_path(self) -> bool:
        """Try to fix Node.js PATH issues."""
        print(f"{Colors.BLUE}🔧 Attempting to fix Node.js PATH issues...{Colors.END}")
        
        # Common Node.js installation paths
        common_paths = [
            r"C:\Program Files\nodejs",
            r"C:\Program Files (x86)\nodejs",
            os.path.expanduser(r"~\AppData\Roaming\npm"),
            r"C:\Users\{}\AppData\Roaming\npm".format(os.getenv('USERNAME', '')),
        ]
        
        # Add paths to current session
        current_path = os.environ.get('PATH', '')
        for path in common_paths:
            if os.path.exists(path) and path not in current_path:
                os.environ['PATH'] = current_path + os.pathsep + path
                print(f"{Colors.GREEN}✅ Added to PATH: {path}{Colors.END}")
        
        return True
    
    def check_nodejs(self) -> bool:
        """Check if Node.js and npm are working."""
        print(f"{Colors.BLUE}🔍 Checking Node.js installation...{Colors.END}")
        
        # Try to fix PATH first
        self.fix_nodejs_path()
        
        # Test Node.js
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                node_version = result.stdout.strip()
                print(f"{Colors.GREEN}✅ Node.js {node_version} detected{Colors.END}")
            else:
                print(f"{Colors.RED}❌ Node.js not working{Colors.END}")
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"{Colors.RED}❌ Node.js not found{Colors.END}")
            return False
        
        # Test npm
        try:
            result = subprocess.run(['npm', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                npm_version = result.stdout.strip()
                print(f"{Colors.GREEN}✅ npm {npm_version} detected{Colors.END}")
                return True
            else:
                print(f"{Colors.RED}❌ npm not working{Colors.END}")
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"{Colors.RED}❌ npm not found{Colors.END}")
            return False
    
    def install_nodejs_guide(self):
        """Show Node.js installation guide."""
        print(f"\n{Colors.YELLOW}📝 Node.js Installation Guide:{Colors.END}")
        print(f"{Colors.BLUE}1. Download Node.js from: https://nodejs.org/{Colors.END}")
        print(f"{Colors.BLUE}2. Choose the LTS version (recommended){Colors.END}")
        print(f"{Colors.BLUE}3. Run the installer and follow these steps:{Colors.END}")
        print(f"   - Accept the license agreement")
        print(f"   - Choose installation directory (default is fine)")
        print(f"   - {Colors.BOLD}IMPORTANT: Check 'Add to PATH' option{Colors.END}")
        print(f"   - Complete the installation")
        print(f"{Colors.BLUE}4. Restart your computer{Colors.END}")
        print(f"{Colors.BLUE}5. Run this script again{Colors.END}")
        print(f"\n{Colors.GREEN}✅ After installation, you'll get the full React NOLIE experience!{Colors.END}")
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        print(f"{Colors.BLUE}🔍 Checking system requirements...{Colors.END}")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print(f"{Colors.RED}❌ Python 3.8+ required. Current: {sys.version}{Colors.END}")
            return False
        print(f"{Colors.GREEN}✅ Python {sys.version.split()[0]} detected{Colors.END}")
        
        # Check Node.js
        if not self.check_nodejs():
            self.install_nodejs_guide()
            return False
        
        # Check required Python packages
        required_packages = ['fastapi', 'uvicorn', 'requests']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"{Colors.GREEN}✅ {package} installed{Colors.END}")
            except ImportError:
                missing_packages.append(package)
                print(f"{Colors.RED}❌ {package} not found{Colors.END}")
        
        if missing_packages:
            print(f"{Colors.YELLOW}📦 Installing missing Python packages...{Colors.END}")
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages, 
                             check=True, capture_output=True)
                print(f"{Colors.GREEN}✅ All Python packages installed successfully{Colors.END}")
            except subprocess.CalledProcessError as e:
                print(f"{Colors.RED}❌ Failed to install packages: {e}{Colors.END}")
                return False
        
        return True
    
    def install_web_dependencies(self) -> bool:
        """Install web dependencies."""
        print(f"{Colors.BLUE}📦 Installing React web dependencies...{Colors.END}")
        
        web_dir = Path("web")
        if not web_dir.exists():
            print(f"{Colors.RED}❌ Web directory not found{Colors.END}")
            return False
        
        try:
            # Clear npm cache first
            subprocess.run(['npm', 'cache', 'clean', '--force'], cwd=web_dir, 
                          capture_output=True, timeout=60)
            
            # Install dependencies
            result = subprocess.run(['npm', 'install'], cwd=web_dir, 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"{Colors.GREEN}✅ React dependencies installed successfully{Colors.END}")
                return True
            else:
                print(f"{Colors.RED}❌ Failed to install React dependencies{Colors.END}")
                print(f"{Colors.RED}Error: {result.stderr}{Colors.END}")
                return False
        except subprocess.TimeoutExpired:
            print(f"{Colors.RED}❌ Installation timed out{Colors.END}")
            return False
        except Exception as e:
            print(f"{Colors.RED}❌ Error installing React dependencies: {e}{Colors.END}")
            return False
    
    def create_directories(self):
        """Create necessary directories."""
        directories = ['data', 'checkpoints', 'outputs', 'logs']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        print(f"{Colors.GREEN}✅ Data directories created{Colors.END}")
    
    def start_api_server(self) -> bool:
        """Start the enhanced API server."""
        print(f"{Colors.BLUE}🚀 Starting NOLIE API Server...{Colors.END}")
        
        try:
            # Use the enhanced simple API
            process = subprocess.Popen([
                sys.executable, 'simple_api.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(process)
            
            # Wait for server to start
            for i in range(15):
                time.sleep(1)
                try:
                    response = requests.get(f"{self.api_url}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"{Colors.GREEN}✅ NOLIE API Server started successfully{Colors.END}")
                        print(f"{Colors.BLUE}🌐 API available at: {self.api_url}{Colors.END}")
                        print(f"{Colors.BLUE}📖 API docs at: {self.api_url}/docs{Colors.END}")
                        return True
                except requests.exceptions.RequestException:
                    continue
            
            print(f"{Colors.RED}❌ API Server failed to start{Colors.END}")
            return False
            
        except Exception as e:
            print(f"{Colors.RED}❌ Error starting API server: {e}{Colors.END}")
            return False
    
    def start_react_server(self) -> bool:
        """Start the React development server."""
        print(f"{Colors.BLUE}🚀 Starting NOLIE React Web Interface...{Colors.END}")
        
        web_dir = Path("web")
        if not web_dir.exists():
            print(f"{Colors.RED}❌ Web directory not found{Colors.END}")
            return False
        
        try:
            # Start the React development server
            process = subprocess.Popen([
                'npm', 'run', 'dev'
            ], cwd=web_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(process)
            
            # Wait for server to start
            for i in range(30):
                time.sleep(1)
                try:
                    response = requests.get(f"{self.web_url}", timeout=2)
                    if response.status_code == 200:
                        print(f"{Colors.GREEN}✅ NOLIE React Interface started successfully{Colors.END}")
                        print(f"{Colors.BLUE}🌐 React app available at: {self.web_url}{Colors.END}")
                        return True
                except requests.exceptions.RequestException:
                    continue
            
            print(f"{Colors.RED}❌ React server failed to start{Colors.END}")
            return False
            
        except Exception as e:
            print(f"{Colors.RED}❌ Error starting React server: {e}{Colors.END}")
            return False
    
    def open_web_interface(self):
        """Open the React web interface in browser."""
        print(f"{Colors.BLUE}🌐 Opening NOLIE React Interface...{Colors.END}")
        webbrowser.open(self.web_url)
        print(f"{Colors.GREEN}✅ React interface opened in browser{Colors.END}")
    
    def test_system(self) -> bool:
        """Test the system functionality."""
        print(f"{Colors.BLUE}🧪 Testing NOLIE system functionality...{Colors.END}")
        
        try:
            # Test API health
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code != 200:
                print(f"{Colors.RED}❌ API health check failed{Colors.END}")
                return False
            
            print(f"{Colors.GREEN}✅ API health check passed{Colors.END}")
            
            # Test React interface
            response = requests.get(f"{self.web_url}", timeout=5)
            if response.status_code != 200:
                print(f"{Colors.RED}❌ React interface check failed{Colors.END}")
                return False
            
            print(f"{Colors.GREEN}✅ React interface check passed{Colors.END}")
            
            # Test prediction endpoint
            test_file = Path("test_api.py")
            if test_file.exists():
                result = subprocess.run([sys.executable, "test_api.py"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"{Colors.GREEN}✅ Ultra-advanced AI prediction test passed{Colors.END}")
                    return True
                else:
                    print(f"{Colors.RED}❌ Prediction test failed{Colors.END}")
                    return False
            else:
                print(f"{Colors.YELLOW}⚠️ Test file not found, skipping prediction test{Colors.END}")
                return True
                
        except Exception as e:
            print(f"{Colors.RED}❌ System test failed: {e}{Colors.END}")
            return False
    
    def show_status(self):
        """Show system status."""
        print(f"\n{Colors.CYAN}📊 NOLIE Full System Status:{Colors.END}")
        print(f"{Colors.BLUE}🔗 API Server: {self.api_url}{Colors.END}")
        print(f"{Colors.BLUE}🌐 React Interface: {self.web_url}{Colors.END}")
        print(f"{Colors.BLUE}📁 Working Directory: {Path.cwd()}{Colors.END}")
        print(f"{Colors.BLUE}🐍 Python Version: {sys.version.split()[0]}{Colors.END}")
        
        # Show running processes
        if self.processes:
            print(f"{Colors.GREEN}✅ {len(self.processes)} service(s) running{Colors.END}")
        else:
            print(f"{Colors.RED}❌ No services running{Colors.END}")
    
    def cleanup(self):
        """Clean up running processes."""
        print(f"\n{Colors.YELLOW}🧹 Cleaning up NOLIE system...{Colors.END}")
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                print(f"{Colors.RED}❌ Error stopping process: {e}{Colors.END}")
        
        self.processes.clear()
        print(f"{Colors.GREEN}✅ Cleanup completed{Colors.END}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n{Colors.YELLOW}🛑 Shutdown signal received{Colors.END}")
        self.cleanup()
        sys.exit(0)
    
    def run(self):
        """Main run method."""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Print banner
            self.print_banner()
            
            # Check dependencies
            if not self.check_dependencies():
                print(f"{Colors.RED}❌ Dependency check failed{Colors.END}")
                return False
            
            # Install React dependencies
            if not self.install_web_dependencies():
                print(f"{Colors.RED}❌ Failed to install React dependencies{Colors.END}")
                return False
            
            # Create directories
            self.create_directories()
            
            # Start API server
            if not self.start_api_server():
                print(f"{Colors.RED}❌ Failed to start API server{Colors.END}")
                return False
            
            # Start React server
            if not self.start_react_server():
                print(f"{Colors.RED}❌ Failed to start React server{Colors.END}")
                return False
            
            # Test system
            if not self.test_system():
                print(f"{Colors.RED}❌ System test failed{Colors.END}")
                return False
            
            # Open web interface
            self.open_web_interface()
            
            # Show status
            self.show_status()
            
            print(f"\n{Colors.GREEN}🎉 NOLIE Full System Started Successfully!{Colors.END}")
            print(f"{Colors.CYAN}📖 Usage Instructions:{Colors.END}")
            print(f"  1. Upload images or videos using the React interface")
            print(f"  2. Get ultra-advanced AI deepfake detection results")
            print(f"  3. View detailed 6-model ensemble analysis")
            print(f"  4. Press Ctrl+C to stop the system")
            
            # Keep running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            
            return True
            
        except Exception as e:
            print(f"{Colors.RED}❌ Unexpected error: {e}{Colors.END}")
            return False
        finally:
            self.cleanup()

def main():
    """Main entry point."""
    app = NOLIEFull()
    success = app.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
