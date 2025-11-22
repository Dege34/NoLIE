#!/usr/bin/env python3
"""
NOLIE Deepfake Detection System - Simple Version
Created by Dogan Ege BULTE

This version works without Node.js by using the enhanced HTML interface.
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

class NOLIESimple:
    """Simple NOLIE system that works without Node.js."""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.api_port = 8000
        self.api_url = f"http://localhost:{self.api_port}"
        
    def print_banner(self):
        """Print the NOLIE banner."""
        banner = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗
║                        {Colors.BOLD}NOLIE{Colors.END}{Colors.CYAN}                              ║
║              {Colors.BOLD}Advanced Deepfake Detection System{Colors.END}{Colors.CYAN}           ║
║                                                              ║
║  🔍 AI-Powered Deepfake Detection                            ║
║  🎯 High Accuracy & Reliability                              ║
║  🌐 Enhanced Web Interface                                   ║
║  📊 Detailed Analysis & Reports                              ║
║                                                              ║
║  {Colors.BOLD}Created by: Dogan Ege BULTE{Colors.END}{Colors.CYAN}                        ║
║  {Colors.YELLOW}Simple Mode - No Node.js Required{Colors.END}{Colors.CYAN}                    ║
╚══════════════════════════════════════════════════════════════╝{Colors.END}
"""
        print(banner)
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        print(f"{Colors.BLUE}🔍 Checking system requirements...{Colors.END}")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print(f"{Colors.RED}❌ Python 3.8+ required. Current: {sys.version}{Colors.END}")
            return False
        print(f"{Colors.GREEN}✅ Python {sys.version.split()[0]} detected{Colors.END}")
        
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
    
    def open_web_interface(self):
        """Open the enhanced HTML web interface."""
        web_file = Path("web_enhanced.html")
        if web_file.exists():
            print(f"{Colors.BLUE}🌐 Opening NOLIE Web Interface...{Colors.END}")
            webbrowser.open(f"file://{web_file.absolute()}")
            print(f"{Colors.GREEN}✅ Web interface opened in browser{Colors.END}")
        else:
            print(f"{Colors.RED}❌ Web interface file not found{Colors.END}")
    
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
            
            # Test prediction endpoint
            test_file = Path("test_api.py")
            if test_file.exists():
                result = subprocess.run([sys.executable, "test_api.py"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"{Colors.GREEN}✅ Prediction test passed{Colors.END}")
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
        print(f"\n{Colors.CYAN}📊 NOLIE System Status:{Colors.END}")
        print(f"{Colors.BLUE}🔗 API Server: {self.api_url}{Colors.END}")
        print(f"{Colors.BLUE}🌐 Web Interface: Enhanced HTML Interface{Colors.END}")
        print(f"{Colors.BLUE}📁 Working Directory: {Path.cwd()}{Colors.END}")
        print(f"{Colors.BLUE}🐍 Python Version: {sys.version.split()[0]}{Colors.END}")
        
        # Show running processes
        if self.processes:
            print(f"{Colors.GREEN}✅ {len(self.processes)} service(s) running{Colors.END}")
        else:
            print(f"{Colors.RED}❌ No services running{Colors.END}")
    
    def show_nodejs_instructions(self):
        """Show instructions for installing Node.js."""
        print(f"\n{Colors.YELLOW}📝 To get the full React experience, install Node.js:{Colors.END}")
        print(f"{Colors.BLUE}1. Download Node.js from: https://nodejs.org/{Colors.END}")
        print(f"{Colors.BLUE}2. Install the LTS version (recommended){Colors.END}")
        print(f"{Colors.BLUE}3. Restart your computer after installation{Colors.END}")
        print(f"{Colors.BLUE}4. Run 'start_nolie.bat' for the full React interface{Colors.END}")
        print(f"{Colors.GREEN}✅ For now, you can use the enhanced HTML interface!{Colors.END}")
    
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
            
            # Create directories
            self.create_directories()
            
            # Start API server
            if not self.start_api_server():
                print(f"{Colors.RED}❌ Failed to start API server{Colors.END}")
                return False
            
            # Test system
            if not self.test_system():
                print(f"{Colors.RED}❌ System test failed{Colors.END}")
                return False
            
            # Open web interface
            self.open_web_interface()
            
            # Show status
            self.show_status()
            
            # Show Node.js instructions
            self.show_nodejs_instructions()
            
            print(f"\n{Colors.GREEN}🎉 NOLIE System Started Successfully!{Colors.END}")
            print(f"{Colors.CYAN}📖 Usage Instructions:{Colors.END}")
            print(f"  1. Upload images or videos using the web interface")
            print(f"  2. Get AI-powered deepfake detection results")
            print(f"  3. View detailed analysis and confidence scores")
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
    app = NOLIESimple()
    success = app.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
