#!/usr/bin/env python3
"""
Simple test script for the deepfake detection API.
"""

import sys
import os
import tempfile
import numpy as np
from PIL import Image
import requests
import time

def create_test_image():
    """Create a test image for testing."""
    # Create a random test image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(temp_file.name)
    temp_file.close()
    
    return temp_file.name

def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Deepfake Detection API...")
    
    # Test health endpoint
    try:
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test prediction endpoint
    try:
        print("\n2. Testing prediction endpoint...")
        
        # Create test image
        test_image_path = create_test_image()
        print(f"   Created test image: {test_image_path}")
        
        # Upload and predict
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/predict", files=files, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful")
            print(f"   Score: {result.get('score', 'N/A')}")
            print(f"   Label: {result.get('label', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 'N/A')}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        # Clean up
        os.unlink(test_image_path)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    if test_api():
        sys.exit(0)
    else:
        sys.exit(1)
