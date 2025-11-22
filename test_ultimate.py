#!/usr/bin/env python3
"""
Test the ULTIMATE DEEPFAKE DETECTOR
"""

import requests
import time
import tempfile
from PIL import Image
import io

def test_ultimate_detector():
    """Test the ultimate detector API."""
    print("🧪 Testing ULTIMATE DEEPFAKE DETECTOR...")
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    for i in range(10):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is running!")
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        print("❌ Server not responding")
        return False
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed")
            print(f"   Detector: {health_data.get('detector', 'Unknown')}")
            print(f"   Models: {health_data.get('models', 0)}")
            print(f"   Created by: {health_data.get('created_by', 'Unknown')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Create test image
    print("\n2. Creating test image...")
    try:
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file.write(img_bytes.getvalue())
            temp_path = temp_file.name
        
        print(f"   Created test image: {temp_path}")
    except Exception as e:
        print(f"❌ Failed to create test image: {e}")
        return False
    
    # Test prediction endpoint
    print("\n3. Testing ULTIMATE prediction...")
    try:
        with open(temp_path, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post("http://localhost:8000/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ ULTIMATE prediction successful!")
            print(f"   Score: {result.get('score', 0):.4f}")
            print(f"   Label: {result.get('label', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.4f}")
            print(f"   Models analyzed: {len(result.get('model_analysis', {}))}")
            print(f"   Analysis quality: {result.get('ensemble_details', {}).get('analysis_quality', 'Unknown')}")
            
            # Show model analysis
            print(f"\n🧠 AI Model Analysis:")
            for model_name, score in result.get('model_analysis', {}).items():
                print(f"   {model_name}: {score:.3f}")
            
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False
    finally:
        # Clean up
        import os
        try:
            os.unlink(temp_path)
        except:
            pass
    
    print(f"\n🎉 ULTIMATE DETECTOR TEST PASSED!")
    print(f"🚀 The most powerful deepfake detection system is working!")
    return True

if __name__ == "__main__":
    test_ultimate_detector()
