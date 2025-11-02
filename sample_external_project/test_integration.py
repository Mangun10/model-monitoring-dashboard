#!/usr/bin/env python3
"""
Quick test script to verify the sample external project works with the new project code system.
"""

import os
import sys
import time
import requests
from api_client import DashboardAPIClient

def test_api_connection():
    """Test if API server is running."""
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_dashboard_connection():
    """Test if dashboard is running."""
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_project_creation():
    """Test creating a project via API."""
    client = DashboardAPIClient()
    try:
        result = client.create_project(
            project_name="Test_Project_Quick_Check",
            description="Quick test project to verify API integration"
        )
        return result.get('project_code') is not None
    except Exception as e:
        print(f"Project creation failed: {e}")
        return False

def main():
    """Run quick tests."""
    print("🧪 Quick Test - Sample External Project Integration")
    print("=" * 55)
    
    # Test 1: API Server
    print("1. Testing API server connection...")
    if test_api_connection():
        print("   ✅ API server is running")
    else:
        print("   ❌ API server is not running")
        print("   📝 Start with: python api_server.py")
        return False
    
    # Test 2: Dashboard
    print("2. Testing dashboard connection...")
    if test_dashboard_connection():
        print("   ✅ Dashboard is running")
    else:
        print("   ⚠️  Dashboard might not be running")
        print("   📝 Start with: streamlit run app.py")
    
    # Test 3: Project Creation
    print("3. Testing project creation...")
    if test_project_creation():
        print("   ✅ Project creation works")
    else:
        print("   ❌ Project creation failed")
        return False
    
    print("\n✅ All tests passed! The integration is working correctly.")
    print("🚀 You can now run: python train_and_evaluate.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)