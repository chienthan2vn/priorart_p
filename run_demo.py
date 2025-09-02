#!/usr/bin/env python3
"""
Launch script for the Patent AI Agent Demo (Mock LLM version)
No LLM infrastructure required - uses simulated responses
"""

import subprocess
import sys
import os

def main():
    """Launch the demo Streamlit application"""
    
    print("🎭 Starting Patent AI Agent - DEMO MODE")
    print("=" * 60)
    print("🚀 Mock LLM Demo - No actual LLM infrastructure required!")
    print("📱 All AI responses are pre-programmed mock data")
    print("🎯 Perfect for testing the interface and workflow")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("streamlit_demo_app.py"):
        print("❌ Error: streamlit_demo_app.py not found in current directory")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit>=1.28.0"])
            print("✅ Streamlit installed successfully")
        except Exception as e:
            print(f"❌ Failed to install Streamlit: {e}")
            print("Please install manually: pip install streamlit>=1.28.0")
            sys.exit(1)
    
    # Check basic dependencies
    try:
        import pandas
        print(f"✅ Pandas available")
    except ImportError:
        print("⚠️  Pandas not found - installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    
    try:
        import pydantic
        print(f"✅ Pydantic available")
    except ImportError:
        print("⚠️  Pydantic not found - installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pydantic>=2.0.0"])
    
    # Test mock extractor import
    try:
        from src.core.mock_extractor import MockCoreConceptExtractor
        print("✅ Mock extractor ready")
    except ImportError as e:
        print(f"❌ Mock extractor import failed: {e}")
        print("Please ensure you're in the correct directory and all files are present")
        sys.exit(1)
    
    # Launch streamlit demo
    try:
        print("\n🌐 Launching Demo Streamlit application...")
        print("📱 The demo will open in your default web browser")
        print("🔗 URL: http://localhost:8502")
        print("\n🎭 DEMO FEATURES:")
        print("  • Mock AI responses (no LLM required)")
        print("  • Full interactive workflow")
        print("  • Real approve/reject/edit functionality") 
        print("  • Complete results export")
        print("\n⚠️  To stop the demo, press Ctrl+C in this terminal\n")
        
        # Run streamlit demo on different port to avoid conflicts
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_demo_app.py",
            "--server.port=8502",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
