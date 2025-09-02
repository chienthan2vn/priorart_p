#!/usr/bin/env python3
"""
Launch script for the Patent AI Agent Streamlit interface
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    
    print("🚀 Starting Patent AI Agent - Streamlit Interface")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("streamlit_app.py"):
        print("❌ Error: streamlit_app.py not found in current directory")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit>=1.28.0"])
    
    # Launch streamlit
    try:
        print("\n🌐 Launching Streamlit application...")
        print("📱 The application will open in your default web browser")
        print("🔗 URL: http://localhost:8501")
        print("\n⚠️  To stop the application, press Ctrl+C in this terminal\n")
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
