#!/usr/bin/env python3
"""
Launch script for Enhanced Streamlit Patent AI Agent
Uses the enhanced_mock_extractor.py with integrated Streamlit functionality
"""

import sys
import subprocess
import os

def main():
    """Launch the enhanced Streamlit application"""
    
    print("🚀 Starting Enhanced Patent AI Agent - Streamlit Interface")
    print("=" * 60)
    print("📋 Enhanced Mock Extractor with Full LangGraph Architecture")
    print("🎭 Uses mock LLM responses for demonstration")
    print("🏗️ Maintains exact multi-agent workflow from original extractor.py")
    print("=" * 60)
    
    # Check if streamlit is available
    try:
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("❌ Streamlit is not installed. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "pandas"])
        print("✅ Streamlit installed successfully")
    
    # Check if the enhanced app file exists
    app_file = "enhanced_streamlit_app.py"
    if not os.path.exists(app_file):
        print(f"❌ Enhanced Streamlit app file '{app_file}' not found!")
        print("Please make sure the file exists in the current directory.")
        sys.exit(1)
    
    print(f"📁 Found enhanced app file: {app_file}")
    print("🌐 Starting Streamlit server...")
    print("=" * 60)
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit server stopped by user")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
