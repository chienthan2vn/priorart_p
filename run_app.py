"""
Startup script for the Patent AI Agent Web Application
"""

import os
import sys
import uvicorn
from pathlib import Path

def main():
    """Start the FastAPI application"""
    print("🚀 Starting Patent AI Agent Web Application")
    print("=" * 60)
    
    # Check if we're in the correct directory
    if not Path("app.py").exists():
        print("❌ Error: app.py not found in current directory")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if static directory exists
    if not Path("static").exists():
        print("❌ Error: static directory not found")
        print("Please ensure the static directory with HTML/JS files exists")
        sys.exit(1)
    
    print("✅ All required files found")
    print("📂 Static files directory: ./static")
    print("🔧 Starting FastAPI server with Uvicorn...")
    print("🌐 Web interface will be available at: http://localhost:8000")
    print("📚 API documentation will be available at: http://localhost:8000/docs")
    print("=" * 60)
    
    try:
        # Start the server
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable auto-reload for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
