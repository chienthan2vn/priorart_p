"""
Startup script for the Mock Patent AI Agent Web Application
Runs without requiring LLM resources - perfect for testing the interface
"""

import os
import sys
import uvicorn
from pathlib import Path

def main():
    """Start the Mock FastAPI application"""
    print("🚀 Starting Mock Patent AI Agent Web Application")
    print("=" * 60)
    print("📝 MOCK VERSION - No LLM resources required!")
    print("✨ Perfect for testing the web interface")
    print("=" * 60)
    
    # Check if we're in the correct directory
    if not Path("app_mock.py").exists():
        print("❌ Error: app_mock.py not found in current directory")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if static directory exists
    if not Path("static").exists():
        print("❌ Error: static directory not found")
        print("Please ensure the static directory with HTML/JS files exists")
        sys.exit(1)
    
    print("✅ All required files found")
    print("📂 Static files directory: ./static")
    print("🎭 Using mock responses for all AI operations")
    print("🔧 Starting Mock FastAPI server with Uvicorn...")
    print("")
    print("🌐 Web interface: http://localhost:8000")
    print("📚 API documentation: http://localhost:8000/docs")
    print("ℹ️  Mock info endpoint: http://localhost:8000/api/mock/info")
    print("🔄 Reset mock data: http://localhost:8000/api/mock/reset")
    print("=" * 60)
    print("💡 Features available in mock mode:")
    print("   ✅ Keyword extraction with realistic sample data")
    print("   ✅ IPC classification with mock predictions")
    print("   ✅ Patent URL analysis with template responses")
    print("   ✅ Similarity evaluation with computed scores")
    print("   ✅ Session management with mock data")
    print("   ✅ All UI components fully functional")
    print("=" * 60)
    
    try:
        # Start the mock server
        uvicorn.run(
            "app_mock:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable auto-reload for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Mock server stopped by user")
    except Exception as e:
        print(f"❌ Error starting mock server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
