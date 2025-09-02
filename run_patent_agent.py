#!/usr/bin/env python3
"""
Patent AI Agent - Unified Launcher
Choose between different interfaces and modes
"""

import subprocess
import sys
import os

def print_header():
    """Print application header"""
    print("🚀 Patent AI Agent - Keyword Extraction System")
    print("=" * 60)
    print("Choose your preferred interface and mode:")
    print()

def print_options():
    """Print available options"""
    print("📋 AVAILABLE OPTIONS:")
    print()
    
    print("1. 🎭 DEMO MODE (Recommended for testing)")
    print("   • Mock LLM responses - no infrastructure needed")
    print("   • Complete Streamlit web interface")
    print("   • Full interactive workflow")
    print("   • Perfect for demonstrations and testing")
    print()
    
    print("2. 🌐 WEB INTERFACE (Real LLM)")
    print("   • Full Streamlit web interface")
    print("   • Real LLM responses (requires setup)")
    print("   • Complete workflow with actual AI")
    print("   • Requires: Ollama + API keys")
    print()
    
    print("3. 💻 COMMAND LINE (Real LLM)")
    print("   • Original command-line interface")
    print("   • Real LLM responses (requires setup)")
    print("   • Terminal-based interaction")
    print("   • Requires: Ollama + API keys")
    print()
    
    print("4. 🧪 TEST SYSTEMS")
    print("   • Test mock demo system")
    print("   • Test real system integration")
    print("   • Validate functionality")
    print()
    
    print("5. ❓ HELP & DOCUMENTATION")
    print("   • View documentation")
    print("   • Setup instructions")
    print("   • Feature comparisons")
    print()

def run_demo_mode():
    """Run the demo mode"""
    print("🎭 Starting Demo Mode...")
    print("✅ No LLM infrastructure required")
    print("🌐 Opening browser at http://localhost:8502")
    print()
    
    if not os.path.exists("streamlit_demo_app.py"):
        print("❌ Demo files not found!")
        return False
    
    try:
        subprocess.run([sys.executable, "run_demo.py"])
        return True
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False

def run_web_interface():
    """Run the real web interface"""
    print("🌐 Starting Web Interface (Real LLM)...")
    print("⚠️  Requires: Ollama running + API keys configured")
    print("🌐 Opening browser at http://localhost:8501")
    print()
    
    if not os.path.exists("streamlit_app.py"):
        print("❌ Web interface files not found!")
        return False
    
    try:
        subprocess.run([sys.executable, "run_streamlit.py"])
        return True
    except Exception as e:
        print(f"❌ Error running web interface: {e}")
        return False

def run_command_line():
    """Run the command line interface"""
    print("💻 Starting Command Line Interface...")
    print("⚠️  Requires: Ollama running + API keys configured")
    print()
    
    if not os.path.exists("main.py"):
        print("❌ Command line files not found!")
        return False
    
    try:
        subprocess.run([sys.executable, "main.py"])
        return True
    except Exception as e:
        print(f"❌ Error running command line: {e}")
        return False

def run_tests():
    """Run test systems"""
    print("🧪 Test Options:")
    print("1. Test Mock Demo System")
    print("2. Test Real System Integration")
    print("3. Back to main menu")
    print()
    
    choice = input("Choose test option (1-3): ").strip()
    
    if choice == "1":
        print("🧪 Testing Mock Demo System...")
        if os.path.exists("test_mock_core.py"):
            subprocess.run([sys.executable, "test_mock_core.py"])
        else:
            print("❌ Mock test file not found!")
    elif choice == "2":
        print("🧪 Testing Real System Integration...")
        if os.path.exists("test_streamlit_integration.py"):
            subprocess.run([sys.executable, "test_streamlit_integration.py"])
        else:
            print("❌ Integration test file not found!")
    elif choice == "3":
        return
    else:
        print("❌ Invalid option")

def show_help():
    """Show help and documentation"""
    print("📚 DOCUMENTATION & HELP:")
    print()
    
    docs = [
        ("README.md", "Main project documentation"),
        ("STREAMLIT_README.md", "Web interface guide"),
        ("DEMO_README.md", "Demo mode documentation"),
        ("requirements.txt", "Dependencies list")
    ]
    
    for doc, desc in docs:
        if os.path.exists(doc):
            print(f"✅ {doc} - {desc}")
        else:
            print(f"❌ {doc} - {desc} (missing)")
    
    print()
    print("🔧 SETUP REQUIREMENTS:")
    print()
    print("Demo Mode:")
    print("  pip install streamlit pandas pydantic")
    print()
    print("Real System:")
    print("  pip install -r requirements.txt")
    print("  # Set up Ollama")
    print("  # Configure API keys in config/settings.py")
    print()
    
    input("Press Enter to continue...")

def main():
    """Main launcher interface"""
    while True:
        print_header()
        print_options()
        
        choice = input("Enter your choice (1-5, or 'q' to quit): ").strip().lower()
        
        if choice in ['q', 'quit', 'exit']:
            print("👋 Goodbye!")
            break
        elif choice == '1':
            run_demo_mode()
        elif choice == '2':
            run_web_interface()
        elif choice == '3':
            run_command_line()
        elif choice == '4':
            run_tests()
        elif choice == '5':
            show_help()
        else:
            print("❌ Invalid choice. Please enter 1-5 or 'q' to quit.")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
