#!/usr/bin/env python3
"""
Test script to validate Streamlit integration with the Patent AI Agent
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from src.core.extractor import CoreConceptExtractor, ValidationFeedback, SeedKeywords
        print("✅ Core extractor imports successful")
    except Exception as e:
        print(f"❌ Core extractor import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print(f"✅ Streamlit import successful (version: {st.__version__})")
    except Exception as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from config.settings import settings
        print("✅ Settings import successful")
    except Exception as e:
        print(f"❌ Settings import failed: {e}")
        return False
    
    return True

def test_extractor_initialization():
    """Test that the extractor can be initialized with custom handler"""
    print("\n🧪 Testing extractor initialization...")
    
    try:
        from src.core.extractor import CoreConceptExtractor
        
        # Test without custom handler
        extractor1 = CoreConceptExtractor()
        print("✅ Basic extractor initialization successful")
        
        # Test with custom handler
        def dummy_handler(state):
            return {"validation_feedback": None}
        
        extractor2 = CoreConceptExtractor(custom_evaluation_handler=dummy_handler)
        print("✅ Custom handler extractor initialization successful")
        
        # Verify the handler is set
        if extractor2.custom_evaluation_handler == dummy_handler:
            print("✅ Custom handler properly assigned")
        else:
            print("❌ Custom handler not properly assigned")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Extractor initialization failed: {e}")
        return False

def test_validation_models():
    """Test that validation models can be created"""
    print("\n🧪 Testing validation models...")
    
    try:
        from src.core.extractor import ValidationFeedback, SeedKeywords
        
        # Test ValidationFeedback creation
        feedback1 = ValidationFeedback(action="approve")
        print("✅ Approve feedback model created")
        
        feedback2 = ValidationFeedback(action="reject", feedback="Test feedback")
        print("✅ Reject feedback model created")
        
        # Test SeedKeywords creation
        keywords = SeedKeywords(
            problem_purpose=["test", "keywords"],
            object_system=["system", "device"],
            environment_field=["agriculture", "IoT"]
        )
        print("✅ SeedKeywords model created")
        
        feedback3 = ValidationFeedback(action="edit", edited_keywords=keywords)
        print("✅ Edit feedback model created")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation models test failed: {e}")
        return False

def test_streamlit_app_syntax():
    """Test that the Streamlit app has valid syntax"""
    print("\n🧪 Testing Streamlit app syntax...")
    
    try:
        # Try to compile the streamlit app
        with open('streamlit_app.py', 'r') as f:
            code = f.read()
        
        compile(code, 'streamlit_app.py', 'exec')
        print("✅ Streamlit app syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Streamlit app syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Patent AI Agent - Streamlit Integration Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_extractor_initialization,
        test_validation_models,
        test_streamlit_app_syntax
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print("❌ Test failed, stopping...")
            break
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Streamlit integration is ready.")
        print("\n🚀 To run the Streamlit app:")
        print("   python run_streamlit.py")
        print("   or")
        print("   streamlit run streamlit_app.py")
    else:
        print("❌ Some tests failed. Please fix the issues before running.")
        sys.exit(1)

if __name__ == "__main__":
    main()
