#!/usr/bin/env python3
"""
Test script for the Mock Patent AI Agent Demo
Validates that the mock system works without requiring LLM infrastructure
"""

import sys
import os
import json

def test_mock_imports():
    """Test that mock components can be imported"""
    print("🧪 Testing mock system imports...")
    
    try:
        from src.core.mock_extractor import MockCoreConceptExtractor, ValidationFeedback, SeedKeywords
        print("✅ Mock extractor imports successful")
    except Exception as e:
        print(f"❌ Mock extractor import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print(f"✅ Streamlit import successful (version: {st.__version__})")
    except Exception as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas import successful")
    except Exception as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    return True

def test_mock_llm():
    """Test that mock LLM produces responses"""
    print("\n🧪 Testing mock LLM responses...")
    
    try:
        from src.core.mock_extractor import MockLLM
        
        llm = MockLLM()
        
        # Test different types of prompts
        test_prompts = [
            ("normalization", "normalization prompt test"),
            ("concept matrix", "concept matrix prompt test"), 
            ("keywords", "seed keywords prompt test"),
            ("summary", "summary prompt test"),
            ("queries", "queries prompt test"),
            ("synonyms", "synonyms prompt test")
        ]
        
        for prompt_type, prompt in test_prompts:
            response = llm.invoke(prompt)
            if response and len(response) > 10:
                print(f"✅ {prompt_type.title()} response generated ({len(response)} chars)")
            else:
                print(f"❌ {prompt_type.title()} response too short or empty")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Mock LLM test failed: {e}")
        return False

def test_mock_extractor():
    """Test that mock extractor can run a complete workflow"""
    print("\n🧪 Testing mock extractor workflow...")
    
    try:
        from src.core.mock_extractor import MockCoreConceptExtractor, ValidationFeedback
        
        # Create extractor with auto-approval handler
        def auto_approve_handler(state):
            return {"validation_feedback": ValidationFeedback(action="approve")}
        
        extractor = MockCoreConceptExtractor(custom_evaluation_handler=auto_approve_handler)
        
        # Test input
        test_input = """
        Smart Irrigation System with IoT Sensors for precision agriculture.
        Problem: Traditional irrigation wastes water.
        Solution: IoT sensors monitor soil moisture and control irrigation automatically.
        """
        
        print("🔄 Running mock extraction workflow...")
        results = extractor.extract_keywords(test_input)
        
        # Validate results structure
        required_keys = ['concept_matrix', 'seed_keywords', 'final_keywords', 'queries', 'final_url']
        for key in required_keys:
            if key in results:
                print(f"✅ {key} present in results")
            else:
                print(f"❌ {key} missing from results")
                return False
        
        # Test specific result content
        if hasattr(results['concept_matrix'], 'problem_purpose'):
            print("✅ Concept matrix has proper structure")
        else:
            print("❌ Concept matrix structure invalid")
            return False
            
        if hasattr(results['seed_keywords'], 'problem_purpose'):
            print("✅ Seed keywords have proper structure")
        else:
            print("❌ Seed keywords structure invalid")
            return False
        
        if isinstance(results['final_keywords'], dict) and len(results['final_keywords']) > 0:
            print("✅ Final keywords generated")
        else:
            print("❌ Final keywords not properly generated")
            return False
            
        print("✅ Mock extractor workflow completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Mock extractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_demo_app_syntax():
    """Test that the demo app has valid syntax"""
    print("\n🧪 Testing demo app syntax...")
    
    try:
        # Try to compile the demo app
        with open('streamlit_demo_app.py', 'r') as f:
            code = f.read()
        
        compile(code, 'streamlit_demo_app.py', 'exec')
        print("✅ Demo app syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Demo app syntax error: {e}")
        return False
    except FileNotFoundError:
        print("❌ Demo app file not found")
        return False
    except Exception as e:
        print(f"❌ Demo app test failed: {e}")
        return False

def test_validation_models():
    """Test validation models work correctly"""
    print("\n🧪 Testing validation models...")
    
    try:
        from src.core.mock_extractor import ValidationFeedback, SeedKeywords
        
        # Test ValidationFeedback
        feedback1 = ValidationFeedback(action="approve")
        feedback2 = ValidationFeedback(action="reject", feedback="Test feedback")
        
        # Test SeedKeywords
        keywords = SeedKeywords(
            problem_purpose=["water", "optimization"],
            object_system=["IoT", "sensors"], 
            environment_field=["agriculture", "farming"]
        )
        
        feedback3 = ValidationFeedback(action="edit", edited_keywords=keywords)
        
        print("✅ All validation models created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Validation models test failed: {e}")
        return False

def main():
    """Run all mock system tests"""
    print("🎭 Patent AI Agent - Mock Demo System Test")
    print("=" * 60)
    
    tests = [
        test_mock_imports,
        test_validation_models,
        test_mock_llm,
        test_mock_extractor,
        test_demo_app_syntax
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
    print(f"📊 Mock System Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Mock demo system is ready.")
        print("\n🎭 To run the mock demo:")
        print("   python run_demo.py")
        print("   or")
        print("   streamlit run streamlit_demo_app.py --server.port=8502")
        print("\n🌟 The demo will work without any LLM infrastructure!")
        print("   • Mock AI responses")
        print("   • Full interactive workflow")
        print("   • Complete results export")
    else:
        print("❌ Some tests failed. Please fix the issues before running the demo.")
        sys.exit(1)

if __name__ == "__main__":
    main()
