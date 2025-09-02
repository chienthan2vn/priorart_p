#!/usr/bin/env python3
"""
Test script for the core Mock Patent AI Agent functionality
Tests only the mock extractor without Streamlit dependencies
"""

import sys
import os
import json

def test_mock_imports():
    """Test that mock components can be imported"""
    print("🧪 Testing mock system imports...")
    
    try:
        from src.core.mock_extractor import MockCoreConceptExtractor, ValidationFeedback, SeedKeywords, MockLLM
        print("✅ Mock extractor imports successful")
        return True
    except Exception as e:
        print(f"❌ Mock extractor import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

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
                
                # Try to parse JSON responses
                try:
                    cleaned_response = response.strip()
                    json.loads(cleaned_response)
                    print(f"  ✅ Response is valid JSON")
                except:
                    print(f"  ⚠️  Response is not JSON (might be intentional)")
            else:
                print(f"❌ {prompt_type.title()} response too short or empty")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Mock LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation_models():
    """Test validation models work correctly"""
    print("\n🧪 Testing validation models...")
    
    try:
        from src.core.mock_extractor import ValidationFeedback, SeedKeywords
        
        # Test ValidationFeedback
        feedback1 = ValidationFeedback(action="approve")
        print(f"✅ Approve feedback: {feedback1.action}")
        
        feedback2 = ValidationFeedback(action="reject", feedback="Test feedback")
        print(f"✅ Reject feedback: {feedback2.action}, {feedback2.feedback}")
        
        # Test SeedKeywords
        keywords = SeedKeywords(
            problem_purpose=["water", "optimization"],
            object_system=["IoT", "sensors"], 
            environment_field=["agriculture", "farming"]
        )
        print(f"✅ SeedKeywords created with {len(keywords.problem_purpose)} problem keywords")
        
        feedback3 = ValidationFeedback(action="edit", edited_keywords=keywords)
        print(f"✅ Edit feedback with {len(feedback3.edited_keywords.object_system)} object keywords")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mock_extractor():
    """Test that mock extractor can run a complete workflow"""
    print("\n🧪 Testing mock extractor workflow...")
    
    try:
        from src.core.mock_extractor import MockCoreConceptExtractor, ValidationFeedback
        
        # Create extractor with auto-approval handler
        def auto_approve_handler(state):
            print("  🤖 Auto-approval handler called")
            return {"validation_feedback": ValidationFeedback(action="approve")}
        
        extractor = MockCoreConceptExtractor(custom_evaluation_handler=auto_approve_handler)
        print("✅ Mock extractor created")
        
        # Test input
        test_input = """
        Smart Irrigation System with IoT Sensors for precision agriculture.
        Problem: Traditional irrigation wastes water and lacks real-time monitoring.
        Solution: IoT sensors monitor soil moisture and control irrigation automatically.
        Technical approach: Wireless sensor network with machine learning algorithms.
        """
        
        print("🔄 Running mock extraction workflow...")
        results = extractor.extract_keywords(test_input)
        print("✅ Workflow completed")
        
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
            print(f"✅ Concept matrix: {results['concept_matrix'].problem_purpose[:50]}...")
        else:
            print("❌ Concept matrix structure invalid")
            return False
            
        if hasattr(results['seed_keywords'], 'problem_purpose'):
            keywords = results['seed_keywords'].problem_purpose
            print(f"✅ Seed keywords ({len(keywords)}): {keywords}")
        else:
            print("❌ Seed keywords structure invalid")
            return False
        
        if isinstance(results['final_keywords'], dict) and len(results['final_keywords']) > 0:
            print(f"✅ Final keywords generated for {len(results['final_keywords'])} terms")
        else:
            print("❌ Final keywords not properly generated")
            return False
            
        if hasattr(results['queries'], 'queries') and len(results['queries'].queries) > 0:
            print(f"✅ {len(results['queries'].queries)} search queries generated")
        else:
            print("❌ Search queries not properly generated")
            return False
            
        if isinstance(results['final_url'], list) and len(results['final_url']) > 0:
            print(f"✅ {len(results['final_url'])} patent URLs found")
        else:
            print("❌ Patent URLs not properly generated")
            return False
        
        print("✅ Mock extractor workflow completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Mock extractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rejection_workflow():
    """Test rejection and retry workflow"""
    print("\n🧪 Testing rejection workflow...")
    
    try:
        from src.core.mock_extractor import MockCoreConceptExtractor, ValidationFeedback
        
        call_count = 0
        
        def rejection_handler(state):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                print("  🚫 First call - rejecting")
                return {"validation_feedback": ValidationFeedback(action="reject", feedback="Test rejection")}
            else:
                print("  ✅ Second call - approving")
                return {"validation_feedback": ValidationFeedback(action="approve")}
        
        extractor = MockCoreConceptExtractor(custom_evaluation_handler=rejection_handler)
        
        test_input = "Test patent idea for rejection workflow"
        
        print("🔄 Running rejection workflow test...")
        results = extractor.extract_keywords(test_input)
        
        if call_count == 2:
            print("✅ Rejection workflow worked - handler called twice")
            return True
        else:
            print(f"❌ Rejection workflow failed - handler called {call_count} times")
            return False
        
    except Exception as e:
        print(f"❌ Rejection workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all core mock system tests"""
    print("🎭 Patent AI Agent - Core Mock System Test")
    print("=" * 60)
    print("🧪 Testing without Streamlit dependencies")
    print("=" * 60)
    
    tests = [
        test_mock_imports,
        test_validation_models,
        test_mock_llm,
        test_mock_extractor,
        test_rejection_workflow
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
    print(f"📊 Core Mock System Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All core tests passed! Mock system is working correctly.")
        print("\n🎭 Core mock functionality validated:")
        print("   ✅ Mock LLM responses")
        print("   ✅ Complete extraction workflow")
        print("   ✅ Human evaluation simulation") 
        print("   ✅ Rejection/retry logic")
        print("   ✅ Data model validation")
        print("\n🚀 To run the full demo (requires Streamlit):")
        print("   pip install streamlit pandas")
        print("   python run_demo.py")
    else:
        print("❌ Some core tests failed. Please fix the issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()
