"""
Test script for Standalone Mock Extractor
Tests the complete workflow without external dependencies
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.standalone_mock_extractor import StandaloneMockCoreConceptExtractor, ValidationFeedback, SeedKeywords

def test_complete_workflow():
    """Test the complete extraction workflow"""
    print("🧪 Testing Standalone Mock Extractor Complete Workflow...")
    
    # Create extractor instance
    extractor = StandaloneMockCoreConceptExtractor(
        model_name="mock-llm",
        use_checkpointer=False
    )
    
    # Test input
    test_input = """
    I want to create a smart irrigation system that uses IoT sensors to monitor soil moisture 
    and automatically controls water distribution based on real-time data and weather conditions.
    The system should optimize water usage while ensuring crops get adequate moisture.
    """
    
    print(f"📝 Input text: {test_input[:100]}...")
    
    try:
        # Run extraction
        print("\n🔄 Running extraction workflow...")
        results = extractor.extract_keywords(test_input)
        
        # Verify results structure
        expected_keys = [
            'input_text', 'problem', 'technical', 'summary_text', 'ipcs',
            'concept_matrix', 'seed_keywords', 'validation_feedback', 
            'final_keywords', 'queries', 'final_url'
        ]
        
        print("\n📊 Checking results structure...")
        for key in expected_keys:
            if key in results:
                print(f"  ✅ {key}: {type(results[key])}")
            else:
                print(f"  ❌ Missing key: {key}")
        
        # Display key results
        print("\n🎯 Key Results:")
        if results.get('concept_matrix'):
            cm = results['concept_matrix']
            print(f"  Problem/Purpose: {cm.problem_purpose}")
            print(f"  Object/System: {cm.object_system}")
            print(f"  Environment/Field: {cm.environment_field}")
        
        if results.get('seed_keywords'):
            sk = results['seed_keywords']
            print(f"  Problem Keywords: {sk.problem_purpose}")
            print(f"  Object Keywords: {sk.object_system}")
            print(f"  Environment Keywords: {sk.environment_field}")
        
        if results.get('final_keywords'):
            print(f"  Final Keywords: {len(results['final_keywords'])} categories")
            for key, synonyms in results['final_keywords'].items():
                print(f"    {key}: {synonyms}")
        
        if results.get('queries'):
            print(f"  Generated Queries: {len(results['queries'].queries)} queries")
            for i, query in enumerate(results['queries'].queries[:3], 1):
                print(f"    {i}. {query}")
        
        if results.get('final_url'):
            print(f"  Found URLs: {len(results['final_url'])} patents")
            for i, url_info in enumerate(results['final_url'][:3], 1):
                print(f"    {i}. {url_info['url']} (scores: {url_info['user_scenario']:.3f}, {url_info['user_problem']:.3f})")
        
        print("\n✅ Standalone Mock Extractor workflow test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_evaluation_handler():
    """Test custom evaluation handler functionality"""
    print("\n🧪 Testing Custom Evaluation Handler...")
    
    def mock_ui_evaluation(state):
        """Mock UI evaluation handler"""
        print("  📱 Custom evaluation handler called")
        print(f"  📊 State keys: {list(state.keys())}")
        
        # Simulate user editing keywords
        if state.get('seed_keywords'):
            edited_keywords = SeedKeywords(
                problem_purpose=["custom_problem", "edited_purpose"],
                object_system=["custom_system", "edited_object"],
                environment_field=["custom_field", "edited_environment"]
            )
            return {"validation_feedback": ValidationFeedback(action="edit", edited_keywords=edited_keywords)}
        
        return {"validation_feedback": ValidationFeedback(action="approve")}
    
    # Create extractor with custom handler
    extractor = StandaloneMockCoreConceptExtractor(
        custom_evaluation_handler=mock_ui_evaluation
    )
    
    test_input = "Smart irrigation system with IoT sensors"
    
    try:
        results = extractor.extract_keywords(test_input)
        
        # Check if custom evaluation was used
        if results.get('validation_feedback') and results['validation_feedback'].action == "edit":
            print("  ✅ Custom evaluation handler worked correctly")
            
            # Check if edited keywords were applied
            if results.get('seed_keywords'):
                sk = results['seed_keywords']
                if "custom_problem" in sk.problem_purpose:
                    print("  ✅ Edited keywords were applied correctly")
                    return True
                else:
                    print("  ❌ Edited keywords were not applied")
                    return False
            else:
                print("  ❌ No seed keywords found after editing")
                return False
        else:
            print("  ❌ Custom evaluation handler not used properly")
            return False
            
    except Exception as e:
        print(f"  ❌ Custom evaluation test failed: {str(e)}")
        return False

def test_architecture_consistency():
    """Test that the architecture follows the original extractor pattern"""
    print("\n🧪 Testing Architecture Consistency...")
    
    try:
        extractor = StandaloneMockCoreConceptExtractor()
        
        # Check that all required components exist
        components = ['llm', 'tavily_search', 'prompts', 'messages', 'validation_messages']
        for component in components:
            if hasattr(extractor, component):
                print(f"  ✅ {component} component exists")
            else:
                print(f"  ❌ Missing {component} component")
                return False
        
        # Check that key methods exist
        methods = [
            'extract_keywords', 'input_normalization', 'step0', 
            'step1_concept_extraction', 'step2_keyword_generation', 
            'step3_human_evaluation', 'manual_editing', 'gen_key',
            'summary_prompt_and_parser', 'call_ipcs_api', 
            'genQuery', 'genUrl', 'evalUrl'
        ]
        
        for method in methods:
            if hasattr(extractor, method) and callable(getattr(extractor, method)):
                print(f"  ✅ {method} method exists")
            else:
                print(f"  ❌ Missing {method} method")
                return False
        
        print("  ✅ All architecture components are present")
        return True
        
    except Exception as e:
        print(f"  ❌ Architecture test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Running Standalone Mock Extractor Tests\n")
    
    # Run tests
    workflow_test = test_complete_workflow()
    handler_test = test_custom_evaluation_handler()
    architecture_test = test_architecture_consistency()
    
    print(f"\n📊 Test Summary:")
    print(f"  Complete Workflow: {'✅ PASS' if workflow_test else '❌ FAIL'}")
    print(f"  Custom Handler: {'✅ PASS' if handler_test else '❌ FAIL'}")
    print(f"  Architecture: {'✅ PASS' if architecture_test else '❌ FAIL'}")
    
    if workflow_test and handler_test and architecture_test:
        print("\n🎉 All tests passed! Standalone Mock Extractor is working correctly.")
        print("💡 This version maintains the exact multi-agent architecture from the original extractor.")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")
        sys.exit(1)
