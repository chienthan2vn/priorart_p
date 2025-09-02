"""
Test script for Enhanced Mock Extractor with LangGraph Framework
Verifies the complete multi-agent LangGraph workflow
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_langgraph_workflow():
    """Test the complete LangGraph multi-agent workflow"""
    print("🧪 Testing Enhanced Mock Extractor with LangGraph Framework...")
    
    try:
        from src.core.enhanced_mock_extractor import EnhancedMockCoreConceptExtractor, ValidationFeedback, SeedKeywords
        print("✅ Successfully imported EnhancedMockCoreConceptExtractor")
    except ImportError as e:
        print(f"❌ Failed to import enhanced mock extractor: {e}")
        print("💡 Make sure you have installed the required dependencies:")
        print("   pip install langgraph pydantic")
        return False
    
    # Create extractor instance with LangGraph
    print("\n🔧 Creating Enhanced Mock Extractor with LangGraph...")
    extractor = EnhancedMockCoreConceptExtractor(
        model_name="mock-llm-langgraph",
        use_checkpointer=False  # Set to True to test checkpointing
    )
    
    # Verify LangGraph components
    print("📊 Verifying LangGraph components...")
    if hasattr(extractor, 'graph'):
        print("  ✅ LangGraph StateGraph exists")
        print(f"  📋 Graph type: {type(extractor.graph)}")
    else:
        print("  ❌ LangGraph StateGraph missing")
        return False
    
    # Test input
    test_input = """
    I want to develop a smart home automation system that uses AI and IoT sensors 
    to automatically control lighting, temperature, and security based on occupancy 
    patterns and user preferences. The system should learn from user behavior and 
    optimize energy consumption while maintaining comfort and security.
    """
    
    print(f"\n📝 Input text: {test_input[:100]}...")
    
    try:
        print("\n🔄 Running LangGraph multi-agent workflow...")
        print("=" * 60)
        
        # Run extraction through LangGraph
        results = extractor.extract_keywords(test_input)
        
        print("=" * 60)
        print("✅ LangGraph workflow completed successfully!")
        
        # Verify results structure
        expected_keys = [
            'input_text', 'problem', 'technical', 'summary_text', 'ipcs',
            'concept_matrix', 'seed_keywords', 'validation_feedback', 
            'final_keywords', 'queries', 'final_url'
        ]
        
        print("\n📊 Checking LangGraph results structure...")
        missing_keys = []
        for key in expected_keys:
            if key in results:
                print(f"  ✅ {key}: {type(results[key])}")
            else:
                print(f"  ❌ Missing key: {key}")
                missing_keys.append(key)
        
        if missing_keys:
            print(f"\n⚠️ Missing keys: {missing_keys}")
            return False
        
        # Display key results from LangGraph workflow
        print("\n🎯 LangGraph Multi-Agent Results:")
        
        if results.get('concept_matrix'):
            cm = results['concept_matrix']
            print(f"  🎯 Problem/Purpose: {cm.problem_purpose[:100]}...")
            print(f"  🔧 Object/System: {cm.object_system[:100]}...")
            print(f"  🌍 Environment/Field: {cm.environment_field[:100]}...")
        
        if results.get('seed_keywords'):
            sk = results['seed_keywords']
            print(f"  🔑 Problem Keywords: {sk.problem_purpose}")
            print(f"  🔑 Object Keywords: {sk.object_system}")
            print(f"  🔑 Environment Keywords: {sk.environment_field}")
        
        if results.get('final_keywords'):
            print(f"  🔍 Final Keywords: {len(results['final_keywords'])} categories")
            for key, synonyms in list(results['final_keywords'].items())[:3]:  # Show first 3
                print(f"    • {key}: {synonyms}")
        
        if results.get('queries'):
            print(f"  🔍 Generated Queries: {len(results['queries'].queries)} queries")
            for i, query in enumerate(results['queries'].queries[:2], 1):  # Show first 2
                print(f"    {i}. {query}")
        
        if results.get('final_url'):
            print(f"  🔗 Found URLs: {len(results['final_url'])} patents")
            for i, url_info in enumerate(results['final_url'][:2], 1):  # Show first 2
                print(f"    {i}. {url_info['url']} (scores: {url_info['user_scenario']:.3f}, {url_info['user_problem']:.3f})")
        
        print("\n🎉 Enhanced Mock Extractor with LangGraph test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ LangGraph workflow failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_langgraph_with_checkpointing():
    """Test LangGraph workflow with checkpointing enabled"""
    print("\n🧪 Testing LangGraph with Checkpointing...")
    
    try:
        from src.core.enhanced_mock_extractor import EnhancedMockCoreConceptExtractor
        
        # Create extractor with checkpointing enabled
        extractor = EnhancedMockCoreConceptExtractor(
            model_name="mock-llm-checkpoint",
            use_checkpointer=True  # Enable checkpointing
        )
        
        test_input = "Smart IoT home automation with AI learning capabilities"
        
        print("🔄 Running LangGraph workflow with checkpointing...")
        results = extractor.extract_keywords(test_input)
        
        if results and len(results) > 5:  # Basic check for results
            print("✅ LangGraph checkpointing test passed!")
            return True
        else:
            print("❌ LangGraph checkpointing test failed - insufficient results")
            return False
            
    except Exception as e:
        print(f"❌ LangGraph checkpointing test failed: {str(e)}")
        return False

def test_custom_evaluation_handler():
    """Test LangGraph with custom evaluation handler"""
    print("\n🧪 Testing LangGraph with Custom Evaluation Handler...")
    
    try:
        from src.core.enhanced_mock_extractor import EnhancedMockCoreConceptExtractor, ValidationFeedback, SeedKeywords
        
        # Custom evaluation handler for testing
        def mock_ui_evaluation(state):
            """Mock UI evaluation handler that simulates user interaction"""
            print("  📱 Custom LangGraph evaluation handler called")
            print(f"  📊 LangGraph state keys: {list(state.keys())}")
            
            # Simulate user editing keywords
            if state.get('seed_keywords'):
                edited_keywords = SeedKeywords(
                    problem_purpose=["smart_automation", "ai_learning"],
                    object_system=["iot_sensors", "home_system"],
                    environment_field=["smart_home", "residential"]
                )
                return {"validation_feedback": ValidationFeedback(action="edit", edited_keywords=edited_keywords)}
            
            return {"validation_feedback": ValidationFeedback(action="approve")}
        
        # Create extractor with custom handler
        extractor = EnhancedMockCoreConceptExtractor(
            custom_evaluation_handler=mock_ui_evaluation
        )
        
        test_input = "AI-powered smart home automation system"
        
        print("🔄 Running LangGraph with custom evaluation handler...")
        results = extractor.extract_keywords(test_input)
        
        # Check if custom evaluation was used and edited keywords were applied
        if (results.get('validation_feedback') and 
            results['validation_feedback'].action == "edit" and
            results.get('seed_keywords') and
            "smart_automation" in results['seed_keywords'].problem_purpose):
            
            print("✅ LangGraph custom evaluation handler test passed!")
            print("✅ Edited keywords were correctly applied in LangGraph workflow")
            return True
        else:
            print("❌ LangGraph custom evaluation handler test failed")
            return False
            
    except Exception as e:
        print(f"❌ LangGraph custom evaluation test failed: {str(e)}")
        return False

def test_langgraph_architecture():
    """Test that LangGraph architecture matches original extractor.py"""
    print("\n🧪 Testing LangGraph Architecture Consistency...")
    
    try:
        from src.core.enhanced_mock_extractor import EnhancedMockCoreConceptExtractor
        
        extractor = EnhancedMockCoreConceptExtractor()
        
        # Check LangGraph-specific components
        langgraph_components = ['graph']
        for component in langgraph_components:
            if hasattr(extractor, component):
                print(f"  ✅ LangGraph {component} exists")
            else:
                print(f"  ❌ Missing LangGraph {component}")
                return False
        
        # Check that all workflow methods exist (should be LangGraph nodes)
        workflow_methods = [
            'input_normalization', 'step0', 'step1_concept_extraction',
            'step2_keyword_generation', 'step3_human_evaluation', 
            'manual_editing', 'gen_key', 'summary_prompt_and_parser',
            'call_ipcs_api', 'genQuery', 'genUrl', 'evalUrl'
        ]
        
        for method in workflow_methods:
            if hasattr(extractor, method) and callable(getattr(extractor, method)):
                print(f"  ✅ LangGraph node method: {method}")
            else:
                print(f"  ❌ Missing LangGraph node method: {method}")
                return False
        
        # Check LangGraph-specific methods
        langgraph_methods = ['_build_graph', '_get_human_action']
        for method in langgraph_methods:
            if hasattr(extractor, method) and callable(getattr(extractor, method)):
                print(f"  ✅ LangGraph method: {method}")
            else:
                print(f"  ❌ Missing LangGraph method: {method}")
                return False
        
        print("✅ LangGraph architecture consistency test passed!")
        return True
        
    except Exception as e:
        print(f"❌ LangGraph architecture test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Running Enhanced Mock Extractor LangGraph Tests\n")
    
    # Run all tests
    basic_test = test_langgraph_workflow()
    checkpoint_test = test_langgraph_with_checkpointing()
    handler_test = test_custom_evaluation_handler()
    architecture_test = test_langgraph_architecture()
    
    print(f"\n📊 LangGraph Test Summary:")
    print(f"  Basic LangGraph Workflow: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"  LangGraph Checkpointing: {'✅ PASS' if checkpoint_test else '❌ FAIL'}")
    print(f"  Custom Handler Integration: {'✅ PASS' if handler_test else '❌ FAIL'}")
    print(f"  LangGraph Architecture: {'✅ PASS' if architecture_test else '❌ FAIL'}")
    
    if all([basic_test, checkpoint_test, handler_test, architecture_test]):
        print("\n🎉 All LangGraph tests passed! Enhanced Mock Extractor is ready for use.")
        print("💡 The multi-agent LangGraph architecture is working correctly with mock data.")
        sys.exit(0)
    else:
        print("\n⚠️ Some LangGraph tests failed. Please check the implementation.")
        sys.exit(1)
