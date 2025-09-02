"""
Enhanced Streamlit App for Patent AI Agent
Uses the enhanced_mock_extractor.py with integrated Streamlit functionality
"""

import streamlit as st
import json
import datetime
import pandas as pd
from src.core.enhanced_mock_extractor import StreamlitEnhancedExtractor, STREAMLIT_AVAILABLE

def main():
    """Main Streamlit application for Enhanced Mock Patent Extractor"""
    
    if not STREAMLIT_AVAILABLE:
        st.error("❌ Streamlit dependencies are not available. Please install streamlit and pandas.")
        return
    
    # Page configuration
    st.set_page_config(
        page_title="Patent AI Agent - Enhanced Mock Extractor",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .demo-notice {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
            text-align: center;
        }
        .step-header {
            font-size: 1.5rem;
            color: #2e8b57;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .concept-box {
            background-color: #f0f8ff;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin: 1rem 0;
        }
        .keyword-box {
            background-color: #f5f5dc;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ffa500;
            margin: 1rem 0;
        }
        .progress-box {
            background-color: #e8f5e8;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #28a745;
            margin: 1rem 0;
        }
        .workflow-step {
            background-color: #f8f9fa;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            margin: 0.25rem 0;
            border-left: 3px solid #007bff;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🚀 Patent AI Agent - Enhanced Mock Extractor</div>', unsafe_allow_html=True)
    
    # Initialize persistent session state
    if 'enhanced_extractor' not in st.session_state:
        st.session_state.enhanced_extractor = None
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = None
    if 'extraction_completed' not in st.session_state:
        st.session_state.extraction_completed = False
    if 'saved_input_text' not in st.session_state:
        st.session_state.saved_input_text = ""
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "enhanced-mock-qwen2.5:3b"
    if 'use_checkpointer_flag' not in st.session_state:
        st.session_state.use_checkpointer_flag = False
    
    # Demo notice
    st.markdown('''
    <div class="demo-notice">
        <h3>🎭 ENHANCED MOCK MODE - Multi-Agent LangGraph Architecture</h3>
        <p>This is the enhanced version that maintains the exact multi-agent LangGraph architecture from the original extractor.py
        but uses mock LLM responses. All AI responses are pre-programmed to showcase the complete workflow and interface.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Model selection (mock)
        model_options = ["enhanced-mock-qwen2.5:3b", "enhanced-mock-llama3.2:3b", "enhanced-mock-phi3.5:3.8b"]
        selected_model = st.selectbox(
            "Select Enhanced Mock Model:",
            model_options,
            index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0,
            help="Choose the enhanced mock model (maintains LangGraph architecture)",
            key="enhanced_model_selector"
        )
        
        # Update session state when selection changes
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            use_checkpointer = st.checkbox(
                "Enable LangGraph Checkpointer",
                value=st.session_state.use_checkpointer_flag,
                help="Enable state checkpointing for the LangGraph workflow",
                key="enhanced_checkpointer_selector"
            )
            
            # Update session state when checkbox changes
            if use_checkpointer != st.session_state.use_checkpointer_flag:
                st.session_state.use_checkpointer_flag = use_checkpointer
            
            simulation_speed = st.slider(
                "Processing Simulation Speed",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.5,
                help="Adjust mock processing speed (1.0 = realistic timing)"
            )
        
        st.markdown("---")
        st.markdown("### 🏗️ LangGraph Architecture")
        st.markdown("""
        <div class="workflow-step">📥 <strong>Input Normalization</strong></div>
        <div class="workflow-step">🎯 <strong>Concept Extraction</strong></div>
        <div class="workflow-step">🔑 <strong>Keyword Generation</strong></div>
        <div class="workflow-step">👤 <strong>Human Evaluation</strong> ⭐</div>
        <div class="workflow-step">📋 <strong>Summary Generation</strong></div>
        <div class="workflow-step">🏷️ <strong>IPC Classification</strong></div>
        <div class="workflow-step">🔍 <strong>Synonym Generation</strong></div>
        <div class="workflow-step">🔍 <strong>Query Generation</strong></div>
        <div class="workflow-step">🌐 <strong>URL Discovery</strong></div>
        <div class="workflow-step">📊 <strong>Relevance Evaluation</strong></div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Key Features")
        st.markdown("""
        - **Multi-Agent Workflow**: Full LangGraph implementation
        - **Interactive Evaluation**: Human-in-the-loop decision making
        - **Mock Responses**: Consistent demo data
        - **State Management**: LangGraph state handling
        - **Export Results**: CSV & JSON downloads
        """)
    
    # Main content area
    st.markdown("## 📝 Patent Idea Input")
    
    # Enhanced sample text
    sample_text = """
    **Idea Title**: Smart Irrigation System with IoT Sensors and AI Control

    **User Scenario**: A precision agriculture company needs to develop an intelligent irrigation system 
    for large-scale farming operations. The system should monitor soil conditions across multiple field 
    zones, predict water requirements using weather data and crop growth models, and automatically 
    control irrigation equipment. The solution must integrate with existing farm management software 
    and provide real-time alerts to farm operators through mobile applications.

    **Technical Problem**: Current irrigation systems operate on fixed schedules without considering 
    real-time environmental conditions, soil moisture variations across different field areas, or 
    crop-specific water requirements at different growth stages. This leads to significant water waste, 
    uneven crop development, increased operational costs, and potential crop stress from over-watering 
    or under-watering. Farmers lack precise data-driven insights for irrigation decision-making.

    **Technical Solution**: Implement a distributed IoT sensor network with machine learning-based 
    irrigation control system. Deploy wireless soil moisture, temperature, and conductivity sensors 
    throughout the field in a mesh network topology. Integrate weather API services and satellite 
    imagery for crop health monitoring. Use edge computing devices to process sensor data locally 
    and machine learning algorithms to predict optimal irrigation timing and duration. Include 
    automated valve control systems, mobile app interface, and integration APIs for farm management 
    software platforms.
    """
    
    # Input text area - preserve value from session state if available
    if st.session_state.saved_input_text:
        input_text = st.text_area(
            "Enter your patent idea description:",
            value=st.session_state.saved_input_text,
            height=300,
            help="Describe your patent idea including the problem, solution, and technical details",
            key="enhanced_input_text_area"
        )
    else:
        input_text = st.text_area(
            "Enter your patent idea description:",
            value=sample_text,
            height=300,
            help="Describe your patent idea including the problem, solution, and technical details",
            key="enhanced_input_text_area"
        )
    
    # Update saved text when input changes
    if input_text != st.session_state.saved_input_text:
        st.session_state.saved_input_text = input_text
    
    # Processing button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Enhanced Extraction", type="primary", use_container_width=True):
            if input_text.strip():
                # Only clear workflow-specific state, preserve configuration
                for key in ['extraction_state','current_step','validation_feedback',
                            'show_reject_form','show_edit_form','awaiting_user_input','ui_interaction_id']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Initialize new extractor
                st.session_state.enhanced_extractor = StreamlitEnhancedExtractor(
                    model_name=st.session_state.selected_model,
                    use_checkpointer=st.session_state.use_checkpointer_flag
                )
                st.session_state.extraction_completed = False
                st.session_state.extraction_results = None
            else:
                st.warning("⚠️ Please enter a patent idea description to continue.")
    
    # Show progress and run extraction if extractor exists and not completed
    if st.session_state.enhanced_extractor and not st.session_state.extraction_completed:
        with st.spinner("🔄 Running enhanced extraction process with LangGraph..."):
            try:
                # Run extraction with UI evaluation
                results = st.session_state.enhanced_extractor.run_extraction_with_ui_evaluation(st.session_state.saved_input_text)
                
                if results:
                    st.session_state.extraction_results = results
                    st.session_state.extraction_completed = True
                    st.success("✅ Enhanced extraction completed successfully!")
                    
            except Exception as e:
                st.error(f"❌ Error during enhanced extraction: {str(e)}")
                import traceback
                st.error("Full traceback:")
                st.code(traceback.format_exc())
    
    # Display results if available
    if st.session_state.extraction_completed and st.session_state.extraction_results:
        results = st.session_state.extraction_results
        
        # Display results
        st.markdown("## 📊 Extraction Results")
        st.info("💡 **Note**: Results generated by enhanced mock system with full LangGraph architecture.")
                    
        # Results tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Summary", "🔑 Keywords", "🔍 Queries", "🔗 URLs", "📊 Analytics"])
        
        with tab1:
            st.markdown("### 📋 Concept Matrix")
            if results.get('concept_matrix'):
                concept_dict = results['concept_matrix'].dict()
                for field, value in concept_dict.items():
                    st.markdown(f"**{field.replace('_', ' ').title()}:**")
                    st.write(f"_{value}_")
                    st.markdown("---")
            
            st.markdown("### 📝 Technical Summary")
            if results.get('summary_text'):
                st.text_area("Generated Summary:", results['summary_text'], height=150, disabled=True)
            
            st.markdown("### 🏷️ IPC Classifications")
            if results.get('ipcs'):
                ipc_data = []
                for ipc in results['ipcs']:
                    ipc_data.append({
                        'IPC Category': ipc.get('category', 'N/A'),
                        'Confidence Score': f"{ipc.get('score', 0):.2f}",
                        'Percentage': f"{ipc.get('score', 0)*100:.1f}%"
                    })
                if ipc_data:
                    ipc_df = pd.DataFrame(ipc_data)
                    st.dataframe(ipc_df, use_container_width=True)
        
        with tab2:
            st.markdown("### 🌱 Seed Keywords")
            if results.get('seed_keywords'):
                keywords_dict = results['seed_keywords'].dict()
                for category, keywords in keywords_dict.items():
                    st.markdown(f"#### {category.replace('_', ' ').title()}")
                    for i, keyword in enumerate(keywords, 1):
                        st.write(f"{i}. `{keyword}`")
                    st.markdown("---")
            
            st.markdown("### 🔍 Expanded Keywords & Synonyms")
            if results.get('final_keywords'):
                for original_keyword, synonyms in results['final_keywords'].items():
                    with st.expander(f"🔍 {original_keyword}"):
                        st.markdown("**Synonyms & Related Terms:**")
                        for i, synonym in enumerate(synonyms, 1):
                            st.write(f"{i}. {synonym}")
        
        with tab3:
            st.markdown("### 🔍 Generated Search Queries")
            if results.get('queries') and hasattr(results['queries'], 'queries'):
                for i, query in enumerate(results['queries'].queries, 1):
                    st.markdown(f"#### Query {i}")
                    st.code(query, language="text")
                    
            st.info("💡 These Boolean queries can be used in patent databases like Google Patents, USPTO, or EPO.")
        
        with tab4:
            st.markdown("### 🔗 Patent URLs Found")
            if results.get('final_url'):
                url_data = []
                for url_info in results['final_url']:
                    if isinstance(url_info, dict):
                        url_data.append({
                            'Patent URL': url_info.get('url', 'N/A'),
                            'Scenario Score': f"{url_info.get('user_scenario', 0):.3f}",
                            'Problem Score': f"{url_info.get('user_problem', 0):.3f}",
                            'Overall Relevance': f"{(url_info.get('user_scenario', 0) + url_info.get('user_problem', 0))/2:.3f}"
                        })
                
                if url_data:
                    urls_df = pd.DataFrame(url_data)
                    st.dataframe(urls_df, use_container_width=True)
                    
                    # Download button for URLs
                    csv = urls_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Patent URLs as CSV",
                        csv,
                        f"enhanced_patent_urls_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                    
            st.warning("⚠️ **Mock Data**: These are simulated patent URLs for demonstration purposes.")
        
        with tab5:
            st.markdown("### 📊 Extraction Analytics")
            
            # Create analytics dashboard
            col1, col2 = st.columns(2)
            
            with col1:
                if results.get('seed_keywords'):
                    keywords_dict = results['seed_keywords'].dict()
                    keyword_counts = {k.replace('_', ' ').title(): len(v) for k, v in keywords_dict.items()}
                    
                    st.markdown("#### Keywords by Category")
                    chart_data = pd.DataFrame(list(keyword_counts.items()), columns=['Category', 'Count'])
                    st.bar_chart(chart_data.set_index('Category'))
            
            with col2:
                if results.get('ipcs'):
                    st.markdown("#### IPC Classification Scores")
                    ipc_chart_data = pd.DataFrame(results['ipcs'])
                    if not ipc_chart_data.empty:
                        st.bar_chart(ipc_chart_data.set_index('category')['score'])
            
            # Summary statistics
            st.markdown("### 📈 Summary Statistics")
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            with stats_col1:
                total_keywords = sum(len(v) for v in results.get('seed_keywords', {}).dict().values()) if results.get('seed_keywords') else 0
                st.metric("Total Keywords", total_keywords)
            
            with stats_col2:
                total_synonyms = len(results.get('final_keywords', {}))
                st.metric("Expanded Terms", total_synonyms)
            
            with stats_col3:
                total_queries = len(results.get('queries', {}).queries) if results.get('queries') and hasattr(results['queries'], 'queries') else 0
                st.metric("Search Queries", total_queries)
            
            with stats_col4:
                total_urls = len(results.get('final_url', []))
                st.metric("Patent URLs", total_urls)
        
        # Download complete results
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Prepare results for download
            download_data = {}
            for key, value in results.items():
                if value is None:
                    continue
                if hasattr(value, "dict"):
                    download_data[key] = value.dict()
                elif isinstance(value, (dict, list, str, int, float, bool)):
                    download_data[key] = value
                else:
                    download_data[key] = str(value)
            
            # Add metadata
            download_data["_extraction_metadata"] = {
                "enhanced_mock_mode": True,
                "langgraph_architecture": True,
                "multi_agent_workflow": True,
                "generated_at": datetime.datetime.now().isoformat(),
                "model": st.session_state.get('selected_model', 'enhanced-mock'),
                "checkpointer_enabled": st.session_state.get('use_checkpointer_flag', False)
            }
            
            json_str = json.dumps(download_data, indent=2, ensure_ascii=False)
            filename = f"enhanced_extraction_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            st.download_button(
                "💾 Download Complete Results (JSON)",
                json_str,
                filename,
                "application/json",
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🚀 Patent AI Agent - Enhanced Mock Extractor with LangGraph<br>
        Multi-agent workflow architecture with mock LLM responses for demonstration<br>
        Built with LangGraph, Streamlit & Enhanced Mock AI System
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
