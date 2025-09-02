"""
Streamlit Demo Web Interface for Patent AI Agent
Demo version using mock LLM responses - no actual LLM infrastructure required
"""

import streamlit as st
import json
import datetime
import logging
import traceback
from typing import Dict, Any, Optional
import pandas as pd

# Import the enhanced mock extractor with LangGraph framework
from src.core.enhanced_mock_extractor import EnhancedMockCoreConceptExtractor, ValidationFeedback, SeedKeywords, ExtractionState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Patent AI Agent - Demo (Mock LLM)",
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
</style>
""", unsafe_allow_html=True)

class StreamlitDemoExtractor:
    """Demo version of Streamlit Patent Extractor using mock responses"""
    
    def __init__(self, model_name: str = None, use_checkpointer: bool = None):
        # Create enhanced mock extractor with LangGraph multi-agent architecture
        logger.info(f"Creating demo extractor with model: {model_name} and use_checkpointer: {use_checkpointer}")
        self.extractor = EnhancedMockCoreConceptExtractor(
            model_name=model_name,
            use_checkpointer=use_checkpointer,
            custom_evaluation_handler=self._ui_human_evaluation
        )
        
    def run_extraction_with_ui_evaluation(self, input_text: str) -> Dict:
        """Run extraction workflow with Streamlit UI for human evaluation"""
        
        # Initialize session state for workflow control
        if 'extraction_state' not in st.session_state:
            st.session_state.extraction_state = None
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 'input'
        if 'validation_feedback' not in st.session_state:
            st.session_state.validation_feedback = None
        if 'final_results' not in st.session_state:
            st.session_state.final_results = None
        if 'awaiting_user_input' not in st.session_state:
            st.session_state.awaiting_user_input = False
            
        try:
            # Show progress container
            progress_container = st.empty()
            with progress_container.container():
                st.markdown('<div class="progress-box">🔄 <strong>Processing in progress...</strong><br>The mock system will simulate realistic processing times.</div>', unsafe_allow_html=True)
            
            # Run the extraction workflow
            results = self.extractor.extract_keywords(input_text, st.session_state.extraction_state)
            
            # Clear progress
            progress_container.empty()
            
            st.session_state.final_results = results
            return results
            
        except Exception as e:
            st.error(f"❌ Error occurred during extraction: {str(e)}")
            st.error("Full traceback:")
            st.code(traceback.format_exc())
            return None
    
    def _ui_human_evaluation(self, state):
        """Streamlit UI version of step3_human_evaluation"""
        logger.info(f"Running UI human evaluation...{state}")
        # Store state for UI access
        if st.session_state.extraction_state == None:
            st.session_state.extraction_state = state
            st.session_state.current_step = 'evaluation'
            
            # Display results and get user feedback through UI
            concept_matrix = st.session_state.extraction_state["concept_matrix"]
            seed_keywords = st.session_state.extraction_state["seed_keywords"]
        else:
            concept_matrix = st.session_state.extraction_state["concept_matrix"]
            seed_keywords = st.session_state.extraction_state["seed_keywords"]
        # concept_matrix = state["concept_matrix"]
        # seed_keywords = state["seed_keywords"]
        
        # Initialize UI state flags if not present
        if 'show_reject_form' not in st.session_state:
            st.session_state.show_reject_form = False
        if 'show_edit_form' not in st.session_state:
            st.session_state.show_edit_form = False
        if 'ui_interaction_id' not in st.session_state:
            st.session_state.ui_interaction_id = 0
        
        # Create unique key suffix to avoid conflicts during reruns
        key_suffix = f"_{st.session_state.ui_interaction_id}"
        
        # Display the evaluation interface
        st.markdown('<div class="step-header">🎯 HUMAN EVALUATION - YOUR DECISION REQUIRED</div>', unsafe_allow_html=True)
        
        # Show concept matrix
        st.markdown("### 📋 Concept Matrix")
        with st.container():
            st.markdown('<div class="concept-box">', unsafe_allow_html=True)
            for field, value in concept_matrix.dict().items():
                st.write(f"**{field.replace('_', ' ').title()}:** {value}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show seed keywords
        st.markdown("### 🔑 Generated Keywords")
        with st.container():
            st.markdown('<div class="keyword-box">', unsafe_allow_html=True)
            for field, keywords in seed_keywords.dict().items():
                st.write(f"**{field.replace('_', ' ').title()}:** {', '.join(keywords)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Only show action buttons if no forms are active
        if not st.session_state.show_reject_form and not st.session_state.show_edit_form:
            # Action buttons
            st.markdown("### 📝 Choose your action:")
            st.info("👆 This is where you make the critical decision about the extracted keywords!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ Approve", key=f"approve_btn{key_suffix}", help="Accept the generated keywords and proceed", type="primary"):
                    feedback = ValidationFeedback(action="approve")
                    st.session_state.validation_feedback = feedback
                    st.success("✅ Keywords approved! Continuing with workflow...")
                    time.sleep(1)  # Brief pause for user feedback
                    st.rerun()
            
            with col2:
                if st.button("❌ Reject", key=f"reject_btn{key_suffix}", help="Reject keywords and restart workflow", type="secondary"):
                    st.session_state.show_reject_form = True
                    st.session_state.ui_interaction_id += 1
                    st.rerun()
            
            with col3:
                if st.button("✏️ Edit", key=f"edit_btn{key_suffix}", help="Manually modify keywords", type="secondary"):
                    st.session_state.show_edit_form = True
                    st.session_state.ui_interaction_id += 1
                    st.rerun()
        
        # Handle reject form
        if st.session_state.get('show_reject_form', False):
            with st.expander("❌ Rejection Feedback", expanded=True):
                st.warning("You are about to reject the generated keywords and restart the workflow.")
                feedback_text = st.text_area(
                    "Optional: Provide feedback for improvement:",
                    help="Explain what's wrong with the keywords to help improve the next iteration",
                    placeholder="e.g., 'Keywords are too generic' or 'Missing specific technical terms'",
                    key=f"reject_feedback_text{key_suffix}"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Submit Rejection", type="primary", key=f"submit_reject{key_suffix}"):
                        feedback = ValidationFeedback(action="reject", feedback=feedback_text)
                        st.session_state.validation_feedback = feedback
                        st.session_state.show_reject_form = False
                        st.warning("❌ Keywords rejected - restarting workflow...")
                        time.sleep(1)
                        st.rerun()
                
                with col2:
                    if st.button("Cancel", key=f"cancel_reject{key_suffix}"):
                        st.session_state.show_reject_form = False
                        st.rerun()
        
        # Handle edit form
        if st.session_state.get('show_edit_form', False):
            with st.expander("✏️ Edit Keywords", expanded=True):
                st.info("**Instructions:** Modify the keywords below. Enter keywords separated by commas.")
                
                edited_data = {}
                
                # Create editable fields for each keyword category
                for field, keywords in seed_keywords.dict().items():
                    field_name = field.replace('_', ' ').title()
                    current_str = ", ".join(keywords)
                    
                    new_keywords = st.text_input(
                        f"{field_name}:",
                        value=current_str,
                        key=f"edit_{field}{key_suffix}",
                        help="Enter keywords separated by commas"
                    )
                    
                    # Parse the input
                    if new_keywords.strip():
                        edited_data[field] = [kw.strip() for kw in new_keywords.split(',') if kw.strip()]
                    else:
                        edited_data[field] = []
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save Changes", type="primary", key=f"save_edit{key_suffix}"):
                        edited_keywords = SeedKeywords(**edited_data)
                        feedback = ValidationFeedback(action="edit", edited_keywords=edited_keywords)
                        st.session_state.validation_feedback = feedback
                        st.session_state.show_edit_form = False
                        st.success("✏️ Keywords manually edited! Continuing with your changes...")
                        time.sleep(1)
                        st.rerun()
                
                with col2:
                    if st.button("Cancel Edit", key=f"cancel_edit{key_suffix}"):
                        st.session_state.show_edit_form = False
                        st.rerun()
        
        # Wait for user action
        if st.session_state.validation_feedback is None:
            st.info("👆 Please choose an action above to continue the workflow...")
            st.stop()
        
        # Return the feedback
        feedback = st.session_state.validation_feedback
        state["validation_feedback"] = feedback
        
        # Reset for next time
        st.session_state.validation_feedback = None
        
        return {"validation_feedback": feedback}

def main():
    """Main Streamlit demo application"""
    
    # Header
    st.markdown('<div class="main-header">🚀 Patent AI Agent - Demo Interface</div>', unsafe_allow_html=True)
    # --- default state flags ---
    if 'run_demo' not in st.session_state:
        st.session_state.run_demo = False
    if 'saved_input_text' not in st.session_state:
        st.session_state.saved_input_text = ""
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'use_checkpointer_flag' not in st.session_state:
        st.session_state.use_checkpointer_flag = False
    # Demo notice
    st.markdown('''
    <div class="demo-notice">
        <h3>🎭 DEMO MODE - Mock LLM Responses</h3>
        <p>This is a demonstration version that simulates LLM responses without requiring actual model infrastructure. 
        All AI responses are pre-programmed mock data designed to showcase the interface and workflow.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("## ⚙️ Demo Configuration")
        
        # Model selection (mock)
        model_options = ["mock-qwen2.5:3b", "mock-llama3.2:3b", "mock-phi3.5:3.8b"]
        selected_model = st.selectbox(
            "Select Mock Model:",
            model_options,
            index=0,
            help="Choose the mock model (all produce similar demo responses)"
        )
        
        # Advanced options (demo)
        with st.expander("🔧 Demo Options"):
            use_checkpointer = st.checkbox(
                "Simulate Checkpointer",
                value=False,
                help="Simulate state checkpointing (demo feature)"
            )
            
            simulation_speed = st.slider(
                "Simulation Speed",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.5,
                help="Adjust mock processing speed (1.0 = normal)"
            )
        
        st.markdown("---")
        st.markdown("### 📊 Demo Workflow")
        st.markdown("""
        1. **Input Processing** ✅
        2. **Concept Extraction** ✅
        3. **Keyword Generation** ✅
        4. **👤 Human Evaluation** ⭐
        5. **Synonym Generation** ✅
        6. **Query Generation** ✅
        7. **URL Discovery** ✅
        8. **Relevance Scoring** ✅
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Key Features")
        st.markdown("""
        - **Interactive Evaluation**: Real approve/reject/edit workflow
        - **Mock Processing**: Simulated AI responses
        - **Full Interface**: Complete UI experience
        - **Export Results**: Download demo data
        """)
    
    # Main content area
    st.markdown("## 📝 Patent Idea Input")
    
    # Enhanced sample text for demo
    sample_text = """
    **Idea title**: Smart Irrigation System with IoT Sensors

    **User scenario**: A farmer managing a large agricultural field needs to optimize water usage 
    while ensuring crops receive adequate moisture. The farmer wants to monitor soil conditions 
    remotely and automatically adjust irrigation based on real-time data from multiple field locations.
    The system should integrate with weather forecasting and provide mobile app control.

    **User problem**: Traditional irrigation systems either over-water or under-water crops because 
    they operate on fixed schedules without considering actual soil moisture, weather conditions, 
    or crop-specific needs. This leads to water waste, increased costs, and potentially reduced 
    crop yields. Farmers lack real-time visibility into field conditions and cannot make data-driven 
    irrigation decisions.

    **Technical solution**: Implement a distributed network of wireless IoT sensors that measure 
    soil moisture, temperature, and humidity at multiple points across the field. The sensors 
    communicate with a central hub that processes the data using machine learning algorithms to 
    determine optimal irrigation timing and duration. The system includes automated valve controls, 
    weather API integration, and a mobile application for remote monitoring and manual override capabilities.
    """
    
    # Input text area
    input_text = st.text_area(
        "Enter your patent idea description:",
        value=sample_text,
        height=250,
        help="Describe your patent idea including the problem, solution, and technical details"
    )
    
    # Processing button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Demo Extraction", type="primary", use_container_width=True):
            if input_text.strip():
                # Clear all workflow-related session state
                for key in ['extraction_state','current_step','validation_feedback','final_results',
                            'show_reject_form','show_edit_form','awaiting_user_input','ui_interaction_id']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.run_demo = True
                st.session_state.saved_input_text = input_text
                st.session_state.selected_model = selected_model
                st.session_state.use_checkpointer_flag = use_checkpointer
                st.session_state.demo_extractor = None
            else:
                st.warning("⚠️ Please enter a patent idea description to continue the demo.")
                # Show progress
    if st.session_state.get('run_demo', False):
        logger.info(f"Running demo extraction process...")
        if st.session_state.demo_extractor is None:
            st.session_state.demo_extractor = StreamlitDemoExtractor(
                model_name=st.session_state.get('selected_model'),
                use_checkpointer=st.session_state.get('use_checkpointer_flag'))
        with st.spinner("🔄 Running demo extraction process..."):
            try:
                # Run extraction with UI evaluation
                results = st.session_state.demo_extractor.run_extraction_with_ui_evaluation(input_text)
                
                if results:
                    st.success("✅ Demo extraction completed successfully!")
                    
                    # Display results
                    st.markdown("## 📊 Demo Results")
                    st.info("💡 **Note**: All results below are generated by mock AI responses for demonstration purposes.")
                    
                    # Results tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🔑 Keywords", "🔍 Queries", "🔗 URLs"])
                    
                    with tab1:
                        st.markdown("### Concept Matrix")
                        if results.get('concept_matrix'):
                            concept_dict = results['concept_matrix'].dict()
                            concept_df = pd.DataFrame([concept_dict])
                            st.dataframe(concept_df, use_container_width=True)
                        
                        st.markdown("### Technical Summary")
                        if results.get('summary_text'):
                            st.text_area("Generated Summary:", results['summary_text'], height=150, disabled=True)
                        
                        st.markdown("### IPC Classifications")
                        if results.get('ipcs'):
                            ipc_data = []
                            for ipc in results['ipcs']:
                                ipc_data.append({
                                    'Category': ipc.get('category', 'N/A'),
                                    'Score': f"{ipc.get('score', 0):.2f}"
                                })
                            if ipc_data:
                                ipc_df = pd.DataFrame(ipc_data)
                                st.dataframe(ipc_df, use_container_width=True)
                    
                    with tab2:
                        st.markdown("### Seed Keywords")
                        if results.get('seed_keywords'):
                            keywords_dict = results['seed_keywords'].dict()
                            for category, keywords in keywords_dict.items():
                                st.write(f"**{category.replace('_', ' ').title()}:** {', '.join(keywords)}")
                        
                        st.markdown("### Expanded Keywords & Synonyms")
                        if results.get('final_keywords'):
                            for original_keyword, synonyms in results['final_keywords'].items():
                                with st.expander(f"🔍 {original_keyword}"):
                                    st.write(f"**Synonyms & Related Terms:** {', '.join(synonyms)}")
                    
                    with tab3:
                        st.markdown("### Generated Search Queries")
                        if results.get('queries') and hasattr(results['queries'], 'queries'):
                            for i, query in enumerate(results['queries'].queries, 1):
                                st.code(f"Query {i}: {query}", language="text")
                                
                        st.info("💡 These Boolean queries can be used in patent databases like Google Patents, USPTO, or EPO.")
                    
                    with tab4:
                        st.markdown("### Patent URLs Found")
                        if results.get('final_url'):
                            url_data = []
                            for url_info in results['final_url']:
                                if isinstance(url_info, dict):
                                    url_data.append({
                                        'URL': url_info.get('url', 'N/A'),
                                        'Scenario Score': f"{url_info.get('user_scenario', 0):.2f}",
                                        'Problem Score': f"{url_info.get('user_problem', 0):.2f}"
                                    })
                            
                            if url_data:
                                urls_df = pd.DataFrame(url_data)
                                st.dataframe(urls_df, use_container_width=True)
                                
                                # Download button for URLs
                                csv = urls_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Demo URLs as CSV",
                                    csv,
                                    f"demo_patent_urls_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    "text/csv"
                                )
                                
                        st.warning("⚠️ **Demo Note**: These are mock patent URLs for demonstration purposes only.")
                    
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
                        
                        # Add demo metadata
                        download_data["_demo_metadata"] = {
                            "demo_mode": True,
                            "mock_responses": True,
                            "generated_at": datetime.datetime.now().isoformat(),
                            "note": "This data was generated by mock AI responses for demonstration purposes"
                        }
                        
                        json_str = json.dumps(download_data, indent=2, ensure_ascii=False)
                        filename = f"demo_extraction_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        
                        st.download_button(
                            "💾 Download Complete Demo Results (JSON)",
                            json_str,
                            filename,
                            "application/json",
                            use_container_width=True
                        )
                
            except Exception as e:
                st.error(f"❌ Error during demo extraction: {str(e)}")
                st.error("Full traceback:")
                st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please enter a patent idea description to continue the demo.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🎭 Patent AI Agent - Demo Mode with Mock LLM<br>
        This demonstration showcases the complete interface and workflow without requiring actual LLM infrastructure.<br>
        Built with LangGraph, Streamlit & Mock AI Responses
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Import time for delays
    import time
    main()
