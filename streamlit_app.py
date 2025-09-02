"""
Streamlit Web Interface for Patent AI Agent
Provides a user-friendly web interface for the patent keyword extraction system
"""

import streamlit as st
import json
import datetime
import logging
import traceback
from typing import Dict, Any, Optional
import pandas as pd

# Import the core extractor
from src.core.extractor import CoreConceptExtractor, ValidationFeedback, SeedKeywords

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Patent AI Agent - Keyword Extraction",
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
    .action-button {
        margin: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        border: none;
        cursor: pointer;
    }
    .approve-btn {
        background-color: #28a745;
        color: white;
    }
    .reject-btn {
        background-color: #dc3545;
        color: white;
    }
    .edit-btn {
        background-color: #ffc107;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitPatentExtractor:
    """Streamlit-integrated Patent Extractor with UI-based human evaluation"""
    
    def __init__(self, model_name: str = None, use_checkpointer: bool = None):
        # Create extractor with custom evaluation handler
        self.extractor = CoreConceptExtractor(
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
            # Run the extraction workflow
            results = self.extractor.extract_keywords(input_text)
            st.session_state.final_results = results
            return results
            
        except Exception as e:
            st.error(f"❌ Error occurred during extraction: {str(e)}")
            st.error("Full traceback:")
            st.code(traceback.format_exc())
            return None
    
    def _ui_human_evaluation(self, state):
        """Streamlit UI version of step3_human_evaluation"""
        
        # Store state for UI access
        if 'extraction_state' not in st.session_state:
            st.session_state.extraction_state = state
            st.session_state.current_step = 'evaluation'
            
            # Display results and get user feedback through UI
            concept_matrix = state["concept_matrix"]
            seed_keywords = state["seed_keywords"]
        else:
            concept_matrix = st.session_state.extraction_state["concept_matrix"]
            seed_keywords = st.session_state.extraction_state["seed_keywords"]
        
        # Display the evaluation interface
        st.markdown('<div class="step-header">🎯 FINAL EVALUATION - HUMAN DECISION</div>', unsafe_allow_html=True)
        
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
        
        # Action buttons
        st.markdown("### 📝 Choose your action:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Approve", key="approve_btn", help="Accept the generated keywords and proceed"):
                feedback = ValidationFeedback(action="approve")
                st.session_state.validation_feedback = feedback
                st.success("✅ Keywords approved!")
                st.rerun()
        
        with col2:
            if st.button("❌ Reject", key="reject_btn", help="Reject keywords and restart workflow"):
                st.session_state.show_reject_form = True
        
        with col3:
            if st.button("✏️ Edit", key="edit_btn", help="Manually modify keywords"):
                st.session_state.show_edit_form = True
        
        # Handle reject form
        if st.session_state.get('show_reject_form', False):
            with st.expander("❌ Rejection Feedback", expanded=True):
                feedback_text = st.text_area(
                    "Optional: Provide feedback for improvement:",
                    help="Explain what's wrong with the keywords to help improve the next iteration"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Submit Rejection", type="primary"):
                        feedback = ValidationFeedback(action="reject", feedback=feedback_text)
                        st.session_state.validation_feedback = feedback
                        st.session_state.show_reject_form = False
                        st.warning("❌ Keywords rejected - restarting workflow")
                        st.rerun()
                
                with col2:
                    if st.button("Cancel"):
                        st.session_state.show_reject_form = False
                        st.rerun()
        
        # Handle edit form
        if st.session_state.get('show_edit_form', False):
            with st.expander("✏️ Edit Keywords", expanded=True):
                st.write("**Current keywords will be displayed. Modify as needed:**")
                
                edited_data = {}
                
                # Create editable fields for each keyword category
                for field, keywords in seed_keywords.dict().items():
                    field_name = field.replace('_', ' ').title()
                    current_str = ", ".join(keywords)
                    
                    new_keywords = st.text_input(
                        f"{field_name}:",
                        value=current_str,
                        key=f"edit_{field}",
                        help="Enter keywords separated by commas"
                    )
                    
                    # Parse the input
                    if new_keywords.strip():
                        edited_data[field] = [kw.strip() for kw in new_keywords.split(',') if kw.strip()]
                    else:
                        edited_data[field] = []
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save Changes", type="primary"):
                        edited_keywords = SeedKeywords(**edited_data)
                        feedback = ValidationFeedback(action="edit", edited_keywords=edited_keywords)
                        st.session_state.validation_feedback = feedback
                        st.session_state.show_edit_form = False
                        st.success("✏️ Keywords manually edited")
                        st.rerun()
                
                with col2:
                    if st.button("Cancel Edit"):
                        st.session_state.show_edit_form = False
                        st.rerun()
        
        # Wait for user action
        if st.session_state.validation_feedback is None:
            st.info("👆 Please choose an action above to continue...")
            st.stop()
        
        # Return the feedback
        feedback = st.session_state.validation_feedback
        state["validation_feedback"] = feedback
        
        # Reset for next time
        st.session_state.validation_feedback = None
        
        return {"validation_feedback": feedback}

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🚀 Patent AI Agent - Keyword Extraction System</div>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Model selection
        model_options = ["qwen2.5:3b-instruct", "llama3.2:3b", "phi3.5:3.8b"]
        selected_model = st.selectbox(
            "Select LLM Model:",
            model_options,
            index=0,
            help="Choose the language model for extraction"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            use_checkpointer = st.checkbox(
                "Use Checkpointer",
                value=False,
                help="Enable state checkpointing for workflow"
            )
            
            temperature = st.slider(
                "Model Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Controls randomness in model responses"
            )
        
        st.markdown("---")
        st.markdown("### 📊 Workflow Steps")
        st.markdown("""
        1. **Input Normalization** - Extract problem & technical aspects
        2. **Concept Extraction** - Create concept matrix
        3. **Keyword Generation** - Generate seed keywords  
        4. **Human Evaluation** - Approve/Reject/Edit
        5. **Synonym Generation** - Expand keywords
        6. **Query Generation** - Create search queries
        7. **URL Generation** - Find patent URLs
        8. **Evaluation** - Score relevance
        """)
    
    # Main content area
    st.markdown("## 📝 Patent Idea Input")
    
    # Sample text for testing
    sample_text = """
    **Idea title**: Smart Irrigation System with IoT Sensors

    **User scenario**: A farmer managing a large agricultural field needs to optimize water usage 
    while ensuring crops receive adequate moisture. The farmer wants to monitor soil conditions 
    remotely and automatically adjust irrigation based on real-time data from multiple field locations.

    **User problem**: Traditional irrigation systems either over-water or under-water crops because 
    they operate on fixed schedules without considering actual soil moisture, weather conditions, 
    or crop-specific needs. This leads to water waste, increased costs, and potentially reduced 
    crop yields.
    """
    
    # Input text area
    input_text = st.text_area(
        "Enter your patent idea description:",
        value=sample_text,
        height=200,
        help="Describe your patent idea including the problem, solution, and technical details"
    )
    
    # Processing button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Extraction Process", type="primary", use_container_width=True):
            if input_text.strip():
                # Clear previous results
                for key in ['extraction_state', 'current_step', 'validation_feedback', 'final_results', 'show_reject_form', 'show_edit_form']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Initialize the streamlit extractor with selected model
                st_extractor = StreamlitPatentExtractor(
                    model_name=selected_model,
                    use_checkpointer=use_checkpointer
                )
                
    # Show progress
    with st.spinner("🔄 Processing patent idea..."):
        try:
            # Run extraction with UI evaluation
            results = st_extractor.run_extraction_with_ui_evaluation(input_text)
            
            if results:
                st.success("✅ Extraction completed successfully!")
                
                # Display results
                st.markdown("## 📊 Final Results")
                
                # Results tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🔑 Keywords", "🔍 Queries", "🔗 URLs"])
                
                with tab1:
                    st.markdown("### Concept Matrix")
                    if results.get('concept_matrix'):
                        concept_df = pd.DataFrame([results['concept_matrix'].dict()])
                        st.dataframe(concept_df, use_container_width=True)
                    
                    st.markdown("### IPC Classifications")
                    if results.get('ipcs'):
                        ipc_data = []
                        for ipc in results['ipcs']:
                            ipc_data.append({
                                'Category': ipc.get('category', 'N/A'),
                                'Score': ipc.get('score', 'N/A')
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
                    
                    st.markdown("### Expanded Keywords")
                    if results.get('final_keywords'):
                        for original_keyword, synonyms in results['final_keywords'].items():
                            with st.expander(f"🔍 {original_keyword}"):
                                st.write(f"**Synonyms & Related Terms:** {', '.join(synonyms)}")
                
                with tab3:
                    st.markdown("### Generated Search Queries")
                    if results.get('queries') and hasattr(results['queries'], 'queries'):
                        for i, query in enumerate(results['queries'].queries, 1):
                            st.write(f"**Query {i}:** `{query}`")
                
                with tab4:
                    st.markdown("### Patent URLs Found")
                    if results.get('final_url'):
                        url_data = []
                        for url_info in results['final_url']:
                            if isinstance(url_info, dict):
                                url_data.append({
                                    'URL': url_info.get('url', 'N/A'),
                                    'Scenario Score': url_info.get('user_scenario', 0),
                                    'Problem Score': url_info.get('user_problem', 0)
                                })
                        
                        if url_data:
                            urls_df = pd.DataFrame(url_data)
                            st.dataframe(urls_df, use_container_width=True)
                            
                            # Download button for URLs
                            csv = urls_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download URLs as CSV",
                                csv,
                                f"patent_urls_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv"
                            )
                
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
                    
                    json_str = json.dumps(download_data, indent=2, ensure_ascii=False)
                    filename = f"extraction_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    st.download_button(
                        "💾 Download Complete Results (JSON)",
                        json_str,
                        filename,
                        "application/json",
                        use_container_width=True
                    )
            
        except Exception as e:
            st.error(f"❌ Error during extraction: {str(e)}")
            st.error("Full traceback:")
            st.code(traceback.format_exc())

    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🚀 Patent AI Agent - Powered by LangGraph & Streamlit<br>
        Built for intelligent patent keyword extraction and prior art search
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
