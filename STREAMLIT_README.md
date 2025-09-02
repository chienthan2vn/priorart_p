# Patent AI Agent - Streamlit Web Interface

This document provides instructions for running the Patent AI Agent with a user-friendly Streamlit web interface.

## 🚀 Quick Start

### Method 1: Using the Launch Script (Recommended)
```bash
python run_streamlit.py
```

### Method 2: Direct Streamlit Command
```bash
streamlit run streamlit_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

## 📋 Features

### 🎯 Interactive Workflow
The Streamlit interface provides a complete visual workflow for patent keyword extraction:

1. **Input Processing** - Enter your patent idea description
2. **Concept Extraction** - AI extracts concept matrix automatically  
3. **Keyword Generation** - AI generates seed keywords for each concept
4. **Human Evaluation** - **Three interactive choices:**
   - ✅ **Approve** - Accept the generated keywords and proceed
   - ❌ **Reject** - Reject keywords and restart with optional feedback
   - ✏️ **Edit** - Manually modify keywords before proceeding
5. **Synonym Expansion** - AI finds related terms and synonyms
6. **Query Generation** - Creates Boolean search queries
7. **Patent Search** - Finds relevant patent URLs
8. **Relevance Scoring** - Evaluates patent relevance

### 🎛️ Configuration Options

#### Sidebar Controls
- **Model Selection**: Choose from available LLM models
  - `qwen2.5:3b-instruct` (default)
  - `llama3.2:3b`
  - `phi3.5:3.8b`
- **Advanced Settings**:
  - Enable/disable checkpointing
  - Adjust model temperature

#### Sample Input
The interface includes a pre-filled sample patent idea about a "Smart Irrigation System with IoT Sensors" for easy testing.

### 📊 Results Display

The interface organizes results in intuitive tabs:

#### 📋 Summary Tab
- **Concept Matrix**: Shows the extracted problem/purpose, object/system, and environment/field
- **IPC Classifications**: Displays International Patent Classification codes with scores

#### 🔑 Keywords Tab  
- **Seed Keywords**: Original keywords extracted from the concept matrix
- **Expanded Keywords**: Synonyms and related terms for each seed keyword

#### 🔍 Queries Tab
- **Search Queries**: Boolean patent search queries generated from keywords
- Ready to use in patent databases

#### 🔗 URLs Tab
- **Patent URLs**: Relevant patent URLs found through web search
- **Relevance Scores**: Scenario and problem relevance scores for each patent
- **CSV Download**: Export URLs and scores for further analysis

### 💾 Export Options
- **Complete Results (JSON)**: Download all extraction results in JSON format
- **Patent URLs (CSV)**: Export patent URLs with relevance scores

## 🔧 Technical Details

### Architecture Integration
The Streamlit interface seamlessly integrates with the existing `CoreConceptExtractor` class through:

1. **Custom Evaluation Handler**: Replaces command-line interaction with web UI
2. **Session State Management**: Maintains workflow state across user interactions  
3. **Real-time Updates**: Immediate feedback for user actions

### Key Components

#### `StreamlitPatentExtractor`
- Wrapper class that initializes `CoreConceptExtractor` with UI evaluation handler
- Manages Streamlit session state for workflow control
- Handles error reporting and progress indication

#### `_ui_human_evaluation`
- Replaces the original `step3_human_evaluation` function
- Provides interactive buttons for Approve/Reject/Edit actions
- Manages expandable forms for rejection feedback and keyword editing

### Human Evaluation Logic

The core evaluation logic mirrors the original command-line interface:

```python
# Three action choices
if action == "approve":
    feedback = ValidationFeedback(action="approve")
elif action == "reject": 
    feedback = ValidationFeedback(action="reject", feedback=feedback_text)
elif action == "edit":
    feedback = ValidationFeedback(action="edit", edited_keywords=edited_keywords)
```

#### Approve Flow
1. User clicks "✅ Approve" button
2. System creates `ValidationFeedback(action="approve")`
3. Workflow continues to synonym generation

#### Reject Flow  
1. User clicks "❌ Reject" button
2. Expandable form appears for optional feedback
3. System creates `ValidationFeedback(action="reject", feedback=text)`
4. Workflow restarts from concept extraction with feedback

#### Edit Flow
1. User clicks "✏️ Edit" button
2. Expandable form shows current keywords in editable text fields
3. User modifies keywords and clicks "Save Changes"
4. System creates `ValidationFeedback(action="edit", edited_keywords=SeedKeywords(...))`
5. Workflow continues with modified keywords

## 🛠️ Troubleshooting

### Common Issues

1. **Streamlit not found**
   - Run: `pip install streamlit>=1.28.0`

2. **Model not available**
   - Ensure Ollama is running with the selected model
   - Check available models: `ollama list`

3. **API Keys missing**
   - Set environment variables for `TAVILY_API_KEY` and `BRAVE_API_KEY`
   - Or update values in `config/settings.py`

4. **Port already in use**
   - Change port: `streamlit run streamlit_app.py --server.port=8502`

### Performance Tips

- Use smaller models (3B parameters) for faster processing
- Enable checkpointing for long workflows
- Lower model temperature for more consistent results

## 🔗 Integration with Original System

The Streamlit interface is fully compatible with the original command-line system:

- **Same Models**: Uses identical LLM models and prompts
- **Same Workflow**: Follows the exact same processing steps
- **Same Output**: Produces identical results to `main.py`
- **Enhanced UX**: Adds visual interface without changing core logic

You can switch between interfaces without any configuration changes.

## 🎯 Next Steps

After running the extraction:

1. **Review Results**: Check all tabs for comprehensive output
2. **Download Data**: Export JSON/CSV files for further analysis
3. **Patent Search**: Use generated queries in patent databases
4. **Iterate**: Run with different inputs or settings to refine results

For advanced usage and API integration, refer to the main `README.md` file.
