# Mock Patent Concept Extractor

This project provides multiple versions of a mock patent concept extractor that maintains the exact multi-agent architecture from the original `extractor.py` but uses constant data instead of real LLM calls.

## Available Versions

### 1. Enhanced Mock Extractor (`enhanced_mock_extractor.py`)
**Full-featured version with LangGraph integration**

- **Features**: Complete LangGraph StateGraph implementation with all nodes and edges
- **Dependencies**: Requires `langgraph`, `pydantic`, and other external libraries
- **Use Case**: For environments with full dependency installation
- **Architecture**: Exact replica of original extractor.py multi-agent workflow

### 2. Standalone Mock Extractor (`standalone_mock_extractor.py`)
**Self-contained version for testing**

- **Features**: Same multi-agent logic but without external dependencies
- **Dependencies**: Only Python standard library + optional pydantic fallback
- **Use Case**: For testing, development, or environments without LangGraph
- **Architecture**: Sequential execution of the same workflow steps

### 3. Basic Mock Extractor (`mock_extractor.py`)
**Simple version for basic testing**

- **Features**: Basic workflow simulation
- **Dependencies**: Minimal
- **Use Case**: Quick testing and prototyping

## Usage

### With Streamlit App

The Streamlit demo app (`streamlit_demo_app.py`) automatically selects the best available version:

```python
# Automatic fallback selection
try:
    from src.core.enhanced_mock_extractor import EnhancedMockCoreConceptExtractor
    EXTRACTOR_CLASS = EnhancedMockCoreConceptExtractor
except ImportError:
    from src.core.standalone_mock_extractor import StandaloneMockCoreConceptExtractor
    EXTRACTOR_CLASS = StandaloneMockCoreConceptExtractor
```

### Direct Usage

#### Enhanced Mock Extractor (with LangGraph)
```python
from src.core.enhanced_mock_extractor import EnhancedMockCoreConceptExtractor

extractor = EnhancedMockCoreConceptExtractor(
    model_name="mock-llm",
    use_checkpointer=False,
    custom_evaluation_handler=None  # Optional UI integration
)

results = extractor.extract_keywords("Your patent idea text here...")
```

#### Standalone Mock Extractor (no external deps)
```python
from src.core.standalone_mock_extractor import StandaloneMockCoreConceptExtractor

extractor = StandaloneMockCoreConceptExtractor(
    model_name="mock-llm",
    use_checkpointer=False,
    custom_evaluation_handler=None
)

results = extractor.extract_keywords("Your patent idea text here...")
```

## Architecture Overview

Both enhanced and standalone versions maintain the exact same multi-agent workflow:

### Workflow Steps
1. **Input Normalization** - Clean and structure input text
2. **Step 0** - Initial processing
3. **Concept Extraction** - Extract core concepts (parallel with summary)
4. **Summary Generation** - Generate technical summary (parallel with concepts)
5. **Keyword Generation** - Generate seed keywords from concepts
6. **Human Evaluation** - Interactive validation with approve/edit/reject options
7. **Manual Editing** - Optional keyword editing (if user chooses edit)
8. **IPC Classification** - Mock patent classification
9. **Synonym Generation** - Expand keywords with synonyms and related terms
10. **Query Generation** - Create Boolean search queries
11. **URL Generation** - Find patent URLs via search
12. **URL Evaluation** - Score patents for relevance

### State Management

Both versions use the same `ExtractionState` structure:

```python
class ExtractionState(TypedDict):
    input_text: str
    problem: Optional[str]
    technical: Optional[str]
    summary_text: str
    ipcs: Any 
    concept_matrix: Optional[ConceptMatrix]
    seed_keywords: Optional[SeedKeywords]
    validation_feedback: Optional[ValidationFeedback]
    final_keywords: dict
    queries: list
    final_url: list
```

### Data Models

All versions use identical Pydantic models:

- `NormalizationOutput` - Normalized input structure
- `ConceptMatrix` - Core concept extraction results
- `SeedKeywords` - Generated keyword categories
- `ValidationFeedback` - User validation responses
- `QueriesResponse` - Generated search queries

## Testing

### Run Comprehensive Tests
```bash
python test_standalone_mock.py
```

This test suite verifies:
- Complete workflow execution
- Custom evaluation handler integration
- Architecture consistency
- Data model compatibility

### Expected Output
```
🚀 Running Standalone Mock Extractor Tests
🧪 Testing Standalone Mock Extractor Complete Workflow...
📊 Test Summary:
  Complete Workflow: ✅ PASS
  Custom Handler: ✅ PASS  
  Architecture: ✅ PASS
🎉 All tests passed! Standalone Mock Extractor is working correctly.
```

## Custom Evaluation Handlers

Both versions support custom evaluation handlers for UI integration:

```python
def custom_ui_handler(state):
    """Custom handler for Streamlit or other UI integration"""
    # Your UI logic here
    # Return validation feedback
    return {"validation_feedback": ValidationFeedback(action="approve")}

extractor = StandaloneMockCoreConceptExtractor(
    custom_evaluation_handler=custom_ui_handler
)
```

## Mock Data

The extractors return consistent mock data for testing:

### Sample Results
- **Problem/Purpose**: "Optimize water usage in agricultural irrigation..."
- **Object/System**: "Smart irrigation system with IoT sensors..."  
- **Environment/Field**: "Agricultural field management, precision farming..."
- **Keywords**: Water optimization, IoT sensors, agriculture, etc.
- **Synonyms**: Irrigation efficiency, smart sensors, farming, etc.
- **Patent URLs**: 5+ mock Google Patents URLs with relevance scores
- **IPC Classifications**: Mock patent classification codes

## Integration Notes

### Streamlit Integration
- Automatic fallback between enhanced and standalone versions
- Custom evaluation handler for interactive UI
- Session state management for workflow control

### Command Line Usage
- Both versions support terminal-based human evaluation
- Interactive approve/edit/reject workflow
- Manual keyword editing capabilities

## Performance

### Enhanced Version
- Uses LangGraph for proper state management
- Supports checkpointing and complex workflows
- Requires external dependencies

### Standalone Version  
- Sequential execution with same logic
- No external dependencies required
- Slightly faster due to reduced overhead
- Suitable for testing and development

## Compatibility

Both versions are designed to be drop-in replacements for the original extractor while using mock data instead of real LLM calls. They maintain:

- Same method signatures
- Same return data structures  
- Same workflow logic
- Same error handling patterns
- Same logging and monitoring

This allows you to test the complete multi-agent system architecture without requiring LLM infrastructure or API keys.
