# Patent AI Agent - Keyword Extraction System

An AI-powered patent keyword extraction system with modular architecture for patent prior art search and analysis.

## 🏗️ Project Architecture

```
priorart_p/
├── src/                           # Core source code
│   ├── core/                      # Main AI agent modules
│   │   ├── extractor.py           # CoreConceptExtractor main class
│   │   ├── mock_extractor.py      # Mock version for testing/demo
│   │   ├── enhanced_mock_extractor.py # Enhanced mock version
│   │   └── standalone_mock_extractor.py # Standalone demo
│   ├── api/                       # External API integrations
│   │   └── ipc_classifier.py      # IPC/CPC classification API
│   ├── crawling/                  # Web scraping modules
│   │   └── patent_crawler.py      # Google Patents crawler
│   ├── evaluation/                # Similarity and evaluation
│   │   └── similarity_evaluator.py # Text similarity scoring
│   ├── prompts/                   # Prompt management
│   │   └── extraction_prompts.py  # LLM prompt templates
│   └── utils/                     # Utility functions
├── config/                        # Configuration
│   └── settings.py               # Settings and API keys
├── static/                       # Web interface assets
│   ├── index.html                # Web UI homepage
│   └── app.js                    # Frontend JavaScript
├── main.py                       # Command-line entry point
├── app.py                        # FastAPI backend server
├── streamlit_app.py              # Real Streamlit web interface
├── streamlit_demo_app.py         # Demo Streamlit interface (mock)
├── run_patent_agent.py           # Unified launcher script
├── run_*.py                      # Various interface launchers
├── requirements.txt              # Python dependencies
└── README.md                     # This documentation
```

## 🚀 Key Features

### 1. **AI-Powered Patent Keyword Extraction** (`src/core/`)
- **LangGraph Workflow**: Multi-step extraction pipeline with state management
- **8-Phase Process**: Input normalization → Concept extraction → Keyword generation → Human validation → Synonym expansion → Query generation → URL discovery → Relevance evaluation
- **Human-in-the-Loop**: Interactive approve/reject/edit workflow
- **Multiple Interfaces**: Command-line, web UI, and demo modes

### 2. **External API Integration** (`src/api/`)
- **IPC Classification**: International Patent Classification via WIPO API
- **Brave Search**: Patent URL discovery through web search
- **Tavily Search**: Additional research and synonym expansion

### 3. **Web Scraping & Data Collection** (`src/crawling/`)
- **Google Patents Crawler**: Extract title, abstract, claims, and descriptions
- **Error Handling**: Robust retry logic and exception handling
- **Content Parsing**: Structured data extraction from patent pages

### 4. **Similarity & Evaluation** (`src/evaluation/`)
- **Semantic Similarity**: Using sentence transformers for text comparison
- **Relevance Scoring**: LLM-based evaluation of patent relevance
- **Multi-metric Analysis**: Scenario and problem relevance scoring

### 5. **Prompt Engineering** (`src/prompts/`)
- **Structured Templates**: Pydantic-based prompt management
- **JSON Output Parsing**: Reliable structured data extraction
- **Context-Aware Prompts**: Dynamic prompt generation based on workflow state

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/chienthan2vn/priorart_p.git
cd priorart_p

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional)
cp .env.example .env
# Edit .env with your API keys
```

## 🎮 Usage Options

### 🎭 Demo Mode - Recommended for Testing & Demos

```bash
# No LLM infrastructure required - uses mock responses
python run_demo.py
# OR
streamlit run streamlit_demo_app.py --server.port=8502
```

**🎯 Demo Mode Features:**
- **🎭 Mock LLM**: No Ollama or API keys needed
- **📝 Complete Workflow**: Full extraction pipeline simulation
- **🎯 Real Interaction**: Actual human evaluation (approve/reject/edit)
- **📊 Full Results**: Complete results with export functionality
- **⚡ Instant Setup**: Run immediately without configuration

👉 **See demo guide**: [DEMO_README.md](DEMO_README.md)

### 🌐 Web Interface (Real LLM)

```bash
# Method 1: Using launcher script
python run_streamlit.py

# Method 2: Direct Streamlit command
streamlit run streamlit_app.py
```

**🎯 Web Interface Features:**
- **🤖 Real AI**: Actual LLM processing with Ollama
- **📝 Input Processing**: Patent idea description input
- **🎯 Interactive Evaluation**: Three interaction choices:
  - ✅ **Approve**: Accept keywords and continue
  - ❌ **Reject**: Reject and restart with feedback
  - ✏️ **Edit**: Manually modify keywords
- **📊 Visual Results**: Tabbed results display
- **💾 Export Options**: JSON/CSV export for further analysis

👉 **Detailed guide**: [STREAMLIT_README.md](STREAMLIT_README.md)

### 🚀 Unified Interface Launcher

```bash
# Choose interface from interactive menu
python run_patent_agent.py
```

### 💻 Command Line Interface

```bash
python main.py
```

### 🌐 FastAPI Backend Server

```bash
# Start REST API server
python app.py
# OR
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 📚 Module Usage Examples

```python
from src.core.extractor import CoreConceptExtractor

# Initialize the extractor
extractor = CoreConceptExtractor(model_name="qwen2.5:3b-instruct")

# Run keyword extraction
results = extractor.extract_keywords(your_patent_text)
```

### 🔌 API Module Usage

```python
# IPC Classification
from src.api.ipc_classifier import get_ipc_predictions
predictions = get_ipc_predictions("your patent summary")

# Patent Crawling
from src.crawling.patent_crawler import lay_thong_tin_patent
patent_info = lay_thong_tin_patent("https://patents.google.com/patent/...")

# Similarity Evaluation
from src.evaluation.similarity_evaluator import PatentSimilarityEvaluator
evaluator = PatentSimilarityEvaluator()
scores = evaluator.evaluate_similarity(text1, text2)
```

## ⚙️ Configuration

All configuration is managed in `config/settings.py`:

```python
from config.settings import settings

# Access settings
print(settings.DEFAULT_MODEL_NAME)
print(settings.MAX_SEARCH_RESULTS)

# Validate API keys
validation = settings.validate_api_keys()
```

## 🧪 Testing

```bash
# Run individual module tests
python test_mock_core.py           # Test mock extraction system
python test_streamlit_integration.py  # Test Streamlit integration
python test_mock_demo.py           # Test demo interface

# Test module imports
python -c "from src.core.extractor import CoreConceptExtractor; print('Core module OK')"
python -c "from src.api.ipc_classifier import get_ipc_predictions; print('API module OK')"
```

## 📋 Detailed Workflow

### 1. **Input Normalization Phase**
- Input: Raw patent idea text
- Output: Structured problem and technical components

### 2. **Concept Extraction Phase** 
- Input: Normalized text
- Output: Concept Matrix (Problem/Purpose, Object/System, Environment/Field)

### 3. **Keyword Generation Phase**
- Input: Concept Matrix
- Output: Seed keywords for each concept category

### 4. **Human Evaluation Phase**
- User choices: Approve, Reject, or Edit keywords
- Interactive interface (CLI/Web/Demo)
- Feedback incorporation for iterative improvement

### 5. **Synonym Expansion Phase**
- Automated keyword expansion via web search
- Generate synonyms and related terms
- Context-aware term extraction

### 6. **Query Generation Phase**
- Create Boolean search queries for patent databases
- Integrate IPC/CPC classification codes
- Optimize search string construction

### 7. **Patent URL Discovery Phase**
- Search for relevant patents using Brave Search API
- Target Google Patents specifically
- Collect candidate patent URLs

### 8. **Relevance Evaluation Phase**
- Extract patent content from URLs
- Score relevance using LLM evaluation
- Rank results by scenario and problem relevance

## 🔧 Core Dependencies

- **LangChain & LangGraph**: LLM application framework and workflow orchestration
- **Ollama**: Local LLM server for AI processing
- **Pydantic**: Data validation and structured output parsing
- **Streamlit**: Web interface framework
- **FastAPI**: REST API backend server
- **Sentence-Transformers**: Semantic similarity analysis
- **BeautifulSoup & Requests**: Web scraping and HTTP client
- **Transformers**: Hugging Face model integration

## 🎯 Available Interfaces

| Interface | Description | Best For | Requirements |
|-----------|-------------|----------|-------------|
| **Demo Mode** | Mock LLM responses | Testing, presentations, training | Minimal - just Streamlit |
| **Streamlit Web** | Full web interface | Interactive usage, exploration | Ollama + API keys |
| **Command Line** | Terminal interface | Automation, scripting | Ollama + API keys |
| **FastAPI Backend** | REST API server | Integration, web apps | Full dependencies |
| **Unified Launcher** | Menu-driven selection | Choosing interfaces | None |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Create Pull Request

## 📜 License

This project is distributed under the MIT License. See `LICENSE` file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/chienthan2vn/priorart_p/issues)
- **Documentation**: Reference the specialized README files:
  - [DEMO_README.md](DEMO_README.md) - Demo mode guide
  - [STREAMLIT_README.md](STREAMLIT_README.md) - Web interface guide
  - [WEB_APP_README.md](WEB_APP_README.md) - FastAPI backend guide

---

**Note**: This modular architecture is optimized for maintainability, extensibility, and testing. Each module has clear responsibilities and can be used independently or as part of the complete system.
