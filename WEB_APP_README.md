# Patent AI Agent - Web Application

A modern web interface for the Patent AI Agent keyword extraction system built with FastAPI backend and Bootstrap frontend.

## 🌟 Features

### 🔍 **Keyword Extraction**
- Interactive form for patent idea input (title, scenario, problem)
- Auto mode for fully automated processing
- Manual mode with human-in-the-loop validation
- Real-time results display with concept matrix and enhanced keywords
- Download results as JSON files

### 🏷️ **IPC Classification** 
- Classify patent text using WIPO IPC classification API
- Visual progress bars for prediction scores
- Top-ranked category predictions

### 🔗 **Patent Analysis**
- Extract information from Google Patents URLs
- Display title, abstract, claims, and description
- Expandable text sections for long content

### 📊 **Similarity Evaluation**
- Compare two texts using multiple similarity algorithms:
  - Cosine similarity with sentence transformers
  - BGE reranker scoring
  - LLM-based rerank scoring
- Visual similarity score displays

### 🗂️ **Session Management**
- Track active extraction sessions
- Check session status
- Delete completed sessions

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Application
```bash
python run_app.py
```

### 3. Access the Web Interface
- **Main Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🏗️ Architecture

### Backend (FastAPI)
- **File**: `app.py`
- **Port**: 8000
- **Features**:
  - RESTful API endpoints
  - Async request handling
  - Background task support
  - CORS enabled for frontend
  - Automatic API documentation

### Frontend (HTML + Bootstrap)
- **Main File**: `static/index.html`
- **JavaScript**: `static/app.js`
- **Features**:
  - Responsive Bootstrap 5 design
  - Tab-based interface
  - Real-time API interactions
  - Loading indicators
  - Error handling
  - Modern gradient UI

## 📡 API Endpoints

### Core Extraction
- `POST /api/extract/start` - Start keyword extraction
- `GET /api/extract/status/{session_id}` - Check extraction status
- `POST /api/extract/validate` - Handle human validation

### Analysis Tools
- `POST /api/ipc/classify` - IPC classification
- `POST /api/patent/analyze` - Patent URL analysis
- `POST /api/similarity/evaluate` - Similarity evaluation

### System
- `GET /health` - System health check
- `GET /api/settings` - Current system settings
- `GET /api/sessions` - List active sessions
- `DELETE /api/sessions/{session_id}` - Delete session
- `GET /api/download/{filename}` - Download results

## 🎨 User Interface

### Navigation Tabs
1. **Keyword Extraction** - Main patent processing interface
2. **IPC Classification** - Patent text classification
3. **Patent Analysis** - URL-based patent information extraction
4. **Similarity Check** - Text similarity evaluation
5. **Sessions** - Session management and monitoring

### Key UI Components
- **Form Inputs**: Floating labels with validation
- **Result Cards**: Clean, organized display of results
- **Progress Indicators**: Loading spinners and progress bars
- **Status Badges**: Color-coded status indicators
- **Keyword Badges**: Styled keyword displays
- **Expandable Content**: Show/hide long text sections

## 🔧 Configuration

### Environment Variables
The application uses settings from `config/settings.py`:
- `TAVILY_API_KEY` - For web search functionality
- `BRAVE_API_KEY` - For patent search
- Model configurations and API endpoints

### Customization
- **Styling**: Modify CSS variables in `static/index.html`
- **API Base URL**: Update `API_BASE` in `static/app.js`
- **Server Settings**: Modify `run_app.py` for host/port changes

## 🧪 Testing the Application

### 1. Health Check
Visit http://localhost:8000/health to verify all components are initialized.

### 2. Example Patent Idea
The interface loads with a pre-filled example:
- **Title**: "Smart Irrigation System with IoT Sensors"
- **Scenario**: Farmer managing agricultural fields
- **Problem**: Traditional irrigation inefficiencies

### 3. Test All Features
- Try keyword extraction in both auto and manual modes
- Test IPC classification with patent text
- Analyze a Google Patents URL
- Compare text similarity
- Monitor sessions

## 🔍 Troubleshooting

### Common Issues
1. **Backend not starting**: Check if all dependencies are installed
2. **Components not initialized**: Verify API keys in settings
3. **Frontend not loading**: Ensure static files are in the correct directory
4. **API errors**: Check the FastAPI logs for detailed error messages

### Debug Mode
The server runs with `reload=True` for development, automatically restarting on code changes.

## 📱 Mobile Support

The interface is fully responsive and works on:
- Desktop computers
- Tablets
- Mobile phones
- Various screen sizes

## 🚀 Production Deployment

For production deployment:

1. **Update settings**:
   ```python
   # In run_app.py, change:
   reload=False  # Disable auto-reload
   log_level="warning"  # Reduce logging
   ```

2. **Use production WSGI server**:
   ```bash
   gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

3. **Set up reverse proxy** (nginx/Apache)

4. **Configure HTTPS** for secure communication

## 📄 License

This web application is part of the Patent AI Agent project and follows the same licensing terms.

---

**Note**: This web interface provides a user-friendly way to interact with the powerful patent analysis capabilities of the underlying AI agent system.
