"""
Mock FastAPI Backend for Patent AI Agent
Simulates LLM responses for testing the web application without requiring actual AI models
"""

import os
import json
import datetime
import asyncio
import random
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Mock data for simulating responses
MOCK_CONCEPT_MATRIX = {
    "problem_purpose": "Optimize water usage and ensure adequate crop moisture through automated irrigation control",
    "object_system": "IoT-based smart irrigation system with soil sensors and automated water control valves", 
    "environment_field": "Agricultural field management and precision farming applications"
}

MOCK_SEED_KEYWORDS = {
    "problem_purpose": ["water optimization", "moisture control", "irrigation efficiency", "crop hydration"],
    "object_system": ["IoT sensors", "soil moisture sensor", "automated valve", "irrigation controller"],
    "environment_field": ["agriculture", "farming", "precision agriculture", "crop management"]
}

MOCK_ENHANCED_KEYWORDS = {
    "water optimization": ["water conservation", "irrigation optimization", "water management", "efficient watering", "water usage control"],
    "IoT sensors": ["internet of things", "wireless sensors", "connected devices", "smart sensors", "sensor network"],
    "agriculture": ["farming", "agronomy", "crop production", "agricultural technology", "farm management"],
    "moisture control": ["humidity regulation", "water content management", "soil moisture monitoring", "irrigation control"],
    "automated valve": ["solenoid valve", "electric valve", "control valve", "irrigation valve", "water valve"]
}

MOCK_IPC_PREDICTIONS = [
    {"rank": 1, "category": "A01G25/16", "score": 85},
    {"rank": 2, "category": "G05B15/02", "score": 72},
    {"rank": 3, "category": "A01G27/00", "score": 68}
]

MOCK_SEARCH_QUERIES = [
    "(water optimization OR irrigation efficiency) AND (IoT sensors OR smart sensors) AND agriculture",
    "(moisture control OR soil moisture) AND (automated valve OR irrigation controller) AND farming",
    "(precision agriculture OR smart farming) AND (sensor network OR wireless sensors)",
    "A01G25/16 AND (irrigation system OR water management)",
    "(crop management OR agricultural technology) AND (automated irrigation OR smart watering)",
    "(farming technology OR agtech) AND (IoT agriculture OR connected farming)"
]

MOCK_PATENT_INFO = {
    "title": "Smart Irrigation System with Wireless Sensor Network - US Patent 10,123,456",
    "abstract": "A smart irrigation system comprising a network of wireless soil moisture sensors, a central control unit, and automated irrigation valves. The system monitors soil conditions in real-time and automatically adjusts water delivery based on crop requirements, weather data, and soil moisture levels. The invention reduces water waste while ensuring optimal crop growth through precision irrigation control.",
    "claims": "1. An irrigation system comprising: a plurality of wireless soil moisture sensors positioned throughout an agricultural area; a central processing unit configured to receive data from the sensors; automated water control valves connected to irrigation lines; wherein the system automatically activates irrigation based on sensor readings and predetermined moisture thresholds. 2. The system of claim 1, further comprising weather monitoring capabilities. 3. The system of claim 1, wherein the sensors communicate via IoT protocols.",
    "description": "The present invention relates to agricultural technology, specifically to automated irrigation systems for optimizing water usage in farming operations. Traditional irrigation methods often result in over-watering or under-watering, leading to crop stress and water waste. The disclosed system addresses these problems by providing real-time monitoring and automated control of irrigation based on actual soil conditions and plant needs. The system includes multiple components working together to provide efficient water management for agricultural applications."
}

MOCK_SIMILARITY_SCORES = {
    "similarities_score": 0.78,
    "rerank_score": 2.34,
    "llm_score": 0.82
}

# Global variables for mock sessions
mock_sessions: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Initializing Mock Patent AI Agent...")
    print("✅ Mock components initialized successfully!")
    yield
    # Shutdown
    print("🔄 Shutting down Mock Patent AI Agent...")

# Initialize FastAPI app
app = FastAPI(
    title="Patent AI Agent API (Mock Version)",
    description="Mock AI-powered patent keyword extraction system for testing",
    version="1.0.0-mock",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for API
class PatentIdeaInput(BaseModel):
    title: str = Field(..., description="Title of the patent idea")
    scenario: str = Field(..., description="User scenario description")
    problem: str = Field(..., description="Problem the invention solves")

class ExtractionRequest(BaseModel):
    input_text: str = Field(..., description="Patent idea text to process")
    use_auto_mode: bool = Field(default=False, description="Skip human validation steps")

class ValidationRequest(BaseModel):
    session_id: str = Field(..., description="Session ID for the extraction")
    action: str = Field(..., description="Action: 'approve', 'reject', or 'edit'")
    feedback: Optional[str] = Field(None, description="Feedback text for rejection")
    edited_keywords: Optional[Dict[str, List[str]]] = Field(None, description="Manually edited keywords")

class IPCClassificationRequest(BaseModel):
    text: str = Field(..., description="Text to classify")

class PatentUrlRequest(BaseModel):
    url: str = Field(..., description="Patent URL to analyze")

class SimilarityRequest(BaseModel):
    text1: str = Field(..., description="First text for comparison")
    text2: str = Field(..., description="Second text for comparison")

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("""
        <html>
            <head><title>Patent AI Agent (Mock)</title></head>
            <body>
                <h1>Patent AI Agent API (Mock Version)</h1>
                <p>Frontend not found. Please check static files.</p>
                <p><a href="/docs">View API Documentation</a></p>
            </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "mock",
        "timestamp": datetime.datetime.now().isoformat(),
        "components": {
            "extractor": True,
            "patent_crawler": True,
            "similarity_evaluator": True,
            "note": "All components are mocked for testing"
        }
    }

@app.get("/api/settings")
async def get_settings():
    """Get current system settings"""
    return {
        "model_name": "mock-llm-v1.0",
        "temperature": 0.7,
        "max_search_results": 5,
        "api_keys_status": {
            "TAVILY_API_KEY": True,
            "BRAVE_API_KEY": True
        },
        "note": "Mock version - simulated responses"
    }

@app.post("/api/extract/start")
async def start_extraction(request: ExtractionRequest):
    """Start the patent keyword extraction process (mocked)"""
    
    # Simulate processing delay
    await asyncio.sleep(1)
    
    # Generate session ID
    session_id = f"mock_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
    
    if request.use_auto_mode:
        # Auto mode - return complete mocked results
        results = {
            "input_text": request.input_text,
            "problem": "Traditional irrigation systems are inefficient and waste water",
            "technical": "IoT-based smart irrigation with automated control",
            "concept_matrix": MOCK_CONCEPT_MATRIX,
            "seed_keywords": MOCK_SEED_KEYWORDS,
            "final_keywords": MOCK_ENHANCED_KEYWORDS,
            "summary_text": "Smart irrigation system utilizing IoT sensors for automated water management in agricultural applications",
            "ipcs": MOCK_IPC_PREDICTIONS,
            "queries": {"queries": MOCK_SEARCH_QUERIES},
            "final_url": [
                {"url": "https://patents.google.com/patent/US10123456B2", "user_scenario": 0.85, "user_problem": 0.78},
                {"url": "https://patents.google.com/patent/US9876543B1", "user_scenario": 0.72, "user_problem": 0.81},
                {"url": "https://patents.google.com/patent/US11234567A1", "user_scenario": 0.69, "user_problem": 0.74}
            ]
        }
        
        # Save mock results to file
        filename = f"extraction_results_{session_id}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return {
            "session_id": session_id,
            "status": "completed",
            "results": results,
            "filename": filename,
            "note": "Mock extraction completed with simulated data"
        }
    else:
        # Manual mode - store session for human-in-the-loop
        mock_sessions[session_id] = {
            "input_text": request.input_text,
            "status": "started",
            "timestamp": datetime.datetime.now().isoformat(),
            "concept_matrix": MOCK_CONCEPT_MATRIX,
            "seed_keywords": MOCK_SEED_KEYWORDS
        }
        
        return {
            "session_id": session_id,
            "status": "started",
            "message": "Mock extraction started. Use /api/extract/status to check progress.",
            "note": "Mock mode - simulated processing"
        }

@app.get("/api/extract/status/{session_id}")
async def get_extraction_status(session_id: str):
    """Get the status of an extraction session"""
    if session_id not in mock_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = mock_sessions[session_id]
    
    # Simulate different statuses based on time
    created_time = datetime.datetime.fromisoformat(session["timestamp"])
    elapsed = (datetime.datetime.now() - created_time).seconds
    
    if elapsed < 30:
        session["status"] = "processing"
        session["stage"] = "concept_extraction"
    elif elapsed < 60:
        session["status"] = "processing"
        session["stage"] = "keyword_generation"
    else:
        session["status"] = "waiting_for_validation"
        session["stage"] = "human_evaluation"
        session["concept_matrix"] = MOCK_CONCEPT_MATRIX
        session["seed_keywords"] = MOCK_SEED_KEYWORDS
    
    return session

@app.post("/api/extract/validate")
async def validate_extraction(request: ValidationRequest):
    """Handle human validation of extraction results (mocked)"""
    if request.session_id not in mock_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = mock_sessions[request.session_id]
    
    # Simulate processing delay
    await asyncio.sleep(0.5)
    
    if request.action == "approve":
        session["status"] = "approved"
        session["validation"] = "approved"
        session["final_keywords"] = MOCK_ENHANCED_KEYWORDS
    elif request.action == "reject":
        session["status"] = "rejected"
        session["validation"] = "rejected"
        session["feedback"] = request.feedback
    elif request.action == "edit":
        session["status"] = "edited"
        session["validation"] = "edited"
        session["edited_keywords"] = request.edited_keywords or MOCK_SEED_KEYWORDS
        session["final_keywords"] = MOCK_ENHANCED_KEYWORDS
    
    return {
        "status": "success",
        "session_updated": True,
        "note": "Mock validation completed"
    }

@app.post("/api/ipc/classify")
async def classify_ipc(request: IPCClassificationRequest):
    """Classify text using IPC classification (mocked)"""
    
    # Simulate processing delay
    await asyncio.sleep(0.8)
    
    # Add some randomization to make it more realistic
    predictions = []
    for i, pred in enumerate(MOCK_IPC_PREDICTIONS):
        predictions.append({
            "rank": pred["rank"],
            "category": pred["category"],
            "score": pred["score"] + random.randint(-5, 5)  # Add some variation
        })
    
    return {
        "text": request.text[:100] + "..." if len(request.text) > 100 else request.text,
        "predictions": predictions,
        "timestamp": datetime.datetime.now().isoformat(),
        "note": "Mock IPC classification"
    }

@app.post("/api/patent/analyze")
async def analyze_patent_url(request: PatentUrlRequest):
    """Analyze a patent from its URL (mocked)"""
    
    # Simulate processing delay
    await asyncio.sleep(1.2)
    
    if not request.url.strip():
        raise HTTPException(status_code=400, detail="URL is required")
    
    if "patents.google.com" not in request.url:
        return {
            "url": request.url,
            "patent_info": {
                "title": "Error: Invalid URL format",
                "abstract": "Please provide a valid Google Patents URL",
                "claims": "URL validation failed",
                "description": "Only Google Patents URLs are supported in this mock version"
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "note": "Mock patent analysis - URL validation failed"
        }
    
    # Return mock patent information
    mock_info = MOCK_PATENT_INFO.copy()
    
    # Vary the patent number based on URL for realism
    url_hash = hash(request.url) % 1000000
    mock_info["title"] = mock_info["title"].replace("10,123,456", f"{url_hash:,}")
    
    return {
        "url": request.url,
        "patent_info": mock_info,
        "timestamp": datetime.datetime.now().isoformat(),
        "note": "Mock patent analysis with simulated data"
    }

@app.post("/api/similarity/evaluate")
async def evaluate_similarity(request: SimilarityRequest):
    """Evaluate similarity between two texts (mocked)"""
    
    # Simulate processing delay
    await asyncio.sleep(1.0)
    
    if not request.text1.strip() or not request.text2.strip():
        raise HTTPException(status_code=400, detail="Both texts are required")
    
    # Generate somewhat realistic similarity scores based on text length and common words
    text1_words = set(request.text1.lower().split())
    text2_words = set(request.text2.lower().split())
    
    # Calculate a simple word overlap similarity
    common_words = text1_words.intersection(text2_words)
    total_words = text1_words.union(text2_words)
    base_similarity = len(common_words) / len(total_words) if total_words else 0
    
    # Add some randomization and scaling
    similarities_score = min(0.95, max(0.1, base_similarity + random.uniform(-0.1, 0.2)))
    rerank_score = similarities_score * 3 + random.uniform(-0.5, 0.5)  # Scale to reranker range
    llm_score = min(0.95, max(0.1, similarities_score + random.uniform(-0.15, 0.15)))
    
    return {
        "text1": request.text1[:100] + "..." if len(request.text1) > 100 else request.text1,
        "text2": request.text2[:100] + "..." if len(request.text2) > 100 else request.text2,
        "similarity_scores": {
            "similarities_score": round(similarities_score, 3),
            "rerank_score": round(rerank_score, 3),
            "llm_score": round(llm_score, 3)
        },
        "timestamp": datetime.datetime.now().isoformat(),
        "note": "Mock similarity evaluation with computed scores"
    }

@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "sessions": list(mock_sessions.keys()),
        "total": len(mock_sessions),
        "timestamp": datetime.datetime.now().isoformat(),
        "note": "Mock session management"
    }

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id not in mock_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del mock_sessions[session_id]
    return {
        "status": "success", 
        "message": f"Mock session {session_id} deleted",
        "note": "Mock session deletion"
    }

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download extraction results file"""
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        filename,
        media_type='application/json',
        filename=filename
    )

# Additional mock endpoints for testing
@app.get("/api/mock/info")
async def mock_info():
    """Information about the mock version"""
    return {
        "version": "1.0.0-mock",
        "description": "Mock version of Patent AI Agent for testing without LLM resources",
        "features": [
            "Simulated keyword extraction with realistic data",
            "Mock IPC classification with sample predictions", 
            "Fake patent analysis with template responses",
            "Computed similarity scores based on text overlap",
            "Session management with mock data"
        ],
        "limitations": [
            "No actual AI/LLM processing",
            "Responses are pre-defined templates",
            "Some randomization for realism",
            "No real patent database access"
        ],
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/api/mock/reset")
async def reset_mock_data():
    """Reset all mock sessions and data"""
    global mock_sessions
    mock_sessions = {}
    
    # Clean up any generated files
    for file in os.listdir("."):
        if file.startswith("extraction_results_mock_session_") and file.endswith(".json"):
            try:
                os.remove(file)
            except:
                pass
    
    return {
        "status": "success",
        "message": "All mock data reset",
        "timestamp": datetime.datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Mock Patent AI Agent...")
    print("📝 This is a mock version for testing without LLM resources")
    print("🌐 Web interface: http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    print("ℹ️  Mock info: http://localhost:8000/api/mock/info")
    uvicorn.run(app, host="0.0.0.0", port=8000)
