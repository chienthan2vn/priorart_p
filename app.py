"""
FastAPI Backend for Patent AI Agent
Provides REST API endpoints for the patent keyword extraction system
"""

import os
import json
import datetime
import asyncio
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Import the core system
from src.core.extractor import CoreConceptExtractor
from src.api.ipc_classifier import get_ipc_predictions
from src.crawling.patent_crawler import PatentCrawler
from src.evaluation.similarity_evaluator import PatentSimilarityEvaluator
from config.settings import settings

# Global variables for system components
extractor = None
patent_crawler = None
similarity_evaluator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global extractor, patent_crawler, similarity_evaluator
    print("🚀 Initializing Patent AI Agent components...")
    
    try:
        extractor = CoreConceptExtractor()
        patent_crawler = PatentCrawler()
        similarity_evaluator = PatentSimilarityEvaluator()
        print("✅ All components initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        # Continue without crashing
    
    yield
    
    # Shutdown
    print("🔄 Shutting down Patent AI Agent...")

# Initialize FastAPI app
app = FastAPI(
    title="Patent AI Agent API",
    description="AI-powered patent keyword extraction and prior art search system",
    version="1.0.0",
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

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for API
class PatentIdeaInput(BaseModel):
    """Input model for patent idea text"""
    title: str = Field(..., description="Title of the patent idea")
    scenario: str = Field(..., description="User scenario description")
    problem: str = Field(..., description="Problem the invention solves")

class ExtractionRequest(BaseModel):
    """Request model for keyword extraction"""
    input_text: str = Field(..., description="Patent idea text to process")
    use_auto_mode: bool = Field(default=False, description="Skip human validation steps")

class ValidationRequest(BaseModel):
    """Request model for human validation"""
    session_id: str = Field(..., description="Session ID for the extraction")
    action: str = Field(..., description="Action: 'approve', 'reject', or 'edit'")
    feedback: Optional[str] = Field(None, description="Feedback text for rejection")
    edited_keywords: Optional[Dict[str, List[str]]] = Field(None, description="Manually edited keywords")

class IPCClassificationRequest(BaseModel):
    """Request model for IPC classification"""
    text: str = Field(..., description="Text to classify")

class PatentUrlRequest(BaseModel):
    """Request model for patent URL analysis"""
    url: str = Field(..., description="Patent URL to analyze")

class SimilarityRequest(BaseModel):
    """Request model for similarity evaluation"""
    text1: str = Field(..., description="First text for comparison")
    text2: str = Field(..., description="Second text for comparison")

# In-memory storage for sessions (in production, use Redis or database)
active_sessions: Dict[str, Dict[str, Any]] = {}

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
            <head><title>Patent AI Agent</title></head>
            <body>
                <h1>Patent AI Agent API</h1>
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
        "timestamp": datetime.datetime.now().isoformat(),
        "components": {
            "extractor": extractor is not None,
            "patent_crawler": patent_crawler is not None,
            "similarity_evaluator": similarity_evaluator is not None
        }
    }

@app.get("/api/settings")
async def get_settings():
    """Get current system settings"""
    return {
        "model_name": settings.DEFAULT_MODEL_NAME,
        "temperature": settings.MODEL_TEMPERATURE,
        "max_search_results": settings.MAX_SEARCH_RESULTS,
        "api_keys_status": settings.validate_api_keys()
    }

@app.post("/api/extract/start")
async def start_extraction(request: ExtractionRequest):
    """Start the patent keyword extraction process"""
    if not extractor:
        raise HTTPException(status_code=503, detail="Extractor not initialized")
    
    try:
        # Generate session ID
        session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(active_sessions)}"
        
        if request.use_auto_mode:
            # Run full extraction without human intervention
            results = await asyncio.to_thread(extractor.extract_keywords, request.input_text)
            
            # Save results
            filename = f"extraction_results_{session_id}.json"
            with open(filename, "w", encoding="utf-8") as f:
                def serialize(obj):
                    if hasattr(obj, "dict"):
                        return obj.dict()
                    return obj
                json.dump({k: serialize(v) for k, v in results.items()}, f, indent=2, ensure_ascii=False)
            
            return {
                "session_id": session_id,
                "status": "completed",
                "results": results,
                "filename": filename
            }
        else:
            # Store session for human-in-the-loop processing
            active_sessions[session_id] = {
                "input_text": request.input_text,
                "status": "started",
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Start processing in background
            # For now, return session info - full implementation would use background tasks
            return {
                "session_id": session_id,
                "status": "started",
                "message": "Extraction started. Use /api/extract/status to check progress."
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.get("/api/extract/status/{session_id}")
async def get_extraction_status(session_id: str):
    """Get the status of an extraction session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return active_sessions[session_id]

@app.post("/api/extract/validate")
async def validate_extraction(request: ValidationRequest):
    """Handle human validation of extraction results"""
    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[request.session_id]
    
    try:
        if request.action == "approve":
            session["status"] = "approved"
            session["validation"] = "approved"
        elif request.action == "reject":
            session["status"] = "rejected"
            session["validation"] = "rejected"
            session["feedback"] = request.feedback
        elif request.action == "edit":
            session["status"] = "edited"
            session["validation"] = "edited"
            session["edited_keywords"] = request.edited_keywords
        
        return {"status": "success", "session_updated": True}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/api/ipc/classify")
async def classify_ipc(request: IPCClassificationRequest):
    """Classify text using IPC classification API"""
    try:
        predictions = get_ipc_predictions(request.text)
        return {
            "text": request.text,
            "predictions": predictions,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IPC classification failed: {str(e)}")

@app.post("/api/patent/analyze")
async def analyze_patent_url(request: PatentUrlRequest):
    """Analyze a patent from its URL"""
    if not patent_crawler:
        raise HTTPException(status_code=503, detail="Patent crawler not initialized")
    
    try:
        patent_info = await asyncio.to_thread(patent_crawler.extract_patent_info, request.url)
        return {
            "url": request.url,
            "patent_info": patent_info,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patent analysis failed: {str(e)}")

@app.post("/api/similarity/evaluate")
async def evaluate_similarity(request: SimilarityRequest):
    """Evaluate similarity between two texts"""
    if not similarity_evaluator:
        raise HTTPException(status_code=503, detail="Similarity evaluator not initialized")
    
    try:
        scores = await asyncio.to_thread(similarity_evaluator.evaluate_similarity, request.text1, request.text2)
        return {
            "text1": request.text1[:100] + "..." if len(request.text1) > 100 else request.text1,
            "text2": request.text2[:100] + "..." if len(request.text2) > 100 else request.text2,
            "similarity_scores": scores,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity evaluation failed: {str(e)}")

@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "sessions": list(active_sessions.keys()),
        "total": len(active_sessions),
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del active_sessions[session_id]
    return {"status": "success", "message": f"Session {session_id} deleted"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
