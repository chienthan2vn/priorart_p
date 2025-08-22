const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(express.static('public'));

// Store active extraction sessions
const activeSessions = new Map();

// Routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// API endpoint to start extraction
app.post('/api/extract', (req, res) => {
  const { problem, technical } = req.body;
  
  if (!problem || !technical) {
    return res.status(400).json({ error: 'Both problem and technical fields are required' });
  }

  const sessionId = Date.now().toString();
  const inputText = `Problem: ${problem}\nTechnical: ${technical}`;
  
  // Store session data
  activeSessions.set(sessionId, {
    inputText,
    problem,
    technical,
    status: 'processing',
    results: null
  });

  // Start Python extraction process
  startExtractionProcess(sessionId, inputText);
  
  res.json({ sessionId, status: 'started' });
});

// API endpoint to get session status
app.get('/api/session/:sessionId', (req, res) => {
  const { sessionId } = req.params;
  const session = activeSessions.get(sessionId);
  
  if (!session) {
    return res.status(404).json({ error: 'Session not found' });
  }
  
  res.json(session);
});

// API endpoint to handle human evaluation
app.post('/api/evaluate/:sessionId', (req, res) => {
  const { sessionId } = req.params;
  const { action, editedKeywords, feedback } = req.body;
  
  const session = activeSessions.get(sessionId);
  if (!session) {
    return res.status(404).json({ error: 'Session not found' });
  }
  
  // Send evaluation response to Python process
  if (session.pythonProcess) {
    const response = {
      action,
      editedKeywords,
      feedback
    };
    
    session.pythonProcess.stdin.write(JSON.stringify(response) + '\n');
  }
  
  res.json({ status: 'evaluation_sent' });
});

function startExtractionProcess(sessionId, inputText) {
  const session = activeSessions.get(sessionId);
  
  // Create a Python script that uses CoreConceptExtractor with web_mode=True
  const pythonScript = `
import sys
import json
import os
sys.path.append('${process.cwd()}')
from src.core.extractor import CoreConceptExtractor

def main():
    input_text = """${inputText.replace(/"/g, '\\"')}"""
    
    try:
        # Initialize extractor with web_mode=True
        extractor = CoreConceptExtractor(web_mode=True)
        
        # Send initial status
        print(json.dumps({"type": "status", "message": "Starting extraction..."}))
        sys.stdout.flush()
        
        # Run extraction
        result = extractor.extract_keywords(input_text)
        
        # Send final results
        serialized_result = {}
        for key, value in result.items():
            if hasattr(value, 'dict'):
                serialized_result[key] = value.dict()
            elif hasattr(value, 'summary'):  # Handle SummaryResponse object
                serialized_result[key] = value.summary
            elif isinstance(value, list):
                serialized_result[key] = [
                    item.dict() if hasattr(item, 'dict') else item 
                    for item in value
                ]
            else:
                serialized_result[key] = value
        
        output = {
            "type": "results",
            "data": serialized_result
        }
        print(json.dumps(output))
        sys.stdout.flush()
        
    except Exception as e:
        import traceback
        error_output = {
            "type": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_output))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
`;

  // Write temporary Python script
  const fs = require('fs');
  const tempScriptPath = `temp_web_extraction_${sessionId}.py`;
  fs.writeFileSync(tempScriptPath, pythonScript);
  
  // Spawn Python process
  const pythonProcess = spawn('python', [tempScriptPath], {
    stdio: ['pipe', 'pipe', 'pipe']
  });
  
  session.pythonProcess = pythonProcess;
  session.awaitingHumanEvaluation = false;
  
  let dataBuffer = '';
  
  pythonProcess.stdout.on('data', (data) => {
    dataBuffer += data.toString();
    
    // Process complete JSON messages
    const lines = dataBuffer.split('\n');
    dataBuffer = lines.pop(); // Keep incomplete line in buffer
    
    lines.forEach(line => {
      if (line.trim()) {
        try {
          const output = JSON.parse(line.trim());
          
          if (output.type === 'status') {
            console.log('Python status:', output.message);
          } else if (output.type === 'human_evaluation_needed') {
            session.results = output.data;
            session.status = 'awaiting_evaluation';
            session.awaitingHumanEvaluation = true;
            
            // Emit to connected clients
            io.emit('extraction_ready', {
              sessionId,
              results: output.data
            });
          } else if (output.type === 'results') {
            session.results = output.data;
            session.status = 'completed';
            session.awaitingHumanEvaluation = false;
            
            // Emit completion to connected clients
            io.emit('extraction_completed', {
              sessionId,
              results: output.data
            });
          } else if (output.type === 'error') {
            session.status = 'error';
            session.error = output.message;
            session.awaitingHumanEvaluation = false;
            
            io.emit('extraction_error', {
              sessionId,
              error: output.message
            });
          }
        } catch (e) {
          console.log('Python output (non-JSON):', line);
        }
      }
    });
  });
  
  pythonProcess.stderr.on('data', (data) => {
    console.error('Python stderr:', data.toString());
  });
  
  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
    
    if (session.status === 'awaiting_evaluation') {
      // Process is waiting for human input, don't mark as completed yet
      return;
    }
    
    if (code !== 0 && session.status !== 'error') {
      session.status = 'error';
      session.error = 'Python process exited unexpectedly';
      io.emit('extraction_error', {
        sessionId,
        error: 'Python process exited unexpectedly'
      });
    }
  });
}

// Socket.io for real-time communication
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`🚀 Patent AI Web Interface running on http://localhost:${PORT}`);
});
