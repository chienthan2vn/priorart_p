/**
 * Frontend JavaScript for Patent AI Agent
 * Handles all API interactions and UI updates
 */

// API Base URL
const API_BASE = '';

// Utility functions
function showLoading(elementId) {
    document.getElementById(elementId).style.display = 'block';
}

function hideLoading(elementId) {
    document.getElementById(elementId).style.display = 'none';
}

function showAlert(message, type = 'info', containerId = null) {
    const alertHtml = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            <strong>${type.charAt(0).toUpperCase() + type.slice(1)}!</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    if (containerId) {
        document.getElementById(containerId).innerHTML = alertHtml + document.getElementById(containerId).innerHTML;
    } else {
        // Show at the top of the current tab
        const activeTab = document.querySelector('.tab-pane.active');
        if (activeTab) {
            activeTab.insertAdjacentHTML('afterbegin', alertHtml);
        }
    }
}

function formatTimestamp(timestamp) {
    return new Date(timestamp).toLocaleString();
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Load example data
    loadExampleData();
    
    // Set up form event listeners
    setupFormHandlers();
    
    // Load sessions when sessions tab is activated
    document.getElementById('sessions-tab').addEventListener('click', loadSessions);
    
    // Check system health
    checkSystemHealth();
});

function loadExampleData() {
    // Load example data into the extraction form
    document.getElementById('ideaTitle').value = 'Smart Irrigation System with IoT Sensors';
    document.getElementById('userScenario').value = 'A farmer managing a large agricultural field needs to optimize water usage while ensuring crops receive adequate moisture. The farmer wants to monitor soil conditions remotely and automatically adjust irrigation based on real-time data from multiple field locations.';
    document.getElementById('userProblem').value = 'Traditional irrigation systems either over-water or under-water crops because they operate on fixed schedules without considering actual soil moisture, weather conditions, or crop-specific needs. This leads to water waste, increased costs, and potentially reduced crop yields.';
}

function setupFormHandlers() {
    // Keyword Extraction Form
    document.getElementById('extractionForm').addEventListener('submit', handleExtraction);
    
    // IPC Classification Form
    document.getElementById('ipcForm').addEventListener('submit', handleIPCClassification);
    
    // Patent Analysis Form
    document.getElementById('patentForm').addEventListener('submit', handlePatentAnalysis);
    
    // Similarity Evaluation Form
    document.getElementById('similarityForm').addEventListener('submit', handleSimilarityEvaluation);
}

async function checkSystemHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        // Show system status in console
        console.log('System Health:', data);
        
        // Show warning if any components are not initialized
        const components = data.components;
        const failedComponents = Object.entries(components)
            .filter(([key, value]) => !value)
            .map(([key]) => key);
            
        if (failedComponents.length > 0) {
            showAlert(`Warning: Some components failed to initialize: ${failedComponents.join(', ')}`, 'warning');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        showAlert('Failed to connect to backend server. Please check if the server is running.', 'danger');
    }
}

// Keyword Extraction Functions
async function handleExtraction(event) {
    event.preventDefault();
    
    const title = document.getElementById('ideaTitle').value;
    const scenario = document.getElementById('userScenario').value;
    const problem = document.getElementById('userProblem').value;
    const autoMode = document.getElementById('autoMode').checked;
    
    if (!title || !scenario || !problem) {
        showAlert('Please fill in all fields', 'warning', 'extractionResults');
        return;
    }
    
    // Format input text
    const inputText = `**Idea title**: ${title}

**User scenario**: ${scenario}

**User problem**: ${problem}`;
    
    const requestData = {
        input_text: inputText,
        use_auto_mode: autoMode
    };
    
    showLoading('extractionLoading');
    document.getElementById('extractionResults').innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE}/api/extract/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Extraction failed');
        }
        
        hideLoading('extractionLoading');
        displayExtractionResults(data);
        
    } catch (error) {
        hideLoading('extractionLoading');
        showAlert(`Extraction failed: ${error.message}`, 'danger', 'extractionResults');
    }
}

function displayExtractionResults(data) {
    const resultsContainer = document.getElementById('extractionResults');
    
    if (data.status === 'completed') {
        // Auto mode - show complete results
        const results = data.results;
        
        let html = `
            <div class="result-card">
                <h5 class="text-success mb-3"><i class="bi bi-check-circle"></i> Extraction Completed</h5>
                <p><strong>Session ID:</strong> ${data.session_id}</p>
                <p><strong>Status:</strong> <span class="status-badge status-success">${data.status}</span></p>
        `;
        
        // Display concept matrix
        if (results.concept_matrix) {
            html += `
                <div class="mt-4">
                    <h6><i class="bi bi-diagram-3"></i> Concept Matrix</h6>
                    <div class="row">
                        <div class="col-md-4">
                            <strong>Problem/Purpose:</strong><br>
                            <small class="text-muted">${results.concept_matrix.problem_purpose}</small>
                        </div>
                        <div class="col-md-4">
                            <strong>Object/System:</strong><br>
                            <small class="text-muted">${results.concept_matrix.object_system}</small>
                        </div>
                        <div class="col-md-4">
                            <strong>Environment/Field:</strong><br>
                            <small class="text-muted">${results.concept_matrix.environment_field}</small>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Display seed keywords
        if (results.seed_keywords) {
            html += `
                <div class="mt-4">
                    <h6><i class="bi bi-tags"></i> Seed Keywords</h6>
                    <div class="row">
            `;
            
            Object.entries(results.seed_keywords).forEach(([category, keywords]) => {
                html += `
                    <div class="col-md-4 mb-3">
                        <strong>${category.replace('_', ' ').toUpperCase()}:</strong><br>
                `;
                keywords.forEach(keyword => {
                    html += `<span class="keyword-badge">${keyword}</span>`;
                });
                html += `</div>`;
            });
            
            html += `</div></div>`;
        }
        
        // Display enhanced keywords
        if (results.final_keywords) {
            html += `
                <div class="mt-4">
                    <h6><i class="bi bi-stars"></i> Enhanced Keywords</h6>
                    <div class="accordion" id="keywordAccordion">
            `;
            
            Object.entries(results.final_keywords).forEach(([keyword, synonyms], index) => {
                html += `
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="heading${index}">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse${index}">
                                ${keyword} (${synonyms.length} synonyms)
                            </button>
                        </h2>
                        <div id="collapse${index}" class="accordion-collapse collapse" data-bs-parent="#keywordAccordion">
                            <div class="accordion-body">
                `;
                synonyms.forEach(synonym => {
                    html += `<span class="keyword-badge">${synonym}</span>`;
                });
                html += `</div></div></div>`;
            });
            
            html += `</div></div>`;
        }
        
        // Display search queries
        if (results.queries && results.queries.queries) {
            html += `
                <div class="mt-4">
                    <h6><i class="bi bi-search"></i> Generated Search Queries</h6>
                    <ol class="list-group list-group-numbered">
            `;
            results.queries.queries.forEach(query => {
                html += `<li class="list-group-item">${query}</li>`;
            });
            html += `</ol></div>`;
        }
        
        // Download button
        if (data.filename) {
            html += `
                <div class="mt-4">
                    <a href="${API_BASE}/api/download/${data.filename}" class="btn btn-outline-primary">
                        <i class="bi bi-download"></i> Download Results (JSON)
                    </a>
                </div>
            `;
        }
        
        html += `</div>`;
        
    } else {
        // Manual mode - show session info
        html = `
            <div class="result-card">
                <h5 class="text-info mb-3"><i class="bi bi-hourglass-split"></i> Extraction Started</h5>
                <p><strong>Session ID:</strong> ${data.session_id}</p>
                <p><strong>Status:</strong> <span class="status-badge status-warning">${data.status}</span></p>
                <p>${data.message}</p>
                <div class="mt-3">
                    <button class="btn btn-info btn-sm" onclick="checkExtractionStatus('${data.session_id}')">
                        <i class="bi bi-arrow-clockwise"></i> Check Status
                    </button>
                </div>
            </div>
        `;
    }
    
    resultsContainer.innerHTML = html;
}

async function checkExtractionStatus(sessionId) {
    try {
        const response = await fetch(`${API_BASE}/api/extract/status/${sessionId}`);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to get status');
        }
        
        showAlert(`Session ${sessionId}: ${data.status}`, 'info');
        
    } catch (error) {
        showAlert(`Status check failed: ${error.message}`, 'danger');
    }
}

// IPC Classification Functions
async function handleIPCClassification(event) {
    event.preventDefault();
    
    const text = document.getElementById('ipcText').value;
    
    if (!text.trim()) {
        showAlert('Please enter text for IPC classification', 'warning', 'ipcResults');
        return;
    }
    
    showLoading('ipcLoading');
    document.getElementById('ipcResults').innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE}/api/ipc/classify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'IPC classification failed');
        }
        
        hideLoading('ipcLoading');
        displayIPCResults(data);
        
    } catch (error) {
        hideLoading('ipcLoading');
        showAlert(`IPC classification failed: ${error.message}`, 'danger', 'ipcResults');
    }
}

function displayIPCResults(data) {
    const resultsContainer = document.getElementById('ipcResults');
    
    let html = `
        <div class="result-card">
            <h5 class="text-success mb-3"><i class="bi bi-tags"></i> IPC Classification Results</h5>
            <p><strong>Processed at:</strong> ${formatTimestamp(data.timestamp)}</p>
    `;
    
    if (data.predictions && data.predictions.length > 0) {
        html += `
            <div class="mt-3">
                <h6>Top Predictions:</h6>
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>IPC Category</th>
                                <th>Score</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        data.predictions.forEach(pred => {
            html += `
                <tr>
                    <td><span class="badge bg-primary">${pred.rank}</span></td>
                    <td><code>${pred.category}</code></td>
                    <td>
                        <div class="progress" style="width: 100px;">
                            <div class="progress-bar" role="progressbar" style="width: ${pred.score}%"></div>
                        </div>
                        <small>${pred.score}%</small>
                    </td>
                </tr>
            `;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    } else {
        html += `<p class="text-muted">No predictions returned.</p>`;
    }
    
    html += `</div>`;
    resultsContainer.innerHTML = html;
}

// Patent Analysis Functions
async function handlePatentAnalysis(event) {
    event.preventDefault();
    
    const url = document.getElementById('patentUrl').value;
    
    if (!url.trim()) {
        showAlert('Please enter a patent URL', 'warning', 'patentResults');
        return;
    }
    
    if (!url.includes('patents.google.com')) {
        showAlert('Please enter a valid Google Patents URL', 'warning', 'patentResults');
        return;
    }
    
    showLoading('patentLoading');
    document.getElementById('patentResults').innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE}/api/patent/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Patent analysis failed');
        }
        
        hideLoading('patentLoading');
        displayPatentResults(data);
        
    } catch (error) {
        hideLoading('patentLoading');
        showAlert(`Patent analysis failed: ${error.message}`, 'danger', 'patentResults');
    }
}

function displayPatentResults(data) {
    const resultsContainer = document.getElementById('patentResults');
    
    let html = `
        <div class="result-card">
            <h5 class="text-success mb-3"><i class="bi bi-file-earmark-text"></i> Patent Analysis Results</h5>
            <p><strong>URL:</strong> <a href="${data.url}" target="_blank">${data.url}</a></p>
            <p><strong>Analyzed at:</strong> ${formatTimestamp(data.timestamp)}</p>
    `;
    
    const patent = data.patent_info;
    
    if (patent) {
        // Title
        html += `
            <div class="mt-3">
                <h6><i class="bi bi-card-heading"></i> Title</h6>
                <p class="border-start border-primary ps-3">${patent.title}</p>
            </div>
        `;
        
        // Abstract
        if (patent.abstract && !patent.abstract.includes('not found')) {
            html += `
                <div class="mt-3">
                    <h6><i class="bi bi-file-text"></i> Abstract</h6>
                    <div class="border-start border-success ps-3">
                        <p class="text-muted">${patent.abstract.substring(0, 500)}${patent.abstract.length > 500 ? '...' : ''}</p>
                        ${patent.abstract.length > 500 ? `
                            <button class="btn btn-sm btn-outline-secondary" onclick="toggleFullText('abstract')">
                                Show Full Text
                            </button>
                            <div id="abstract-full" style="display: none;">
                                <p class="text-muted mt-2">${patent.abstract}</p>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        }
        
        // Claims
        if (patent.claims && !patent.claims.includes('not found')) {
            html += `
                <div class="mt-3">
                    <h6><i class="bi bi-list-ol"></i> Claims</h6>
                    <div class="border-start border-warning ps-3">
                        <p class="text-muted">${patent.claims.substring(0, 500)}${patent.claims.length > 500 ? '...' : ''}</p>
                        ${patent.claims.length > 500 ? `
                            <button class="btn btn-sm btn-outline-secondary" onclick="toggleFullText('claims')">
                                Show Full Text
                            </button>
                            <div id="claims-full" style="display: none;">
                                <p class="text-muted mt-2">${patent.claims}</p>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        }
        
        // Description
        if (patent.description && !patent.description.includes('not found')) {
            html += `
                <div class="mt-3">
                    <h6><i class="bi bi-journal-text"></i> Description</h6>
                    <div class="border-start border-info ps-3">
                        <p class="text-muted">${patent.description.substring(0, 500)}${patent.description.length > 500 ? '...' : ''}</p>
                        ${patent.description.length > 500 ? `
                            <button class="btn btn-sm btn-outline-secondary" onclick="toggleFullText('description')">
                                Show Full Text
                            </button>
                            <div id="description-full" style="display: none;">
                                <p class="text-muted mt-2">${patent.description}</p>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        }
    } else {
        html += `<p class="text-muted">No patent information could be extracted.</p>`;
    }
    
    html += `</div>`;
    resultsContainer.innerHTML = html;
}

function toggleFullText(elementId) {
    const element = document.getElementById(`${elementId}-full`);
    const button = element.previousElementSibling;
    
    if (element.style.display === 'none') {
        element.style.display = 'block';
        button.textContent = 'Show Less';
    } else {
        element.style.display = 'none';
        button.textContent = 'Show Full Text';
    }
}

// Similarity Evaluation Functions
async function handleSimilarityEvaluation(event) {
    event.preventDefault();
    
    const text1 = document.getElementById('text1').value;
    const text2 = document.getElementById('text2').value;
    
    if (!text1.trim() || !text2.trim()) {
        showAlert('Please enter both texts for similarity evaluation', 'warning', 'similarityResults');
        return;
    }
    
    showLoading('similarityLoading');
    document.getElementById('similarityResults').innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE}/api/similarity/evaluate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text1: text1, text2: text2 })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Similarity evaluation failed');
        }
        
        hideLoading('similarityLoading');
        displaySimilarityResults(data);
        
    } catch (error) {
        hideLoading('similarityLoading');
        showAlert(`Similarity evaluation failed: ${error.message}`, 'danger', 'similarityResults');
    }
}

function displaySimilarityResults(data) {
    const resultsContainer = document.getElementById('similarityResults');
    
    const scores = data.similarity_scores;
    
    let html = `
        <div class="result-card">
            <h5 class="text-success mb-3"><i class="bi bi-bar-chart"></i> Similarity Evaluation Results</h5>
            <p><strong>Evaluated at:</strong> ${formatTimestamp(data.timestamp)}</p>
            
            <div class="row mt-4">
                <div class="col-md-6">
                    <h6>Text 1 (Preview)</h6>
                    <p class="text-muted small border-start border-primary ps-2">${data.text1}</p>
                </div>
                <div class="col-md-6">
                    <h6>Text 2 (Preview)</h6>
                    <p class="text-muted small border-start border-secondary ps-2">${data.text2}</p>
                </div>
            </div>
            
            <div class="mt-4">
                <h6>Similarity Scores</h6>
                <div class="row">
    `;
    
    // Similarity Score
    if (scores.similarities_score !== undefined) {
        const percentage = (scores.similarities_score * 100).toFixed(1);
        html += `
            <div class="col-md-4 mb-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title text-primary">${percentage}%</h5>
                        <p class="card-text">Cosine Similarity</p>
                        <div class="progress">
                            <div class="progress-bar bg-primary" style="width: ${percentage}%"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Rerank Score
    if (scores.rerank_score !== undefined) {
        const normalizedScore = Math.max(0, Math.min(100, (scores.rerank_score + 10) * 5)); // Normalize for display
        html += `
            <div class="col-md-4 mb-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title text-success">${scores.rerank_score.toFixed(3)}</h5>
                        <p class="card-text">BGE Rerank Score</p>
                        <div class="progress">
                            <div class="progress-bar bg-success" style="width: ${normalizedScore}%"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    // LLM Score
    if (scores.llm_score !== undefined) {
        const percentage = (scores.llm_score * 100).toFixed(1);
        html += `
            <div class="col-md-4 mb-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title text-warning">${percentage}%</h5>
                        <p class="card-text">LLM Rerank Score</p>
                        <div class="progress">
                            <div class="progress-bar bg-warning" style="width: ${percentage}%"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    html += `
                </div>
            </div>
        </div>
    `;
    
    resultsContainer.innerHTML = html;
}

// Session Management Functions
async function loadSessions() {
    showLoading('sessionsLoading');
    document.getElementById('sessionsResults').innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE}/api/sessions`);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to load sessions');
        }
        
        hideLoading('sessionsLoading');
        displaySessions(data);
        
    } catch (error) {
        hideLoading('sessionsLoading');
        showAlert(`Failed to load sessions: ${error.message}`, 'danger', 'sessionsResults');
    }
}

function displaySessions(data) {
    const resultsContainer = document.getElementById('sessionsResults');
    
    let html = `
        <div class="result-card">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h5 class="text-info mb-0"><i class="bi bi-collection"></i> Active Sessions</h5>
                <span class="badge bg-info">${data.total} sessions</span>
            </div>
            <p><strong>Last updated:</strong> ${formatTimestamp(data.timestamp)}</p>
    `;
    
    if (data.sessions && data.sessions.length > 0) {
        html += `
            <div class="table-responsive">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Session ID</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        data.sessions.forEach(sessionId => {
            html += `
                <tr>
                    <td><code>${sessionId}</code></td>
                    <td>
                        <button class="btn btn-sm btn-outline-info me-2" onclick="checkExtractionStatus('${sessionId}')">
                            <i class="bi bi-info-circle"></i> Status
                        </button>
                        <button class="btn btn-sm btn-outline-danger" onclick="deleteSession('${sessionId}')">
                            <i class="bi bi-trash"></i> Delete
                        </button>
                    </td>
                </tr>
            `;
        });
        
        html += `
                    </tbody>
                </table>
            </div>
        `;
    } else {
        html += `<p class="text-muted">No active sessions found.</p>`;
    }
    
    html += `</div>`;
    resultsContainer.innerHTML = html;
}

async function deleteSession(sessionId) {
    if (!confirm(`Are you sure you want to delete session ${sessionId}?`)) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/sessions/${sessionId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to delete session');
        }
        
        showAlert(`Session ${sessionId} deleted successfully`, 'success');
        loadSessions(); // Refresh the sessions list
        
    } catch (error) {
        showAlert(`Failed to delete session: ${error.message}`, 'danger');
    }
}
