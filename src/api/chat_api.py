"""
Simple chat API for testing the RAG system.
This is like creating a phone line that your chat interface can call!
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from loguru import logger
import openai

# Add src to path so we can import our RAG system
sys.path.append(str(Path(__file__).parent.parent))

from rag.pipeline import RAGPipeline


# Data models for our API (like forms that define what data we expect)
class ChatMessage(BaseModel):
    message: str
    max_results: int = 5
    similarity_threshold: float = 0.1
    include_sources: bool = True
    use_ai: bool = True  # Whether to use OpenAI for response generation


class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    query_time: float
    total_sources: int
    success: bool
    error: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    documents_processed: int
    chunks_created: int


class WebsiteIngestRequest(BaseModel):
    urls: List[str]
    max_depth: int = 2
    max_pages_per_site: int = 50
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None


class WebsiteIngestResponse(BaseModel):
    success: bool
    message: str
    websites_processed: int
    pages_scraped: int
    chunks_created: int


class SystemStats(BaseModel):
    total_documents: int
    vector_store_type: str
    embedding_model: str
    system_ready: bool
    unique_urls: int = 0
    openai_available: bool = False


class APIStatus(BaseModel):
    openai_configured: bool
    openai_model: str = "gpt-4o-mini"
    rag_system_ready: bool


# Create our FastAPI app (this is like setting up a restaurant!)
app = FastAPI(
    title="RAG Chat System",
    description="A simple chat interface for testing your RAG system with company documents",
    version="1.0.0"
)

# Allow web browsers to talk to our API (CORS = Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance (our smart assistant!)
rag_pipeline: Optional[RAGPipeline] = None

# Initialize OpenAI client
openai_client = None

def initialize_openai():
    """Initialize OpenAI client if API key is available"""
    global openai_client
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        openai_client = openai.OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully")
    else:
        logger.warning("OPENAI_API_KEY not found in environment variables")

async def generate_ai_response(query: str, sources: List[Dict[str, Any]], fallback_message: str = None) -> str:
    """Generate AI response using OpenAI with RAG context"""
    if not openai_client:
        return fallback_message or "AI response generation is not available (missing API key)"
    
    try:
        # Prepare context from retrieved documents (more defensive)
        context_parts = []
        for i, source in enumerate(sources[:3]):  # Use top 3 sources
            try:
                content = str(source.get('content', source.get('snippet', '')))[:400]  # Shorter limit
                url = str(source.get('url', 'Unknown source'))[:100]  # Limit URL length
                if content.strip():
                    context_parts.append(f"Document {i+1}:\n{content}")
            except Exception as source_e:
                logger.warning(f"Error processing source {i}: {source_e}")
                continue
        
        context = "\n\n".join(context_parts) if context_parts else "No specific information found."
        
        # Simpler, shorter prompts to avoid token issues
        system_prompt = "You are a helpful assistant for Nupo. Answer based on the provided context. Be concise and professional."

        user_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

        # Log the final prompt being sent to OpenAI
        logger.info("=" * 80)
        logger.info("🤖 SENDING TO OPENAI:")
        logger.info(f"System Prompt: {system_prompt}")
        logger.info("-" * 40)
        logger.info(f"User Prompt: {user_prompt}")
        logger.info("=" * 80)
        
        # Generate AI response using OpenAI with retrieved context
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        
        logger.info("✅ OpenAI API call completed successfully")
        
        # More defensive response extraction
        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                logger.info("🤖 OPENAI RESPONSE:")
                logger.info(f"Response: {str(content).strip()}")
                logger.info("=" * 80)
                return str(content).strip()
        
        return fallback_message or "I found information but couldn't generate a response."
        
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return fallback_message or "I found relevant information in your documents, but couldn't generate an AI response at this time."


def initialize_rag():
    """Initialize the RAG pipeline (wake up our smart assistant!)"""
    global rag_pipeline
    try:
        logger.info("🚀 Initializing RAG pipeline with semantic embeddings...")
        rag_pipeline = RAGPipeline()
        logger.info("RAG pipeline initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """This runs when we start our API server"""
    logger.info("Starting RAG Chat API...")
    success = initialize_rag()
    initialize_openai()  # Initialize OpenAI client
    if not success:
        logger.error("Failed to initialize RAG system!")


@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the chat interface HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🤖 RAG Chat System</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
                display: grid;
                grid-template-columns: 1fr 2fr;
                min-height: 80vh;
            }
            
            .sidebar {
                background: #f8f9fa;
                padding: 30px;
                border-right: 1px solid #e9ecef;
            }
            
            .main-chat {
                display: flex;
                flex-direction: column;
                height: 80vh;
            }
            
            .chat-header {
                background: #4a90e2;
                color: white;
                padding: 20px;
                text-align: center;
            }
            
            .chat-messages {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                background: #fafafa;
            }
            
            .message {
                margin-bottom: 20px;
                padding: 15px;
                border-radius: 15px;
                max-width: 80%;
                word-wrap: break-word;
            }
            
            .user-message {
                background: #4a90e2;
                color: white;
                margin-left: auto;
            }
            
            .bot-message {
                background: white;
                border: 1px solid #e9ecef;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .sources {
                margin-top: 10px;
                padding-top: 10px;
                border-top: 1px solid #e9ecef;
                font-size: 0.9em;
                color: #666;
            }
            
            .source-item {
                background: #e3f2fd;
                padding: 8px;
                margin: 5px 0;
                border-radius: 8px;
                border-left: 4px solid #2196f3;
            }
            
            .chat-input {
                padding: 20px;
                background: white;
                border-top: 1px solid #e9ecef;
                display: flex;
                gap: 10px;
            }
            
            .chat-input input {
                flex: 1;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 25px;
                outline: none;
                font-size: 16px;
            }
            
            .chat-input button {
                background: #4a90e2;
                color: white;
                border: none;
                padding: 15px 25px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                transition: background 0.3s;
            }
            
            .chat-input button:hover {
                background: #357abd;
            }
            
            .upload-section {
                margin-bottom: 30px;
            }
            
            .upload-area {
                border: 2px dashed #4a90e2;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
            }
            
            .upload-area:hover {
                background: #f0f8ff;
                border-color: #357abd;
            }
            
            .stats {
                background: #e8f5e8;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
                color: #666;
            }
            
            .error {
                background: #ffe6e6;
                color: #d63031;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            
            .success {
                background: #e8f5e8;
                color: #00b894;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            
            h1, h2, h3 {
                color: #2d3436;
                margin-bottom: 15px;
            }
            
            .welcome-message {
                background: linear-gradient(135deg, #74b9ff, #0984e3);
                color: white;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 20px;
            }
            
            @media (max-width: 768px) {
                .container {
                    grid-template-columns: 1fr;
                    grid-template-rows: auto 1fr;
                }
                
                .sidebar {
                    order: 2;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="sidebar">
                <h2>📚 Document Manager</h2>
                
                <div class="stats" id="stats">
                    <h3>System Status</h3>
                    <p>Loading...</p>
                </div>
                
                <div class="upload-section">
                    <h3>📁 Upload Documents</h3>
                    <div class="upload-area" onclick="document.getElementById('file-input').click()">
                        <p>📁 Click to upload documents</p>
                        <p style="font-size: 0.9em; color: #666;">Supports: PDF, DOCX, TXT, MD, JSON</p>
                    </div>
                    <input type="file" id="file-input" multiple accept=".pdf,.docx,.txt,.md,.json" style="display: none;">
                </div>
                
                <div class="upload-section">
                    <h3>🕷️ Scrape Websites</h3>
                    <div style="margin-bottom: 10px;">
                        <textarea id="website-urls" placeholder="Enter website URLs (one per line)&#10;https://example.com&#10;https://docs.example.com" 
                                  style="width: 100%; height: 80px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-family: monospace;"></textarea>
                    </div>
                    <div style="margin-bottom: 10px;">
                        <label style="font-size: 0.9em;">Max Depth: <input type="number" id="max-depth" value="1" min="0" max="3" style="width: 60px;"></label>
                        <label style="font-size: 0.9em; margin-left: 15px;">Max Pages: <input type="number" id="max-pages" value="20" min="1" max="100" style="width: 60px;"></label>
                    </div>
                    <div style="margin-bottom: 10px; font-size: 0.85em; color: #666;">
                        ⚠️ <strong>Large sites:</strong> For sites with 500+ pages, use Max Depth=1 and Max Pages=50 to start
                    </div>
                    <button onclick="scrapeWebsites()" style="background: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
                        🕷️ Scrape Websites
                    </button>
                </div>
                
                <div id="upload-status"></div>
                
                <div style="margin-top: 30px;">
                    <h3>🔧 Settings</h3>
                    <label>Max Results: <input type="number" id="max-results" value="5" min="1" max="20"></label><br><br>
                    <label>Similarity Threshold: <input type="number" id="similarity-threshold" value="0.1" min="0" max="1" step="0.1"></label><br><br>
                    <label><input type="checkbox" id="include-sources" checked> Show Sources</label><br><br>
                    <label><input type="checkbox" id="use-ai" checked> Use AI for Responses</label>
                    <div id="api-status" style="margin-top: 10px; padding: 8px; border-radius: 4px; font-size: 12px;"></div>
                </div>
            </div>
            
            <div class="main-chat">
                <div class="chat-header">
                    <h1>🤖 RAG Chat Assistant</h1>
                    <p>Ask questions about your uploaded documents!</p>
                </div>
                
                <div class="chat-messages" id="chat-messages">
                    <div class="welcome-message">
                        <h3>👋 Welcome to your RAG Chat System!</h3>
                        <p>Upload some documents on the left, then ask me questions about them.</p>
                        <p>I'll search through your documents and give you smart answers with sources!</p>
                    </div>
                </div>
                
                <div class="chat-input">
                    <input type="text" id="message-input" placeholder="Ask a question about your documents..." onkeypress="if(event.key==='Enter') sendMessage()">
                    <button onclick="sendMessage()">Send 🚀</button>
                </div>
            </div>
        </div>

        <script>
            // JavaScript to make our chat interface interactive!
            
            async function loadStats() {
                try {
                    const [statsResponse, statusResponse] = await Promise.all([
                        fetch('/api/stats'),
                        fetch('/api/status')
                    ]);
                    const stats = await statsResponse.json();
                    const status = await statusResponse.json();
                    
                    document.getElementById('stats').innerHTML = `
                        <h3>📊 System Status</h3>
                        <p><strong>Status:</strong> ${stats.system_ready ? '✅ Ready' : '❌ Not Ready'}</p>
                        <p><strong>Documents:</strong> ${stats.total_documents}</p>
                        <p><strong>Unique URLs:</strong> ${stats.unique_urls}</p>
                        <p><strong>Vector Store:</strong> ${stats.vector_store_type}</p>
                        <p><strong>Model:</strong> ${stats.embedding_model.split('/').pop()}</p>
                        <p><strong>AI Responses:</strong> ${stats.openai_available ? '✅ Available' : '❌ Not configured'}</p>
                    `;
                    
                    // Update API status indicator
                    const apiStatusDiv = document.getElementById('api-status');
                    if (status.openai_configured) {
                        apiStatusDiv.innerHTML = '✅ OpenAI API configured';
                        apiStatusDiv.style.background = '#d4edda';
                        apiStatusDiv.style.color = '#155724';
                    } else {
                        apiStatusDiv.innerHTML = '⚠️ OpenAI API key not configured - using basic responses';
                        apiStatusDiv.style.background = '#fff3cd';
                        apiStatusDiv.style.color = '#856404';
                    }
                    
                } catch (error) {
                    document.getElementById('stats').innerHTML = `
                        <h3>📊 System Status</h3>
                        <p class="error">❌ Error loading stats</p>
                    `;
                    document.getElementById('api-status').innerHTML = '❌ Status check failed';
                    document.getElementById('api-status').style.background = '#f8d7da';
                }
            }
            
            async function sendMessage() {
                const input = document.getElementById('message-input');
                const message = input.value.trim();
                
                if (!message) return;
                
                // Add user message to chat
                addMessage(message, 'user');
                input.value = '';
                
                // Show loading
                const loadingDiv = addMessage('🤔 Thinking...', 'bot');
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            max_results: parseInt(document.getElementById('max-results').value),
                            similarity_threshold: parseFloat(document.getElementById('similarity-threshold').value),
                            include_sources: document.getElementById('include-sources').checked,
                            use_ai: document.getElementById('use-ai').checked
                        })
                    });
                    
                    const data = await response.json();
                    
                    // Remove loading message
                    loadingDiv.remove();
                    
                    if (data.success) {
                        let botMessage = data.response;
                        let sourcesHtml = '';
                        
                        if (data.include_sources && data.sources.length > 0) {
                            sourcesHtml = '<div class="sources"><strong>📚 Sources:</strong>';
                            data.sources.forEach((source, index) => {
                                sourcesHtml += `
                                    <div class="source-item">
                                        <strong>${source.filename || 'Unknown'}</strong> 
                                        (Score: ${(source.similarity_score || 0).toFixed(3)})
                                        <br><em>${source.snippet || 'No snippet available'}</em>
                                    </div>
                                `;
                            });
                            sourcesHtml += '</div>';
                        }
                        
                        addMessage(botMessage + sourcesHtml, 'bot');
                    } else {
                        addMessage(`❌ Error: ${data.error}`, 'bot');
                    }
                    
                } catch (error) {
                    loadingDiv.remove();
                    addMessage('❌ Sorry, there was an error processing your message.', 'bot');
                }
            }
            
            function addMessage(content, sender) {
                const messagesDiv = document.getElementById('chat-messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                messageDiv.innerHTML = content;
                
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                
                return messageDiv;
            }
            
            // File upload handling
            document.getElementById('file-input').addEventListener('change', async function(e) {
                const files = e.target.files;
                if (files.length === 0) return;
                
                const statusDiv = document.getElementById('upload-status');
                statusDiv.innerHTML = '<div class="loading">📤 Uploading and processing files...</div>';
                
                const formData = new FormData();
                for (let file of files) {
                    formData.append('files', file);
                }
                
                try {
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        statusDiv.innerHTML = `
                            <div class="success">
                                ✅ Success! Processed ${result.documents_processed} documents 
                                into ${result.chunks_created} searchable chunks.
                            </div>
                        `;
                        loadStats(); // Refresh stats
                    } else {
                        statusDiv.innerHTML = `<div class="error">❌ ${result.message}</div>`;
                    }
                    
                } catch (error) {
                    statusDiv.innerHTML = '<div class="error">❌ Error uploading files</div>';
                }
            });
            
            function scrapeWebsites() {
                var urls = document.getElementById('website-urls').value.trim().split(/\\n/);
                var maxDepth = document.getElementById('max-depth').value;
                var maxPages = document.getElementById('max-pages').value;
                var statusDiv = document.getElementById('upload-status');
                
                if (!urls[0]) {
                    alert('Please enter website URLs');
                    return;
                }
                
                statusDiv.innerHTML = '<div style="padding: 10px; background: #fff3cd; border-radius: 5px;">🕷️ Scraping websites...</div>';
                
                fetch('/api/ingest-websites', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        urls: urls,
                        max_depth: parseInt(maxDepth),
                        max_pages_per_site: parseInt(maxPages)
                    })
                })
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.success) {
                        statusDiv.innerHTML = '<div style="padding: 10px; background: #d4edda; border-radius: 5px;">✅ Success! Scraped ' + data.pages_scraped + ' pages.</div>';
                        document.getElementById('website-urls').value = '';
                        loadStats();
                    } else {
                        statusDiv.innerHTML = '<div style="padding: 10px; background: #f8d7da; border-radius: 5px;">❌ Error: ' + data.message + '</div>';
                    }
                })
                .catch(function(e) {
                    statusDiv.innerHTML = '<div style="padding: 10px; background: #f8d7da; border-radius: 5px;">⚠️ Check server logs - scraping may have worked despite this error.</div>';
                });
            }
            
            // Load stats on page load
            loadStats();
        </script>
    </body>
    </html>
    """
    return html_content


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_rag(message_data: ChatMessage):
    """
    Chat endpoint - this is where the magic happens!
    When someone asks a question, we search our documents and give a smart answer.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # Search for relevant documents using RAG pipeline
        search_result = rag_pipeline.search(
            query=message_data.message,
            top_k=message_data.max_results,
            threshold=message_data.similarity_threshold,
            rerank=True
        )
        
        if not search_result['success']:
            return ChatResponse(
                response="Sorry, I couldn't search the documents right now.",
                sources=[],
                query_time=search_result.get('search_time', 0),
                total_sources=0,
                success=False,
                error=search_result.get('error', 'Unknown error')
            )
        
        # Create a response based on the found documents
        sources = search_result['results']
        
        if not sources:
            if message_data.use_ai and os.getenv('OPENAI_API_KEY'):
                response_text = await generate_ai_response(
                    message_data.message, 
                    [], 
                    "I couldn't find specific information in your company documents."
                )
            else:
                response_text = """🤔 I couldn't find any relevant information in your documents for that question. 
                
Try:
- Asking about topics that are covered in your uploaded documents
- Using different keywords
- Lowering the similarity threshold in the settings"""
        else:
            if message_data.use_ai and os.getenv('OPENAI_API_KEY'):
                try:
                    response_text = await generate_ai_response(message_data.message, sources)
                except Exception as ai_error:
                    logger.error(f"OpenAI call failed: {ai_error}")
                    response_text = f"""📚 Found relevant information in your documents, but AI response failed.

Raw info: {sources[0].get('snippet', 'Information found in your documents.')[:200]}...

I found {len(sources)} relevant document(s)."""
            else:
                # Fallback to simple response
                response_text = f"""📚 Based on your documents, here's what I found:

{sources[0].get('snippet', 'Information found in your documents.')}

I found {len(sources)} relevant document(s) that might help answer your question."""
        
        return ChatResponse(
            response=response_text,
            sources=sources if message_data.include_sources else [],
            query_time=search_result['search_time'],
            total_sources=len(sources),
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(
            response="Sorry, there was an error processing your question.",
            sources=[],
            query_time=0,
            total_sources=0,
            success=False,
            error=str(e)
        )


@app.post("/api/upload", response_model=DocumentUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload endpoint - this is where you can add new documents to your RAG system!
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not files:
        return DocumentUploadResponse(
            success=False,
            message="No files provided",
            documents_processed=0,
            chunks_created=0
        )
    
    try:
        # Save uploaded files temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        saved_files = []
        for file in files:
            if file.filename:
                file_path = temp_dir / file.filename
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                saved_files.append(str(file_path))
        
        if not saved_files:
            return DocumentUploadResponse(
                success=False,
                message="No valid files to process",
                documents_processed=0,
                chunks_created=0
            )
        
        # Process documents through RAG pipeline
        result = rag_pipeline.ingest_documents(
            data_sources=saved_files,
            chunk_documents=True,
            batch_size=50
        )
        
        # Clean up temp files
        for file_path in saved_files:
            try:
                os.remove(file_path)
            except:
                pass
        
        if result['success']:
            return DocumentUploadResponse(
                success=True,
                message=f"Successfully processed {result['total_documents']} documents",
                documents_processed=result['total_documents'],
                chunks_created=result['total_chunks']
            )
        else:
            return DocumentUploadResponse(
                success=False,
                message="Failed to process documents",
                documents_processed=0,
                chunks_created=0
            )
            
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        return DocumentUploadResponse(
            success=False,
            message=f"Error processing files: {str(e)}",
            documents_processed=0,
            chunks_created=0
        )


@app.post("/api/ingest-websites", response_model=WebsiteIngestResponse)
async def ingest_websites(request: WebsiteIngestRequest):
    """
    Website ingestion endpoint - scrape websites and add them to the RAG system!
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not request.urls:
        return WebsiteIngestResponse(
            success=False,
            message="No URLs provided",
            websites_processed=0,
            pages_scraped=0,
            chunks_created=0
        )
    
    try:
        logger.info(f"Starting website ingestion for {len(request.urls)} URLs")
        
        # Process websites through RAG pipeline
        result = rag_pipeline.ingest_websites(
            urls=request.urls,
            max_depth=request.max_depth,
            max_pages_per_site=request.max_pages_per_site,
            include_patterns=request.include_patterns,
            exclude_patterns=request.exclude_patterns,
            chunk_documents=True
        )
        
        if result['success']:
            return WebsiteIngestResponse(
                success=True,
                message=f"Successfully scraped and indexed {len(request.urls)} websites",
                websites_processed=len(request.urls),
                pages_scraped=result['total_documents'],
                chunks_created=result['total_chunks']
            )
        else:
            return WebsiteIngestResponse(
                success=False,
                message=result.get('error', 'Failed to process websites'),
                websites_processed=0,
                pages_scraped=0,
                chunks_created=0
            )
            
    except Exception as e:
        logger.error(f"Error in website ingestion endpoint: {e}")
        return WebsiteIngestResponse(
            success=False,
            message=f"Error processing websites: {str(e)}",
            websites_processed=0,
            pages_scraped=0,
            chunks_created=0
        )


@app.get("/api/stats", response_model=SystemStats)
async def get_system_stats():
    """Get statistics about the RAG system"""
    if not rag_pipeline:
        return SystemStats(
            total_documents=0,
            vector_store_type="unknown",
            embedding_model="unknown",
            system_ready=False
        )
    
    try:
        stats = rag_pipeline.get_stats()
        
        # Get count of unique URLs
        existing_urls = rag_pipeline.components['vector_store'].get_existing_urls()
        
        return SystemStats(
            total_documents=stats['vector_store']['total_documents'],
            vector_store_type=stats['vector_store']['store_type'],
            embedding_model=stats['embedding_model'],
            system_ready=True,
            unique_urls=len(existing_urls),
            openai_available=openai_client is not None
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return SystemStats(
            total_documents=0,
            vector_store_type="error",
            embedding_model="error",
            system_ready=False
        )


@app.get("/api/status", response_model=APIStatus)
async def get_api_status():
    """Get API configuration status"""
    return APIStatus(
        openai_configured=openai_client is not None,
        rag_system_ready=rag_pipeline is not None
    )


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the chat server"""
    logger.info(f"Starting RAG Chat Server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    
    run_server()
