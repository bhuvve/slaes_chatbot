"""
FastAPI Backend for Sales Analytics Chatbot
AI-powered sales data assistant
"""

import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver

# Import from existing chatbot.py
from chatbot import (
    ChatState,
    Logger,
    query_analyzer_node,
    sql_generator_node,
    query_executor_node,
    response_generator_node
)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Sales Analytics Chatbot",
    description="AI-powered sales data assistant",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Session Management
# -----------------------------
class SessionManager:
    """Manages chatbot sessions with unique session IDs."""
    
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
        self.memory = MemorySaver()  # Shared memory for all sessions
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create the LangGraph agent (reusing chatbot.py nodes)."""
        graph = StateGraph(ChatState)
        
        # Add nodes from chatbot.py
        graph.add_node("query_analyzer", query_analyzer_node)
        graph.add_node("sql_generator", sql_generator_node)
        graph.add_node("query_executor", query_executor_node)
        graph.add_node("response_generator", response_generator_node)
        
        # Set entry point
        graph.set_entry_point("query_analyzer")
        
        # Connect edges
        graph.add_edge("query_analyzer", "sql_generator")
        graph.add_edge("sql_generator", "query_executor")
        graph.add_edge("query_executor", "response_generator")
        graph.add_edge("response_generator", END)
        
        # Compile with memory
        return graph.compile(checkpointer=self.memory)
    
    def create_session(self) -> str:
        """Create a new session and return the session ID."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": datetime.now(ZoneInfo("Pacific/Auckland")),
            "message_count": 0
        }
        Logger.info(f"New session created: {session_id[:8]}...")
        return session_id
    
    def get_session(self, session_id: str) -> dict:
        """Get session info or None if not found."""
        return self.sessions.get(session_id)
    
    def chat(self, session_id: str, message: str) -> str:
        """Process a chat message for a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        # Update session
        self.sessions[session_id]["message_count"] += 1
        
        # Invoke agent with session-specific thread_id
        config = {
            "configurable": {
                "thread_id": session_id
            }
        }
        
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )
        
        # Extract AI response
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content
        
        return "I'm sorry, I couldn't process your request."


# Global session manager
session_manager = SessionManager()


# -----------------------------
# Request/Response Models
# -----------------------------
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    timestamp: str


class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    message: str


# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the chat frontend."""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/session", response_model=SessionResponse)
async def create_session():
    """Create a new chat session (auto-called by frontend on load)."""
    session_id = session_manager.create_session()
    nz_time = datetime.now(ZoneInfo("Pacific/Auckland"))
    
    return SessionResponse(
        session_id=session_id,
        created_at=nz_time.isoformat(),
        message="Session created successfully. Start chatting!"
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message."""
    try:
        # Validate session
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Process message
        response = session_manager.chat(request.session_id, request.message)
        
        nz_time = datetime.now(ZoneInfo("Pacific/Auckland"))
        
        return ChatResponse(
            session_id=request.session_id,
            response=response,
            timestamp=nz_time.strftime("%H:%M:%S")
        )
    
    except Exception as e:
        Logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "created_at": session["created_at"].isoformat(),
        "message_count": session["message_count"]
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(ZoneInfo("Pacific/Auckland")).isoformat(),
        "active_sessions": len(session_manager.sessions)
    }


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("📊 Sales Analytics Chatbot - FastAPI Server")
    print("=" * 60)
    print("🌐 Open http://localhost:8002 in your browser")
    print("📚 API docs at http://localhost:8002/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
