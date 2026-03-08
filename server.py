"""
FastAPI server for Doctor Copilot
Serves a lightweight HTML/CSS/JS UI and exposes API endpoints to initialize
and run the multi-agent cardiology pipeline.

Run with:
    pip install -r requirements.txt
    uvicorn server:app --reload --host 0.0.0.0 --port 8000

The server initializes DoctorCopilotOrchestrator at startup so the heavy
NLP model (BioBERT) is loaded once.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from pathlib import Path
from typing import Any, Dict

# Import orchestrator (agents live here)
from cardiology_orchestrator import DoctorCopilotOrchestrator

app = FastAPI(title="Doctor Copilot API")

# Serve static UI files from ./web
app.mount("/static", StaticFiles(directory="web"), name="static")

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator once at startup."""
    # Run the potentially blocking initialization in a thread
    loop = asyncio.get_running_loop()
    app.state.orchestrator = await asyncio.to_thread(DoctorCopilotOrchestrator)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Return the UI index page."""
    index_path = Path("web/index.html")
    if not index_path.exists():
        return PlainTextResponse("UI not found. Ensure /web/index.html exists", status_code=500)
    return HTMLResponse(index_path.read_text())


@app.get("/api/status")
async def status() -> Dict[str, Any]:
    """Return orchestrator status and agent info."""
    orch = app.state.orchestrator
    agent_type = orch.ingestion_agent.__class__.__name__ if orch.ingestion_agent else None
    return {
        "status": "ready" if orch else "not ready",
        "agent_type": agent_type,
        "biobert_active": getattr(orch, 'ingestion_agent', None) is not None and 'BioBERT' in str(agent_type)
    }


@app.post("/api/process")
async def process(request: Request):
    """Process a clinical note and return a report JSON and formatted_text."""
    payload = await request.json()
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in request body")

    orch = app.state.orchestrator
    if not orch:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    try:
        # Call CPU-bound processing in a thread to avoid blocking
        report = await asyncio.to_thread(orch.process_patient, text)

        # Also produce a formatted text version
        formatted = orch.report_agent.generate_formatted_report(report)

        return JSONResponse({"report": report, "formatted": formatted})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/initialize")
async def initialize():
    """(Re-)initialize the orchestrator on demand."""
    try:
        app.state.orchestrator = await asyncio.to_thread(DoctorCopilotOrchestrator)
        return {"status": "initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Simple health check
@app.get("/api/health")
async def health():
    return {"healthy": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
