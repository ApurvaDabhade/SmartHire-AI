import sys
import os
import json
import time
import re
from pathlib import Path
from typing import List, Optional

sys.dont_write_bytecode = True

# Ensure demo directory is in sys.path
BASE_DIR = Path(__file__).parent.resolve()
DEMO_DIR = BASE_DIR / "demo"
if str(DEMO_DIR) not in sys.path:
  sys.path.insert(0, str(DEMO_DIR))

from dotenv import load_dotenv
load_dotenv(override=True)

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy

from retriever import SelfQueryRetriever
from llm_agent import ChatBot
from ingest_data import read_resume_csv, ingest, format_import_report

# Configuration
DATA_PATH = os.getenv("DATA_PATH", "./data/supplementary-data/pdf-resumes.csv")
FAISS_PATH = os.getenv("FAISS_PATH", "./vectorstore-pdf")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

app = FastAPI(title="SmartHire.Ai API", version="1.0.0")

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# App state containers
state = {
  "default_df": None,
  "default_vectordb": None,
  "current_df": None,
  "current_vectordb": None,
  "embedding_model": None,
  "retriever": None,
  "chatbot": None,
  "is_custom_dataset": False,
  "custom_dataset_name": None,
}


def initialize_app_state():
  print("[Startup] Initializing embedding model...")
  state["embedding_model"] = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
  )

  print(f"[Startup] Loading dataset from {DATA_PATH}...")
  full_data_path = BASE_DIR / DATA_PATH if not os.path.isabs(DATA_PATH) else Path(DATA_PATH)
  state["default_df"] = pd.read_csv(full_data_path)
  state["current_df"] = state["default_df"]

  print(f"[Startup] Loading FAISS index from {FAISS_PATH}...")
  full_faiss_path = BASE_DIR / FAISS_PATH if not os.path.isabs(FAISS_PATH) else Path(FAISS_PATH)
  state["default_vectordb"] = FAISS.load_local(
    str(full_faiss_path),
    state["embedding_model"],
    distance_strategy=DistanceStrategy.COSINE,
    allow_dangerous_deserialization=True,
  )
  state["current_vectordb"] = state["default_vectordb"]

  state["retriever"] = SelfQueryRetriever(state["current_vectordb"], state["current_df"])
  state["chatbot"] = ChatBot(api_key=GROQ_API_KEY, model=GROQ_MODEL)
  print("[Startup] Ready!")


# Initialize on import
initialize_app_state()


class ChatRequest(BaseModel):
  message: str
  rag_mode: str = "Generic RAG"  # "Generic RAG" or "RAG Fusion"
  chat_history: Optional[List[str]] = []


@app.get("/api/status")
def get_status():
  return {
    "status": "ready",
    "badge": "MiniLM · FAISS · GPT-OSS-20B",
    "embedding_model": "all-MiniLM-L6-v2",
    "vector_store": "FAISS",
    "llm_model": GROQ_MODEL,
    "resumes_count": len(state["current_df"]) if state["current_df"] is not None else 0,
    "is_custom_dataset": state["is_custom_dataset"],
    "custom_dataset_name": state["custom_dataset_name"],
    "has_api_key": bool(GROQ_API_KEY),
  }


@app.get("/api/candidate/{candidate_id}")
def get_candidate(candidate_id: str):
  df = state["current_df"]
  if df is None:
    raise HTTPException(status_code=404, detail="No dataset loaded")
  
  matched = df[df["ID"].astype(str) == str(candidate_id)]
  if matched.empty:
    raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found")
  
  row = matched.iloc[0]
  resume_text = str(row.get("Resume", ""))
  
  # Basic extracted fields if available
  extra_info = {}
  for col in df.columns:
    if col not in ("Resume", "ID"):
      extra_info[col] = str(row[col]) if pd.notna(row[col]) else ""
      
  return {
    "id": str(candidate_id),
    "resume": resume_text,
    "metadata": extra_info,
  }


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
  question = req.message.strip()
  if not question:
    raise HTTPException(status_code=400, detail="Message cannot be empty")

  rag_mode = "RAG Fusion" if "fusion" in req.rag_mode.lower() else "Generic RAG"
  chatbot = state["chatbot"]
  retriever = state["retriever"]

  def event_generator():
    start_time = time.time()
    try:
      # Perform retrieval
      document_list = retriever.retrieve_docs(question, chatbot, rag_mode)
      time_elapsed = time.time() - start_time

      if not isinstance(document_list, list):
        document_list = []

      query_type = retriever.meta_data.get("query_type", "no_retrieve")
      subquestions = retriever.meta_data.get("subquestion_list", [])
      doc_scores = retriever.meta_data.get("retrieved_docs_with_scores", {})

      # Parse candidates for frontend cards
      candidates = []
      for rank, doc_text in enumerate(document_list[:5], start=1):
        m = re.match(r"^Applicant ID\s+([A-Za-z0-9._\-]+)\s*\n(.*)", doc_text, flags=re.DOTALL)
        if m:
          c_id = m.group(1).strip()
          c_body = m.group(2).strip()
        else:
          c_id = f"Candidate-{rank}"
          c_body = doc_text.strip()

        # Score formatted
        score = None
        if isinstance(doc_scores, dict) and c_id in doc_scores:
          score = round(float(doc_scores[c_id]), 4)

        # Snippet
        snippet = c_body[:300].strip() + ("..." if len(c_body) > 300 else "")

        candidates.append({
          "rank": rank,
          "id": c_id,
          "score": score,
          "snippet": snippet,
        })

      # Stream metadata event
      meta_payload = {
        "type": "meta",
        "query_type": query_type,
        "rag_mode": rag_mode,
        "time_elapsed": round(time_elapsed, 3),
        "subquestions": subquestions if rag_mode == "RAG Fusion" else [],
        "candidates": candidates,
        "total_retrieved": len(document_list),
      }
      yield f"data: {json.dumps(meta_payload)}\n\n"

      # History formatting
      clean_history = [h[:400] for h in (req.chat_history or [])][-4:]

      # Stream LLM tokens
      stream = chatbot.generate_message_stream(question, document_list, clean_history, query_type)
      for chunk in stream:
        token = getattr(chunk, "content", "")
        if token:
          yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

      yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as exc:
      err_msg = str(exc)
      if "413" in err_msg or "rate_limit_exceeded" in err_msg or "Request too large" in err_msg:
        err_display = "The request was too large for Groq API limit. Please try a shorter job description."
      elif "401" in err_msg or "invalid_api_key" in err_msg:
        err_display = "Invalid Groq API key. Please check GROQ_API_KEY in .env."
      else:
        err_display = f"Error generating response: {err_msg}"
      yield f"data: {json.dumps({'type': 'error', 'message': err_display})}\n\n"

  return StreamingResponse(
    event_generator(),
    media_type="text/event-stream",
    headers={
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  )


@app.post("/api/upload")
async def upload_resumes(file: UploadFile = File(...)):
  if not file.filename.lower().endswith(".csv"):
    raise HTTPException(status_code=400, detail="Only CSV files are supported.")
  
  try:
    content = await file.read()
    df_load = read_resume_csv(content)
    report = getattr(df_load, "attrs", {}).get("import_report") or {}
    
    # Ingest and create new FAISS index
    new_vectordb = ingest(df_load, "Resume", state["embedding_model"])
    
    # Update active state
    state["current_df"] = df_load
    state["current_vectordb"] = new_vectordb
    state["retriever"] = SelfQueryRetriever(new_vectordb, df_load)
    state["is_custom_dataset"] = True
    state["custom_dataset_name"] = file.filename
    
    report_text = format_import_report(report)
    return {
      "success": True,
      "message": f"Successfully indexed {len(df_load)} resumes from {file.filename}.",
      "count": len(df_load),
      "report": report,
      "report_text": report_text,
    }
  except Exception as exc:
    raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/reset")
def reset_default_dataset():
  state["current_df"] = state["default_df"]
  state["current_vectordb"] = state["default_vectordb"]
  state["retriever"] = SelfQueryRetriever(state["default_vectordb"], state["default_df"])
  state["is_custom_dataset"] = False
  state["custom_dataset_name"] = None
  return {
    "success": True,
    "message": "Reset to default resume pool.",
    "count": len(state["default_df"]),
  }


# Serve frontend static files
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
  app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

  @app.get("/")
  def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

  @app.get("/favicon.ico")
  def serve_favicon():
    return FileResponse(str(FRONTEND_DIR / "favicon.svg"), media_type="image/svg+xml")
