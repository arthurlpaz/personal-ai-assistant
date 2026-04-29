"""
Document ingestion endpoints.

POST /api/ingest  – upload a file (.txt .md .py .pdf) to the knowledge base
GET  /api/ingest/status – check if vectorstore exists
"""

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.rag.ingest import ingest_file
from app.schemas.chat import IngestResponse

router = APIRouter(prefix="/api", tags=["ingest"])

ALLOWED_EXTENSIONS = {".txt", ".md", ".py", ".rst", ".pdf"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Write upload to a temp file, then ingest
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks = ingest_file(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)

    return IngestResponse(filename=file.filename, chunks_added=chunks, status="ok")


@router.get("/ingest/status")
def ingest_status():
    has_vs = os.path.exists("vectorstore")
    return {"vectorstore_ready": has_vs}
