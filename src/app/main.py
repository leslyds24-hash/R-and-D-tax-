from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Dict, Any
import pandas as pd
from io import BytesIO
import json
import zipfile
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from .models import ProjectRecord
from .reasoning import analyze_project, analyze_project_async
from .trace import ImmutableTraceLogger
from .auth import enforce_api_key, require_role
from .explainability_pack import ExplainabilityGenerator

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "qwen2.5:7b")
USE_LLM = bool(OPENAI_API_KEY)

app = FastAPI(title="AI R&D Tax Credit Agent - MVP API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = ImmutableTraceLogger()
classification_cache: Dict[str, Dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=4)
explainability_gen = ExplainabilityGenerator()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify_rnd")
async def classify_rnd(
    file: UploadFile = File(...),
    user_id: str = Form("demo-user"),
    api_key: str = Depends(enforce_api_key),
) -> Dict[str, Any]:
    """
    Accept a CSV of projects and return classifications + write traces.
    Requires a valid API key.
    """
    df = pd.read_csv(file.file)
    expected_cols = {"project_id", "project_name", "description"}
    if not expected_cols.issubset(set(df.columns)):
        return {"error": f"CSV must contain columns: {expected_cols}"}

    results = []
    loop = asyncio.get_event_loop()
    
    for _, row in df.iterrows():
        record = ProjectRecord(
            project_id=str(row.get("project_id")),
            project_name=str(row.get("project_name")),
            description=str(row.get("description")),
            department=str(row.get("department")) if "department" in df.columns else None,
            cost=float(row.get("cost")) if "cost" in df.columns and pd.notnull(row.get("cost")) else None,
            start_date=str(row.get("start_date")) if "start_date" in df.columns else None,
            end_date=str(row.get("end_date")) if "end_date" in df.columns else None,
        )
        # Run sync analyze_project in thread pool to avoid blocking event loop
        classification, trace = await loop.run_in_executor(
            executor,
            analyze_project,
            record,
            user_id
        )
        path = logger.write_trace(trace)

        payload = {
            "project_id": classification.project_id,
            "project_name": record.project_name,
            "description": record.description,  # Cached for explanation generation
            "cost": record.cost or 0.0,         # Cached for explanation generation
            "eligible": classification.eligible,
            "confidence": classification.confidence,
            "rationale": classification.rationale,
            "region": classification.region,
            "trace_path": path,
        }
        payload = _enrich_payload(payload)
        results.append(payload)
        classification_cache[classification.project_id] = payload

    return {"count": len(results), "results": results}


def _enrich_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate UI-specific fields missing from the core reasoning for the frontend."""
    eligible = payload.get("eligible", False)
    conf = payload.get("confidence", 0.0)

    # 1. Confidence Band
    if conf >= 0.8:
        band = "HIGH"
    elif conf >= 0.5:
        band = "MED"
    else:
        band = "LOW"

    # 2. Recommendation String
    if eligible and band == "HIGH":
        rec = "Strongly Approve"
    elif eligible:
        rec = "Approve (Borderline)"
    elif not eligible and band == "HIGH":
        rec = "Likely Reject"
    else:
        rec = "Reject (Insufficient Evidence)"

    payload["confidence_band"] = band
    payload["recommendation"] = rec
    payload["primary_risk"] = "Technological Uncertainty" if not eligible else "Documentation Sufficiency"

    # Mock 4-part test for UI visualization (logic matches standard reasoning patterns)
    payload["four_part_test"] = {
        "permitted_purpose": "met" if eligible else "uncertain",
        "elimination_uncertainty": "met" if eligible and conf > 0.7 else "uncertain",
        "process_experimentation": "met" if eligible and conf > 0.8 else "uncertain",
        "technological_nature": "met" if eligible else "met",
    }
    payload["decision_flippers"] = [
        "Project involves genuine CS/Engineering research.",
        "Systematic trial and error was documented."
    ] if eligible else ["Lack of evidence for systematic experimentation."]

    return payload


def _get_project_data(project_id: str) -> Dict[str, Any]:
    """
    Lookup project classification + trace metadata.
    MVP: in-memory cache populated by /classify_rnd.
    """
    if project_id not in classification_cache:
        raise ValueError(f"project_id {project_id} not found in cache. Run /classify_rnd first.")
    return classification_cache[project_id]


@app.post("/generate_form_6765")
async def generate_form_6765(
    project_id: str = Form(...),
    api_key: str = Depends(enforce_api_key),
):
    """
    Generate a simple Form 6765-style PDF for a given project (MVP).
    """
    try:
        data = _get_project_data(project_id)
    except ValueError as e:
        return {"error": str(e)}

    try:
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=LETTER)

        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, 750, "Form 6765 (MVP) - Credit for Increasing Research Activities")

        c.setFont("Helvetica", 11)
        c.drawString(72, 720, f"Project ID: {data['project_id']}")
        c.drawString(72, 705, f"Project Name: {data.get('project_name', '')}")
        c.drawString(72, 690, f"Region: {data.get('region', '')}")
        c.drawString(72, 675, f"Eligible: {data['eligible']}")
        c.drawString(72, 660, f"Confidence: {data['confidence']:.2f}")

        rationale = (data.get("rationale") or "")[:1000]
        c.drawString(72, 640, "LLM Rationale (truncated):")
        text_obj = c.beginText(72, 625)
        text_obj.setFont("Helvetica", 10)
        for line in rationale.split("\n"):
            text_obj.textLine(line)
        c.drawText(text_obj)

        c.showPage()
        c.save()
        buf.seek(0)

        headers = {
            "Content-Disposition": f'attachment; filename="form6765_{project_id}.pdf"',
            "Content-Length": str(len(buf.getvalue()))
        }
        return StreamingResponse(buf, media_type="application/pdf", headers=headers)
    except Exception as e:
        return {"error": f"PDF generation failed: {str(e)}"}


@app.post("/audit_package")
async def audit_package(
    project_id: str = Form(...),
    api_key: str = Depends(enforce_api_key),
):
    """
    Return a ZIP with:
    - classification_summary.json (project + eligibility + rationale)
    - trace_pointer.txt (path to the encrypted trace file)
    - cfo_explanation.md (Plain English narrative)
    """
    try:
        data = _get_project_data(project_id)
    except ValueError as e:
        return {"error": str(e)}

    try:
        # Generate CFO explanation
        # We run this in executor to avoid blocking if it's slow (calling LLM)
        loop = asyncio.get_event_loop()
        cfo_narrative = await loop.run_in_executor(
            executor,
            explainability_gen.generate_cfo_explanation,
            data.get("project_name", ""),
            data.get("description", ""),
            data.get("eligible", False),
            data.get("rationale", ""),
            data.get("confidence", 0.0),
            data.get("cost", 0.0)
        )

        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("classification_summary.json", json.dumps(data, indent=2))
            zf.writestr("trace_pointer.txt", data.get("trace_path", ""))
            zf.writestr("cfo_explanation.md", cfo_narrative)

        zip_buf.seek(0)
        headers = {
            "Content-Disposition": f'attachment; filename="audit_package_{project_id}.zip"',
            "Content-Length": str(len(zip_buf.getvalue()))
        }
        return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)
    except Exception as e:
        return {"error": f"ZIP package generation failed: {str(e)}"}