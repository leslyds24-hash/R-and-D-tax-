import streamlit as st
import pandas as pd
import requests
import os
import json
import zipfile
from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

st.set_page_config(page_title="AI R&D Tax Credit Agent - MVP", layout="wide")

st.title("AI R&D Tax Credit Agent - MVP (Phase 1)")
st.write("Upload a CSV of project descriptions to classify R&D tax credit eligibility and export supporting documents.")

CONFIDENCE_THRESHOLD = 0.75

if "projects" not in st.session_state:
    st.session_state["projects"] = {}  # project_id -> project dict with ai_decision/human_decision
if "needs_review_ids" not in st.session_state:
    st.session_state["needs_review_ids"] = []

def get_final_decision(project: dict):
    """Return (label, confidence, source) preferring human override when present."""
    human = project.get("human_decision")
    if human:
        return human.get("final_label"), human.get("confidence", 1.0), "Human"
    ai = project.get("ai_decision", {})
    return ai.get("label"), ai.get("confidence", 0.0), "AI"

def build_display_df():
    rows = []
    for pid, project in st.session_state["projects"].items():
        label, conf, source = get_final_decision(project)
        rows.append(
            {
                "project_id": pid,
                "project_name": project.get("project_name", ""),
                "eligible": label == "Eligible",
                "confidence": conf,
                "decision_source": source,
                "status": project.get("status", "AI Classified"),
                "region": project.get("region"),
                "recommendation": project.get("ai_decision", {}).get("recommendation", "UNKNOWN"),
                "confidence_band": project.get("ai_decision", {}).get("confidence_band", "UNKNOWN"),
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def ingest_results(results):
    """Sync backend results into session state and flag low-confidence items."""
    projects = st.session_state["projects"]
    st.session_state["needs_review_ids"] = []
    for row in results:
        pid = str(row.get("project_id"))
        ai_decision = {
            "label": "Eligible" if row.get("eligible") else "Not Eligible",
            "confidence": float(row.get("confidence", 0.0)),
            "rationale": row.get("rationale", ""),
            "overall_rationale": row.get("overall_rationale", row.get("rationale", "")),
            "recommendation": row.get("recommendation", "UNKNOWN"),
            "confidence_band": row.get("confidence_band", "UNKNOWN"),
            "primary_risk": row.get("primary_risk", ""),
            "four_part_test": row.get("four_part_test", {}),
            "decision_flippers": row.get("decision_flippers", []),
        }
        project = projects.get(pid, {})
        project.update(
            {
                "project_id": pid,
                "project_name": row.get("project_name", ""),
                "region": row.get("region"),
                "trace_path": row.get("trace_path"),
                "ai_decision": ai_decision,
            }
        )

        if project.get("human_decision"):
            project["status"] = "Reviewed"
        else:
            if ai_decision["confidence"] < CONFIDENCE_THRESHOLD or ai_decision["confidence_band"] == "LOW":
                project["status"] = "Needs Review"
                st.session_state["needs_review_ids"].append(pid)
            else:
                project["status"] = "AI Classified"

        projects[pid] = project

    st.session_state["results_df"] = build_display_df()
    st.session_state["results_raw"] = results

def make_export_payload(project_id: str) -> dict:
    project = st.session_state["projects"][project_id]
    label, conf, source = get_final_decision(project)
    rationale = project.get("human_decision", {}).get("rationale") or project.get("ai_decision", {}).get("rationale", "")
    return {
        "project_id": project_id,
        "project_name": project.get("project_name", ""),
        "region": project.get("region"),
        "eligible": label == "Eligible",
        "eligible_label": label,
        "confidence": conf,
        "decision_source": source,
        "status": project.get("status", ""),
        "rationale": rationale,
        "trace_path": project.get("trace_path", ""),
    }

def render_form_6765_pdf_local(payload: dict) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 750, "Form 6765 (MVP) - Credit for Increasing Research Activities")
    c.setFont("Helvetica", 11)
    c.drawString(72, 720, f"Project ID: {payload['project_id']}")
    c.drawString(72, 705, f"Project Name: {payload.get('project_name', '')}")
    c.drawString(72, 690, f"Region: {payload.get('region', '')}")
    c.drawString(72, 675, f"Eligible: {payload.get('eligible_label', '')}")
    c.drawString(72, 660, f"Confidence: {payload.get('confidence', 0.0):.2f} ({payload.get('decision_source', '')})")
    rationale = (payload.get("rationale") or "")[:1200]
    c.drawString(72, 640, "Rationale (truncated):")
    text_obj = c.beginText(72, 625)
    text_obj.setFont("Helvetica", 10)
    for line in rationale.split("\n"):
        text_obj.textLine(line)
    c.drawText(text_obj)
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()

backend_url = st.text_input(
    "Backend URL",
    value=os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"),
)
user_id = st.text_input("User ID (for trace)", value="demo-user")

_default_api_key = os.environ.get("API_KEY_DEFAULT")
if not _default_api_key:
    _valid_keys_env = os.environ.get("VALID_API_KEYS", "")
    if _valid_keys_env:
        _default_api_key = _valid_keys_env.split(",")[0].strip()
api_key = st.text_input("API Key (X-API-Key)", value=_default_api_key or "", type="password")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

results_df = st.session_state.get("results_df", None)

if uploaded and st.button("Analyze"):
    if not api_key:
        st.error("API key is required.")
    else:
        with st.spinner("Classifying... (this may take several minutes)"):
            try:
                resp = requests.post(
                    f"{backend_url}/classify_rnd",
                    files={"file": uploaded},
                    data={"user_id": user_id},
                    headers={"X-API-Key": api_key},
                    timeout=600,
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    if "results" in payload:
                        ingest_results(payload["results"])
                        results_df = st.session_state.get("results_df")
                        st.success(f"Processed {payload.get('count', len(payload['results']))} rows.")
                        if results_df is not None and not results_df.empty:
                            st.dataframe(results_df, use_container_width=True)
                    else:
                        st.error(payload)
                else:
                    st.error(f"Error: {resp.status_code} -> {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# Reload results
if results_df is not None and not results_df.empty:
    st.markdown("### Classification Summary")
    st.dataframe(results_df, use_container_width=True)

    st.markdown("### Detailed Analysis")
    results_raw = st.session_state.get("results_raw", [])
    for idx, result in enumerate(results_raw):
        pid = result.get("project_id", f"Project {idx}")
        rec = result.get("recommendation", "UNKNOWN")
        band = result.get("confidence_band", "UNKNOWN")
        
        status_emoji = "✅" if "ELIGIBLE" in rec else "❌"
        with st.expander(f"{status_emoji} **{pid}** — {rec} ({band})"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Eligible:** {result.get('eligible', False)}")
                st.write(f"**Confidence:** {result.get('confidence', 0):.2f} ({band})")
                if result.get("primary_risk"):
                    st.warning(f"**Primary Risk:** {result.get('primary_risk')}")
                st.write("**Rationale:**")
                st.write(result.get("overall_rationale", result.get("rationale", "")))
            with col2:
                four_part = result.get("four_part_test", {})
                if four_part:
                    st.write("**Four-Part Test Scores:**")
                    for k,v in four_part.items():
                        st.write(f"• {k.replace('_',' ').title()}: {v:.2f}")
                flippers = result.get("decision_flippers", [])
                if flippers:
                    st.write("**Decision Sensitivity:**")
                    for f in flippers:
                        st.write(f"• {f}")

    st.markdown("### Expert Review Queue")
    needs_review = [pid for pid in st.session_state.get("needs_review_ids", []) if pid in st.session_state["projects"]]
    if needs_review:
        selected_pid = st.selectbox("Select project for review", needs_review)
        selected_proj = st.session_state["projects"][selected_pid]
        ai_decision = selected_proj["ai_decision"]
        with st.form(f"review_{selected_pid}"):
            final_label = st.radio("Final Decision", ["Eligible", "Not Eligible"], index=(0 if ai_decision["label"]=="Eligible" else 1))
            final_rationale = st.text_area("Expert Rationale", height=100)
            if st.form_submit_button("Commit Review"):
                st.session_state["projects"][selected_pid]["human_decision"] = {"final_label": final_label, "rationale": final_rationale, "confidence": 1.0}
                st.session_state["projects"][selected_pid]["status"] = "Reviewed"
                st.session_state["needs_review_ids"].remove(selected_pid)
                st.session_state["results_df"] = build_display_df()
                st.rerun()

    st.markdown("### Form 6765 & Audit Package")
    project_ids = results_df["project_id"].astype(str).tolist()
    selected_project = st.selectbox("Select Project ID for Export", project_ids)

    st.markdown("#### Form 6765 Configuration")
    c1, c2 = st.columns(2)
    with c1:
        tax_year = st.number_input("Tax Year", value=datetime.utcnow().year)
        name_on_return = st.text_input("Name on Return", value="Sample Taxpayer Inc.")
        identifying_number = st.text_input("Identifying Number", value="00-0000000")
        ruleset_version = st.text_input("Ruleset Version", value="2024.1")
        created_by = st.text_input("Created By", value=user_id or "streamlit-user")
        override_reason = st.text_area("Override Reason (optional, 30+ chars)")
        override_role = st.selectbox("Override Role", ["", "ADMIN", "PARTNER", "DIRECTOR"])
    with c2:
        qre_wages = st.number_input("QRE Wages", value=0.0)
        qre_supplies = st.number_input("QRE Supplies", value=0.0)
        qre_contract = st.number_input("Contract Research", value=0.0)
        credit_method = st.selectbox("Credit Method", ["ASC", "REGULAR"])
        section_280c = st.selectbox("Section 280C", ["FULL", "REDUCED"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Backend Form 6765"):
            payload = {
                "header": {"tax_year": int(tax_year), "name_on_return": name_on_return, "identifying_number": identifying_number},
                "inputs": {"qre_wages": qre_wages, "qre_supplies": qre_supplies, "qre_contract_research_gross": qre_contract, "credit_method": credit_method, "section_280c_choice": section_280c},
                "project_ids": [selected_project],
                "ruleset_version": ruleset_version, "created_by": created_by,
                "override_reason": override_reason or None, "save_pdf": True
            }
            h = {"X-API-Key": api_key}
            if override_role: h["X-Role"] = override_role
            with st.spinner("Generating..."):
                r = requests.post(f"{backend_url}/form6765/generate", json=payload, headers=h)
                if r.status_code == 200:
                    st.success("Form generated!")
                    fid = r.json().get("form_version", {}).get("form_version_id")
                    if fid:
                        pr = requests.get(f"{backend_url}/form6765/form/{fid}/pdf", headers=h)
                        st.download_button("Download PDF", pr.content, f"form6765_{fid}.pdf", "application/pdf")
                else: st.error(r.text)

    with col2:
        if st.button("Download Audit Package"):
            r = requests.post(f"{backend_url}/audit_package", data={"project_id": selected_project}, headers={"X-API-Key": api_key})
            if r.status_code == 200:
                st.download_button("Save Audit Package", r.content, f"audit_{selected_project}.zip", "application/zip")
            else: st.error(r.text)

st.markdown("---")
st.caption("Tip: Start the backend with `uvicorn src.app.main:app --reload --port 8000` before running Streamlit.")