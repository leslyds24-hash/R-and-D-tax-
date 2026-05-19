import streamlit as st
import pandas as pd
import requests
import os
from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

st.set_page_config(
    page_title="R&D Tax Credit — AI Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Page background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #1e2535;
    }

    /* Top header strip */
    .header-block {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 28px 0 8px 0;
        border-bottom: 1px solid #1e2535;
        margin-bottom: 28px;
    }
    .header-block h1 {
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
        color: #f0f2f6;
        letter-spacing: -0.5px;
    }
    .header-block p {
        margin: 4px 0 0 0;
        font-size: 0.875rem;
        color: #8b92a5;
    }

    /* Section labels */
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #4a90d9;
        margin-bottom: 10px;
    }

    /* Metric cards */
    .metric-card {
        background: #161b27;
        border: 1px solid #1e2535;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #f0f2f6;
        line-height: 1;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #8b92a5;
        margin-top: 4px;
        letter-spacing: 0.5px;
    }
    .metric-card .value.green { color: #22c55e; }
    .metric-card .value.red   { color: #ef4444; }
    .metric-card .value.amber { color: #f59e0b; }

    /* Result badges */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.4px;
    }
    .badge-eligible   { background: #14532d; color: #4ade80; }
    .badge-ineligible { background: #450a0a; color: #f87171; }
    .badge-review     { background: #451a03; color: #fb923c; }
    .badge-high       { background: #14532d; color: #4ade80; }
    .badge-med        { background: #451a03; color: #fb923c; }
    .badge-low        { background: #450a0a; color: #f87171; }

    /* Divider */
    hr { border-color: #1e2535 !important; }

    /* Expander header */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: #d1d5db !important;
    }

    /* Table styling */
    .dataframe thead th {
        background: #161b27 !important;
        color: #8b92a5 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.75

# ── Session state bootstrap ────────────────────────────────────────────────────
if "projects" not in st.session_state:
    st.session_state["projects"] = {}
if "needs_review_ids" not in st.session_state:
    st.session_state["needs_review_ids"] = []

# ── Sidebar — configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")

    backend_url = st.text_input(
        "Backend URL",
        value=os.environ.get("BACKEND_URL", "https://r-and-d-tax.onrender.com"),
        help="FastAPI backend endpoint",
    )
    user_id = st.text_input("User ID", value="demo-user", help="Used for audit trace")

    _default_key = os.environ.get("API_KEY_DEFAULT") or (
        os.environ.get("VALID_API_KEYS", "").split(",")[0].strip() or ""
    )
    api_key = st.text_input("API Key", value=_default_key, type="password")

    st.markdown("---")
    st.markdown(
        "<span style='color:#8b92a5;font-size:0.75rem'>R&D Tax Credit AI · v1.0</span>",
        unsafe_allow_html=True,
    )

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="header-block">
        <div>
            <h1>🔬 R&D Tax Credit — AI Classification</h1>
            <p>Upload a CSV of project descriptions · Classify eligibility · Export Form 6765 & audit package</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Helper functions ───────────────────────────────────────────────────────────

def get_final_decision(project: dict):
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
                "ID": pid,
                "Project Name": project.get("project_name", ""),
                "Eligible": label == "Eligible",
                "Confidence": conf,
                "Band": project.get("ai_decision", {}).get("confidence_band", "—"),
                "Source": source,
                "Status": project.get("status", "AI Classified"),
                "Recommendation": project.get("ai_decision", {}).get("recommendation", "—"),
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def ingest_results(results):
    projects = st.session_state["projects"]
    st.session_state["needs_review_ids"] = []
    for row in results:
        pid = str(row.get("project_id"))
        ai_decision = {
            "label": "Eligible" if row.get("eligible") else "Not Eligible",
            "confidence": float(row.get("confidence", 0.0)),
            "rationale": row.get("rationale", ""),
            "overall_rationale": row.get("overall_rationale", row.get("rationale", "")),
            "recommendation": row.get("recommendation", "—"),
            "confidence_band": row.get("confidence_band", "—"),
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
        elif ai_decision["confidence"] < CONFIDENCE_THRESHOLD or ai_decision["confidence_band"] == "LOW":
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
    rationale = (
        project.get("human_decision", {}).get("rationale")
        or project.get("ai_decision", {}).get("rationale", "")
    )
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
    c.drawString(72, 750, "Form 6765 — Credit for Increasing Research Activities")
    c.setFont("Helvetica", 11)
    c.drawString(72, 720, f"Project ID: {payload['project_id']}")
    c.drawString(72, 705, f"Project Name: {payload.get('project_name', '')}")
    c.drawString(72, 690, f"Region: {payload.get('region', '')}")
    c.drawString(72, 675, f"Eligible: {payload.get('eligible_label', '')}")
    c.drawString(72, 660, f"Confidence: {payload.get('confidence', 0.0):.2f} ({payload.get('decision_source', '')})")
    rationale = (payload.get("rationale") or "")[:1200]
    c.drawString(72, 640, "Rationale:")
    text_obj = c.beginText(72, 625)
    text_obj.setFont("Helvetica", 10)
    for line in rationale.split("\n"):
        text_obj.textLine(line)
    c.drawText(text_obj)
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


# ── Upload & Analyze ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Upload</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "CSV file — columns: project_id, project_name, description",
    type=["csv"],
    label_visibility="visible",
)

if uploaded and st.button("Analyze Projects", type="primary", use_container_width=False):
    if not api_key:
        st.error("An API key is required. Set it in the sidebar.")
    else:
        with st.spinner("Classifying projects — this may take a few minutes…"):
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
                        st.success(
                            f"Classified {payload.get('count', len(payload['results']))} projects successfully."
                        )
                    else:
                        st.error(str(payload))
                else:
                    st.error(f"Backend error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# ── Results ────────────────────────────────────────────────────────────────────
results_df = st.session_state.get("results_df")
results_raw = st.session_state.get("results_raw", [])

if results_df is not None and not results_df.empty:
    st.markdown("---")

    # Summary metrics
    total = len(results_df)
    eligible_count = int(results_df["Eligible"].sum())
    review_count = len(st.session_state.get("needs_review_ids", []))
    avg_conf = results_df["Confidence"].mean()

    st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="value">{total}</div><div class="label">Projects Analyzed</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="value green">{eligible_count}</div><div class="label">Eligible</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="value red">{total - eligible_count}</div><div class="label">Not Eligible</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        color = "green" if avg_conf >= 0.75 else "amber" if avg_conf >= 0.5 else "red"
        st.markdown(
            f'<div class="metric-card"><div class="value {color}">{avg_conf:.0%}</div><div class="label">Avg. Confidence</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Results table
    st.markdown('<div class="section-label">Classification Results</div>', unsafe_allow_html=True)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Detailed analysis
    st.markdown('<div class="section-label">Detailed Analysis</div>', unsafe_allow_html=True)
    for idx, result in enumerate(results_raw):
        pid = str(result.get("project_id", f"Project {idx}"))
        rec = result.get("recommendation", "—")
        band = result.get("confidence_band", "—")
        eligible = result.get("eligible", False)

        badge_class = "badge-eligible" if eligible else "badge-ineligible"
        band_class = f"badge-{band.lower()}" if band in ("HIGH", "MED", "LOW") else "badge-review"

        label = (
            f'<span class="badge {badge_class}">{"Eligible" if eligible else "Not Eligible"}</span>'
            f'&nbsp;<span class="badge {band_class}">{band}</span>'
        )
        with st.expander(f"{pid} — {result.get('project_name', '')}"):
            st.markdown(label, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown("**Confidence**")
                st.progress(min(float(result.get("confidence", 0)), 1.0))
                st.caption(f"{result.get('confidence', 0):.1%} — {band}")

                if result.get("primary_risk"):
                    st.warning(f"**Risk:** {result['primary_risk']}")

                st.markdown("**Rationale**")
                st.write(result.get("overall_rationale") or result.get("rationale", "—"))

            with col_r:
                four_part = result.get("four_part_test", {})
                if four_part:
                    st.markdown("**Four-Part Test**")
                    _criterion_color = {"met": "#28a745", "uncertain": "#ffc107", "not_met": "#dc3545"}
                    _criterion_label = {"met": "Met", "uncertain": "Uncertain", "not_met": "Not Met"}
                    for k, v in four_part.items():
                        color = _criterion_color.get(str(v), "#6c757d")
                        label = _criterion_label.get(str(v), str(v).replace("_", " ").title())
                        st.markdown(
                            f"**{k.replace('_', ' ').title()}:** "
                            f'<span style="color:{color};font-weight:600">{label}</span>',
                            unsafe_allow_html=True,
                        )

                flippers = result.get("decision_flippers", [])
                if flippers:
                    st.markdown("**Decision Sensitivity**")
                    for f in flippers:
                        st.caption(f"• {f}")

    st.markdown("---")

    # Expert review queue
    needs_review = [
        pid
        for pid in st.session_state.get("needs_review_ids", [])
        if pid in st.session_state["projects"]
    ]
    if needs_review:
        st.markdown('<div class="section-label">Expert Review Queue</div>', unsafe_allow_html=True)
        st.info(f"{len(needs_review)} project(s) require human review (confidence below threshold).")
        selected_pid = st.selectbox("Select project", needs_review)
        selected_proj = st.session_state["projects"][selected_pid]
        ai_decision = selected_proj["ai_decision"]

        with st.form(f"review_{selected_pid}"):
            st.markdown(f"**AI decision:** {ai_decision['label']} ({ai_decision['confidence']:.1%})")
            final_label = st.radio(
                "Final Decision",
                ["Eligible", "Not Eligible"],
                index=0 if ai_decision["label"] == "Eligible" else 1,
                horizontal=True,
            )
            final_rationale = st.text_area("Expert Rationale", height=100, placeholder="Provide justification for the override…")
            submitted = st.form_submit_button("Commit Review", type="primary")
            if submitted:
                st.session_state["projects"][selected_pid]["human_decision"] = {
                    "final_label": final_label,
                    "rationale": final_rationale,
                    "confidence": 1.0,
                }
                st.session_state["projects"][selected_pid]["status"] = "Reviewed"
                st.session_state["needs_review_ids"].remove(selected_pid)
                st.session_state["results_df"] = build_display_df()
                st.rerun()

    st.markdown("---")

    # Export
    st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
    project_ids = results_df["ID"].astype(str).tolist()
    selected_project = st.selectbox("Project", project_ids)

    with st.expander("Form 6765 Configuration", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            tax_year = st.number_input("Tax Year", value=datetime.utcnow().year, step=1)
            name_on_return = st.text_input("Name on Return", value="Sample Taxpayer Inc.")
            identifying_number = st.text_input("EIN / Identifying Number", value="00-0000000")
            ruleset_version = st.text_input("Ruleset Version", value="2024.1")
            created_by = st.text_input("Created By", value=user_id or "streamlit-user")
            override_reason = st.text_area("Override Reason (optional, 30+ chars)", height=80)
            override_role = st.selectbox("Override Role", ["", "ADMIN", "PARTNER", "DIRECTOR"])
        with c2:
            qre_wages = st.number_input("QRE Wages ($)", value=0.0, format="%.2f")
            qre_supplies = st.number_input("QRE Supplies ($)", value=0.0, format="%.2f")
            qre_contract = st.number_input("Contract Research ($)", value=0.0, format="%.2f")
            credit_method = st.selectbox("Credit Method", ["ASC", "REGULAR"])
            section_280c = st.selectbox("Section 280C", ["FULL", "REDUCED"])

    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("Generate Form 6765", use_container_width=True):
            payload = {
                "header": {
                    "tax_year": int(tax_year),
                    "name_on_return": name_on_return,
                    "identifying_number": identifying_number,
                },
                "inputs": {
                    "qre_wages": qre_wages,
                    "qre_supplies": qre_supplies,
                    "qre_contract_research_gross": qre_contract,
                    "credit_method": credit_method,
                    "section_280c_choice": section_280c,
                },
                "project_ids": [selected_project],
                "ruleset_version": ruleset_version,
                "created_by": created_by,
                "override_reason": override_reason or None,
                "save_pdf": True,
            }
            headers = {"X-API-Key": api_key}
            if override_role:
                headers["X-Role"] = override_role
            with st.spinner("Generating Form 6765…"):
                r = requests.post(f"{backend_url}/form6765/generate", json=payload, headers=headers)
                if r.status_code == 200:
                    fid = r.json().get("form_version", {}).get("form_version_id")
                    if fid:
                        pr = requests.get(f"{backend_url}/form6765/form/{fid}/pdf", headers=headers)
                        st.download_button(
                            "Download Form 6765 PDF",
                            pr.content,
                            f"form6765_{fid}.pdf",
                            "application/pdf",
                            use_container_width=True,
                        )
                    st.success("Form 6765 generated.")
                else:
                    st.error(r.text)

    with btn_col2:
        if st.button("Download Audit Package", use_container_width=True):
            r = requests.post(
                f"{backend_url}/audit_package",
                data={"project_id": selected_project},
                headers={"X-API-Key": api_key},
            )
            if r.status_code == 200:
                st.download_button(
                    "Save Audit Package (.zip)",
                    r.content,
                    f"audit_{selected_project}.zip",
                    "application/zip",
                    use_container_width=True,
                )
            else:
                st.error(r.text)
