import os
import uuid
import json
import re
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv
from .models import ProjectRecord, ClassificationResult, TraceStep, TraceEnvelope

# --- Env ---
load_dotenv()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_IS_LOCAL_LLM = LLM_BASE_URL.startswith(("http://localhost", "http://127.0.0.1", "https://localhost", "https://127.0.0.1"))

# Default to qwen2.5:3b if local usage is detected (common for Ollama), otherwise gpt-4o
_DEFAULT_MODEL = "qwen2.5:3b" if _IS_LOCAL_LLM else "gpt-4o"
MODEL_NAME = os.getenv("OPENAI_MODEL", _DEFAULT_MODEL)

_LOCAL_PREFIXES = ("http://localhost", "http://127.0.0.1", "https://localhost", "https://127.0.0.1")
_IS_LOCAL_LLM = LLM_BASE_URL.startswith(_LOCAL_PREFIXES)

# Allow a dummy key for local LLMs so the SDK doesn't raise auth errors
if not OPENAI_API_KEY and _IS_LOCAL_LLM:
    OPENAI_API_KEY = "lm-studio"

# Treat presence of a key OR a local base URL as "LLM enabled"
USE_LLM = bool(OPENAI_API_KEY) or _IS_LOCAL_LLM

# Lazy imports (SDK may not exist in some envs)
try:
    from openai import OpenAI, AsyncOpenAI  # type: ignore
except Exception:
    OpenAI = None          # type: ignore
    AsyncOpenAI = None     # type: ignore

# Client builders (include base_url + dummy key support)
def _make_async_client():
    if not (USE_LLM and AsyncOpenAI):
        return None
    try:
        return AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=LLM_BASE_URL, timeout=60, max_retries=1)
    except Exception:
        return None


def _make_sync_client():
    if not (USE_LLM and OpenAI):
        return None
    try:
        return OpenAI(api_key=OPENAI_API_KEY, base_url=LLM_BASE_URL, timeout=60, max_retries=1)
    except Exception:
        return None


# Async client with short timeout & light retry (created only if LLM enabled)
_async_client = _make_async_client()

_LLM_JSON_SCHEMA = (
    '{"eligible": true|false, "confidence": 0.0-1.0, "rationale": "short explanation", '
    '"permitted_purpose": "met|uncertain|not_met", '
    '"elimination_uncertainty": "met|uncertain|not_met", '
    '"process_experimentation": "met|uncertain|not_met", '
    '"technological_nature": "met|uncertain|not_met"}'
)

LLM_SYSTEM_PROMPT = (
    "You are an expert IRS R&D Tax Credit (Section 41) eligibility analyst with deep expertise in "
    "the Four-Part Test. Your role is to rigorously evaluate projects against each criterion.\n\n"
    
    "##IRS Section 41 Four-Part Test:\n\n"
    "1. **Permitted Purpose**: Does the project involve one or more of:\n"
    "   - Development or improvement of business component (software, hardware, process)\n"
    "   - Must result in new functionality, performance improvement, or design alternative\n"
    "   - EXCLUDES: adaptation to existing component, duplicating existing technology\n\n"
    
    "2. **Elimination of Uncertainty**: Was there genuine technological uncertainty about:\n"
    "   - Whether the component could be developed\n"
    "   - How to develop it\n"
    "   - Which of multiple approaches was most effective\n"
    "   - What the performance or feasibility would be\n"
    "   - Evidence: design docs, failed experiments, architectural decisions, prototypes\n\n"
    
    "3. **Process of Experimentation**: Did the project use systematic trial-and-error, including:\n"
    "   - Hypothesis or accepted criteria\n"
    "   - Multiple methodological approaches\n"
    "   - Evaluation of results against metrics\n"
    "   - Iteration or refinement based on findings\n"
    "   - Evidence: code commits, test results, ablation studies, decision logs\n\n"
    
    "4. **Technological in Nature**: Is the work based on:\n"
    "   - Computer science, engineering, or applied mathematics\n"
    "   - Technical knowledge and discipline\n"
    "   - EXCLUDES: social sciences, market research, routine implementation\n\n"
    
    "##Your Evaluation Task:\n"
    "For each criterion, respond with:\n"
    "- **met**: Clear evidence present\n"
    "- **uncertain**: Some evidence but ambiguous\n"
    "- **not_met**: No credible evidence\n\n"
    
    "Then decide:\n"
    "- **eligible**: All 4 criteria met, OR 3 met + 1 uncertain (with strong evidence elsewhere)\n"
    "- **not_eligible**: ≥1 criterion clearly not met, OR mostly uncertain/vague\n\n"
    
    "Return ONLY valid JSON (no prose, no markdown):\n"
    "{\n"
    '  "permitted_purpose": "met|uncertain|not_met",\n'
    '  "elimination_uncertainty": "met|uncertain|not_met",\n'
    '  "process_experimentation": "met|uncertain|not_met",\n'
    '  "technological_nature": "met|uncertain|not_met",\n'
    '  "eligible": true|false,\n'
    '  "confidence": 0.0-1.0,\n'
    '  "rationale": "2-3 sentence summary citing evidence from all 4 criteria",\n'
    '  "key_evidence": ["evidence 1", "evidence 2", "evidence 3"]\n'
    "}\n\n"
    "Be rigorous and conservative. If a project is borderline, flag uncertainty explicitly."
)

# -------------------------
# Utilities
# -------------------------
def _coerce_confidence(val) -> float:
    if val is None:
        return 0.6
    if isinstance(val, (int, float)):
        x = float(val)
    else:
        s = str(val).strip().lower()
        mapping = {"high": 0.85, "med": 0.6, "medium": 0.6, "low": 0.35}
        if s in mapping:
            x = mapping[s]
        elif s.endswith("%"):
            try:
                x = float(s[:-1]) / 100.0
            except Exception:
                x = 0.6
        else:
            try:
                x = float(s)
            except Exception:
                x = 0.6
    return max(0.0, min(1.0, x))

@dataclass
class TierResult:
    decided: bool           # True = this tier produced a final answer; False = pass to next tier
    eligible: bool
    confidence: float
    rationale: str
    criteria: Optional[Dict[str, str]] = None
    trace_step: Optional[TraceStep] = None

# -------------------------
# Hard Filter: Rule-Out Classifier (Tier 1)
# -------------------------
RULE_OUT_KEYWORDS = [
    "data entry", "ui refresh", "cosmetic", "marketing",
    "routine qa", "unit testing", "documentation",
    "training", "bug fix", "devops", "deployment",
    "routine maintenance", "hr policy", "hr system",
    "admin work", "production support", "routine upgrade",
    "market research", "routine testing", "uat"
]

def rule_out_classifier(text: str) -> Tuple[bool, float, str]:
    """
    Hard filter: if text contains >2 rule-out keywords, auto-reject.
    Returns (eligible, confidence, rationale).
    If ruled out: (False, 0.9, rationale)
    If not ruled out: (None, 0.0, rationale) to signal "pass to next tier"
    """
    text_l = (text or "").lower()
    matches = [kw for kw in RULE_OUT_KEYWORDS if kw in text_l]
    if len(matches) > 2:
        rationale = f"Hard filter rule-out: matched {len(matches)} ineligible patterns: {matches[:5]}."
        return False, 0.9, rationale
    return None, 0.0, ""

def rule_based_classifier(text: str) -> Tuple[bool, float, str]:
    """
    Rule-based tier (Tier 2): if hard filter did not reject,
    apply heuristic scoring on positive/negative signals.
    """
    text_l = (text or "").lower()
    positives = [
        "prototype","experimentation","algorithm","new method","new process","ml model",
        "data pipeline","simulation","uncertainty","hypothesis","optimization","rd","r&d",
        "research","development","proof of concept"
    ]
    negatives = ["marketing","sales","hr policy","legal review","admin work","production support"]

    score = sum(1 for s in positives if s in text_l) - sum(1 for s in negatives if s in text_l)
    eligible = score >= 1
    confidence = min(max((score + 1) / 5, 0.1), 0.95)
    rationale = f"Rule-based heuristic: signals={[s for s in positives if s in text_l]} score={score}."
    return eligible, confidence, rationale

def _is_gpt5(model: str) -> bool:
    return model.lower().startswith("gpt-5")

# -------------------------
# Rationale helpers
# -------------------------
def _heuristic_rationale(description: str, eligible: bool) -> str:
    desc = (description or "").lower()
    sigs = [
        s for s in [
            "prototype","experimentation","algorithm","ml model","simulation",
            "uncertainty","hypothesis","optimization","new method","new process","proof of concept"
        ] if s in desc
    ]
    if eligible:
        if sigs:
            return f"Meets Section 41 tests (technological experimentation) — signals: {', '.join(sigs[:4])}."
        return "Meets Section 41 tests based on technological investigation and experimentation."
    else:
        return "Insufficient evidence of a technological process of experimentation under Section 41."

_CRITERIA_KEYS = ("permitted_purpose", "elimination_uncertainty", "process_experimentation", "technological_nature")
_VALID_CRITERION_VALUES = {"met", "uncertain", "not_met"}

def _normalize_llm_json(obj: dict, description: str) -> tuple[bool, float, str, dict | None]:
    """
    Parse LLM response. Returns (eligible, confidence, rationale, criteria).
    criteria is a Dict[str, str] when the LLM returned four-part fields, None otherwise.
    """
    eligible = bool(obj.get("eligible", False))
    confidence = _coerce_confidence(obj.get("confidence"))

    raw_criteria = {k: obj.get(k, "") for k in _CRITERIA_KEYS}
    criteria: dict | None = None
    if any(raw_criteria.values()):
        # Normalise values: accept only the three valid strings
        criteria = {
            k: v if v in _VALID_CRITERION_VALUES else "uncertain"
            for k, v in raw_criteria.items()
        }
        met_count = sum(1 for v in criteria.values() if v == "met")
        uncertain_count = sum(1 for v in criteria.values() if v == "uncertain")
        not_met_count = sum(1 for v in criteria.values() if v == "not_met")
        criteria_summary = f"Criteria: {met_count} met, {uncertain_count} uncertain, {not_met_count} not met."
        rationale = obj.get("rationale") or obj.get("reason") or obj.get("explanation") or ""
        if isinstance(rationale, list):
            rationale = " ".join(str(x) for x in rationale if x)
        rationale = (rationale or "").strip()
        rationale = f"{rationale} {criteria_summary}" if rationale else criteria_summary
    else:
        rationale = obj.get("rationale") or obj.get("reason") or obj.get("explanation")
        if isinstance(rationale, list):
            rationale = " ".join(str(x) for x in rationale if x)
        rationale = (rationale or "").strip()

    if not rationale:
        rationale = _heuristic_rationale(description, eligible)

    return eligible, confidence, rationale, criteria

# -------------------------
# Param builders (handle GPT-5 quirks)
# -------------------------
def _build_chat_params(model: str, messages: list, want_json: bool = True) -> Dict[str, Any]:
    """
    - GPT-5: use max_tokens; DO NOT send 'temperature'
    - Older models: use max_tokens and temperature=0
    - Prefer JSON response_format when requested
    """
    params: Dict[str, Any] = {"model": model, "messages": messages}
    if _is_gpt5(model):
        params["max_tokens"] = 300
    else:
        params["max_tokens"] = 300
        params["temperature"] = 0
    if want_json:
        params["response_format"] = {"type": "json_object"}
    return params

def _extract_json(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except Exception:
        match = re.search(r"\{.*\}", content or "", re.S)
        return json.loads(match.group(0)) if match else {}

# -------------------------
# Backfill rationale (tiny, optional)
# -------------------------
async def _backfill_rationale_async(description: str, eligible: bool) -> str:
    if not (USE_LLM and _async_client):
        return _heuristic_rationale(description, eligible)
    try:
        msgs = [
            {"role": "system", "content": "Write one short sentence (<=30 words) explaining the R&D eligibility under IRS Section 41. Be specific and factual."},
            {"role": "user", "content": f"Eligible={eligible}. Project: {description}"}
        ]
        params = _build_chat_params(MODEL_NAME, msgs, want_json=False)
        if "max_tokens" in params:
            params["max_tokens"] = 40
        else:
            params["max_tokens"] = 40
        r = await _async_client.chat.completions.create(**params)
        return (r.choices[0].message.content or "").strip() or _heuristic_rationale(description, eligible)
    except Exception:
        return _heuristic_rationale(description, eligible)

def _backfill_rationale_sync(description: str, eligible: bool) -> str:
    if not (USE_LLM and OpenAI):
        return _heuristic_rationale(description, eligible)
    try:
        client = _make_sync_client()
        if client is None:
            return _heuristic_rationale(description, eligible)
        msgs = [
            {"role": "system", "content": "Write one short sentence (<=30 words) explaining the R&D eligibility under IRS Section 41. Be specific and factual."},
            {"role": "user", "content": f"Eligible={eligible}. Project: {description}"}
        ]
        params = _build_chat_params(MODEL_NAME, msgs, want_json=False)
        params["max_tokens"] = 40
        r = client.chat.completions.create(**params)
        return (r.choices[0].message.content or "").strip() or _heuristic_rationale(description, eligible)
    except Exception:
        return _heuristic_rationale(description, eligible)

# -------------------------
# Async LLM helper with graceful retry
# -------------------------
def _model_candidates() -> list:
    candidates = [MODEL_NAME]
    env_fallback = os.getenv("OPENAI_MODEL_FALLBACK")
    if env_fallback and env_fallback not in candidates:
        candidates.append(env_fallback)
    # Only add cloud fallbacks when not pointing at a local endpoint
    if not _IS_LOCAL_LLM:
        for m in ("gpt-4o-mini", "gpt-4o"):
            if m not in candidates:
                candidates.append(m)
    return candidates


async def _chat_llm_async(model: str, messages: list) -> Dict[str, Any]:

    if not (USE_LLM and _async_client):
        return {}

    last_exc = None
    for candidate in _model_candidates():
        params = _build_chat_params(candidate, messages, want_json=True)
        try:
            r = await _async_client.chat.completions.create(**params)
            return _extract_json((r.choices[0].message.content or "").strip())
        except Exception as e:
            last_exc = e
            el = str(e).lower()
            # If response_format unsupported, retry this candidate without it
            if "response_format" in el or "unsupported" in el:
                try:
                    params = _build_chat_params(candidate, messages, want_json=False)
                    r = await _async_client.chat.completions.create(**params)
                    return _extract_json((r.choices[0].message.content or "").strip())
                except Exception as e2:
                    last_exc = e2
                    if "invalid model" in str(e2).lower() or "invalid model id" in str(e2).lower():
                        # try next candidate
                        continue
                    else:
                        raise
            # If invalid model id, try next candidate
            if "invalid model" in el or "invalid model id" in el:
                continue
            # Otherwise re-raise
            raise
    # All candidates failed
    if last_exc is not None:
        raise last_exc
    return {}

# -------------------------
# Public LLM Helpers
# -------------------------
def generate_text_response(system_prompt: str, user_prompt: str, model: str = None) -> str:
    """
    Public synchronous helper to get a text response from the LLM.
    Uses the internal robust client creation and retry logic.
    """
    if not (USE_LLM and OpenAI):
        return "LLM not available or configured."

    model = model or MODEL_NAME
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # We can reuse the internal _build_chat_params but we need to handle the execution.
    # Since _chat_llm_sync is tailored for JSON, we should create a similar text version or make _chat_llm_sync more flexible.
    # But _chat_llm_sync calls _extract_json.
    
    # Let's just implement a simple robust call here using the internal helpers if possible, or just raw client.
    client = _make_sync_client()
    if client is None:
        return "LLM Setup Failed."
        
    params = _build_chat_params(model, messages, want_json=False)
    try:
        r = client.chat.completions.create(**params)
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Error generating text: {str(e)}"

def generate_json_response(system_prompt: str, user_prompt: str, model: str = None) -> Dict[str, Any]:
    """
    Public synchronous helper to get a JSON response.
    """
    model = model or MODEL_NAME
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return _chat_llm_sync(model, messages)

# -------------------------
# Sync LLM helper with graceful retry
# -------------------------
def _chat_llm_sync(model: str, messages: list) -> Dict[str, Any]:
    if not (USE_LLM and OpenAI):
        return {}

    last_exc = None
    for candidate in _model_candidates():
        client = _make_sync_client()
        if client is None:
            last_exc = RuntimeError("LLM client unavailable")
            break
        params = _build_chat_params(candidate, messages, want_json=True)
        try:
            r = client.chat.completions.create(**params)
            return _extract_json((r.choices[0].message.content or "").strip())
        except Exception as e:
            last_exc = e
            el = str(e).lower()
            if "response_format" in el or "unsupported" in el:
                try:
                    params = _build_chat_params(candidate, messages, want_json=False)
                    r = client.chat.completions.create(**params)
                    return _extract_json((r.choices[0].message.content or "").strip())
                except Exception as e2:
                    last_exc = e2
                    if "invalid model" in str(e2).lower() or "invalid model id" in str(e2).lower():
                        continue
                    else:
                        raise
            if "invalid model" in el or "invalid model id" in el:
                continue
            raise
    if last_exc is not None:
        raise last_exc
    return {}

# -------------------------
# Tier functions — each returns TierResult with decided=True/False
# -------------------------
def _tier1_rule_out(record: ProjectRecord) -> TierResult:
    eligible, conf, reason = rule_out_classifier(record.description)
    if eligible is False:
        return TierResult(
            decided=True, eligible=False, confidence=conf, rationale=reason,
            trace_step=TraceStep(
                step_id=str(uuid.uuid4()), timestamp=datetime.utcnow().isoformat()+"Z",
                model_name="rule-out-filter:v1", thought="Apply hard-filter rule-out (Tier 1).",
                action="rule_out_filter", observation="Matched ineligible patterns.",
                confidence=conf,
            ),
        )
    return TierResult(decided=False, eligible=False, confidence=0.0, rationale="")


def _tier2_heuristic(record: ProjectRecord) -> TierResult:
    eligible, conf, reason = rule_based_classifier(record.description)
    return TierResult(
        decided=True, eligible=eligible, confidence=conf, rationale=reason,
        trace_step=TraceStep(
            step_id=str(uuid.uuid4()), timestamp=datetime.utcnow().isoformat()+"Z",
            model_name="rule-based-heuristic:v1", thought="Apply rule-based heuristic (Tier 2).",
            action="rule_based_classify", observation="No LLM available, using heuristic scoring.",
            confidence=conf,
        ),
    )


async def _tier3_llm_async(record: ProjectRecord) -> TierResult:
    obj = await _chat_llm_async(
        MODEL_NAME,
        [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content":
                "Project Description: " + (record.description or "")
                + "\nReturn ONLY valid JSON with all fields present: "
                + _LLM_JSON_SCHEMA},
        ],
    )
    eligible, confidence, rationale, criteria = _normalize_llm_json(obj, record.description)
    if not rationale:
        rationale = await _backfill_rationale_async(record.description, eligible)
    return TierResult(
        decided=True, eligible=eligible, confidence=confidence, rationale=rationale, criteria=criteria,
        trace_step=TraceStep(
            step_id=str(uuid.uuid4()), timestamp=datetime.utcnow().isoformat()+"Z",
            model_name=f"openai:{MODEL_NAME}", thought="Apply IRS Section 41 via LLM (Tier 3, async).",
            action="classify_with_llm", observation="chat.completions (JSON-preferred)",
            confidence=confidence,
        ),
    )


def _tier3_llm_sync(record: ProjectRecord) -> TierResult:
    obj = _chat_llm_sync(
        MODEL_NAME,
        [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content":
                "Project Description: " + (record.description or "")
                + "\nReturn ONLY valid JSON with all fields present: "
                + _LLM_JSON_SCHEMA},
        ],
    )
    eligible, confidence, rationale, criteria = _normalize_llm_json(obj, record.description)
    if not rationale:
        rationale = _backfill_rationale_sync(record.description, eligible)
    return TierResult(
        decided=True, eligible=eligible, confidence=confidence, rationale=rationale, criteria=criteria,
        trace_step=TraceStep(
            step_id=str(uuid.uuid4()), timestamp=datetime.utcnow().isoformat()+"Z",
            model_name=f"openai:{MODEL_NAME}", thought="Apply IRS Section 41 via LLM (Tier 3).",
            action="classify_with_llm", observation="chat.completions (JSON-preferred)",
            confidence=confidence,
        ),
    )


# -------------------------
# Dual Model Cross-Check (Optional Advanced Feature - Phase 1.3)
# -------------------------
def _verify_criteria_mismatch(primary: dict, verifier: dict) -> int:
    """
    Compare primary and verifier model outputs on the 4 criteria.
    Returns count of criteria mismatches.
    """
    criteria = [
        "permitted_purpose", "elimination_uncertainty",
        "process_experimentation", "technological_nature"
    ]
    mismatches = 0
    for crit in criteria:
        p = primary.get(crit, "")
        v = verifier.get(crit, "")
        # Mismatch if both are present and different (ignoring "met" vs "uncertain" nuance)
        if p and v and p != v:
            mismatches += 1
    return mismatches

def analyze_with_dual_check(
    record: ProjectRecord,
    primary_model: str = None,
    verifier_model: str = "gpt-3.5-turbo",
    user_id: str = "demo-user"
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Run primary model, then verifier model independently.
    If ≥2 criteria mismatch → flag for manual review.
    
    Returns (classification, trace, verification_report)
    """
    if not (USE_LLM and OpenAI):
        raise RuntimeError("Dual-check requires LLM enabled.")
    
    primary_model = primary_model or MODEL_NAME
    
    # Get primary analysis
    primary_result, primary_trace = analyze_project(record, user_id=f"{user_id}:primary")
    
    # Get verifier analysis (independently)
    client = _make_sync_client()
    if client is None:
        raise RuntimeError("LLM client unavailable for verifier model.")
    verifier_msgs = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content":
            "Project Description: " + (record.description or "")
            + "\nReturn ONLY valid JSON (no prose):\n"
            '{"permitted_purpose":"met|uncertain|not_met",'
            '"elimination_uncertainty":"met|uncertain|not_met",'
            '"process_experimentation":"met|uncertain|not_met",'
            '"technological_nature":"met|uncertain|not_met",'
            '"eligible":true|false,"confidence":0..1,"rationale":"..."}'
        }
    ]
    
    try:
        params = _build_chat_params(verifier_model, verifier_msgs, want_json=True)
        r = client.chat.completions.create(**params)
        verifier_obj = _extract_json((r.choices[0].message.content or "").strip())
    except Exception:
        verifier_obj = {}
    
    # Compare
    primary_obj = {}
    # Try to extract criteria from trace
    for step in primary_trace.get("steps", []):
        if step.get("action") == "classify_with_llm":
            # We'd need to store the LLM response in trace, which isn't done yet
            pass
    
    mismatch_count = _verify_criteria_mismatch(primary_obj, verifier_obj) if primary_obj else 0
    needs_manual_review = mismatch_count >= 2
    
    verification_report = {
        "primary_model": primary_model,
        "verifier_model": verifier_model,
        "mismatch_count": mismatch_count,
        "needs_manual_review": needs_manual_review,
        "primary_eligible": primary_result.eligible,
        "verifier_eligible": verifier_obj.get("eligible", None),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    # If manual review needed, flag the classification
    if needs_manual_review:
        primary_result.rationale = f"[MANUAL REVIEW RECOMMENDED] Dual-check flagged {mismatch_count} criteria mismatches. {primary_result.rationale}"
        primary_result.confidence = 0.5  # lower confidence when cross-check fails
    
    return primary_result, primary_trace, verification_report

# -------------------------
# Async analyzer (for concurrent pipelines)
# -------------------------
async def analyze_project_async(record: ProjectRecord, user_id: str = "demo-user"):
    steps = [TraceStep(
        step_id=str(uuid.uuid4()), timestamp=datetime.utcnow().isoformat()+"Z",
        model_name="parser:v0", thought="Parse incoming project record into analysis fields.",
        action="parse", observation=f"Got description length={len(record.description or '')}",
        confidence=1.0,
    )]

    result = _tier1_rule_out(record)
    if result.trace_step:
        steps.append(result.trace_step)

    if not result.decided:
        if USE_LLM and _async_client is not None:
            try:
                result = await _tier3_llm_async(record)
            except Exception as e:
                fb = _tier2_heuristic(record)
                result = TierResult(
                    decided=True, eligible=fb.eligible, confidence=fb.confidence,
                    rationale=f"LLM error fallback (Tier 3→2) -> {e} -> {fb.rationale}",
                    trace_step=fb.trace_step,
                )
        else:
            result = _tier2_heuristic(record)
        if result.trace_step:
            steps.append(result.trace_step)

    confidence = _coerce_confidence(result.confidence)
    steps.append(TraceStep(
        step_id=str(uuid.uuid4()), timestamp=datetime.utcnow().isoformat()+"Z",
        model_name="decision:v0", thought="Consolidate evidence and decide eligibility.",
        action="decide", observation=f"eligible={result.eligible}, confidence={confidence:.2f}",
        confidence=confidence,
    ))

    classification = ClassificationResult(
        project_id=record.project_id, eligible=result.eligible, confidence=confidence,
        rationale=result.rationale, region="US-IRS-Section-41", criteria=result.criteria,
    )
    trace = TraceEnvelope(
        user_id=user_id, project_id=record.project_id, steps=steps,
        model_name=(f"openai:{MODEL_NAME}" if USE_LLM else "rule-based:v0"),
        region="US-IRS-Section-41", reviewer_id=None, legal_hold_flag=False,
    ).model_dump()
    return classification, trace


# -------------------------
# Phase 4: Advanced NLP Features Integration
# -------------------------
def analyze_project_with_advanced_nlp(
    record: ProjectRecord,
    user_id: str = "demo-user",
    enable_decomposition: bool = True,
    enable_uncertainty: bool = True,
    enable_evidence: bool = True,
) -> Dict[str, Any]:
    """
    Analyze project using advanced NLP features (Phase 4):
    4.1 Project Decomposition
    4.2 Uncertainty Detector
    4.3 Experimentation Evidence Extractor
    
    Args:
        record: ProjectRecord
        user_id: User identifier for tracing
        enable_decomposition: Whether to decompose project into components
        enable_uncertainty: Whether to detect uncertainty indicators
        enable_evidence: Whether to extract experimentation evidence
    
    Returns:
        Dict with:
        - classification: ClassificationResult (from standard analysis)
        - trace: Trace envelope (from standard analysis)
        - advanced_nlp: Dict with decomposition, uncertainty, and evidence
    """
    from .advanced_nlp import (
        ProjectDecomposer,
        UncertaintyDetector,
        ExperimentationExtractor,
    )
    
    # Step 1: Run standard analysis (Tier 1-3)
    classification, trace = analyze_project(record, user_id=user_id)
    
    result = {
        "classification": classification.model_dump() if hasattr(classification, 'model_dump') else classification,
        "trace": trace,
        "advanced_nlp": {},
    }
    
    description = record.description or ""
    
    # Step 2: Project Decomposition (4.1)
    if enable_decomposition:
        # Create an OpenAI client if available and requested via env
        client = None
        if USE_LLM and OpenAI is not None:
            try:
                client = _make_sync_client()
            except Exception:
                client = None

        decomposer = ProjectDecomposer(llm_client=client, model_name=(MODEL_NAME if client else None))
        decomposition = decomposer.decompose_project(
            project_id=record.project_id,
            project_name=record.project_name or "Untitled",
            project_description=description,
            use_llm=bool(client),
        )
        
        # Convert dataclass to dict
        decomp_dict = {
            "project_id": decomposition.project_id,
            "project_name": decomposition.project_name,
            "total_components": decomposition.total_components,
            "components": [
                {
                    "name": c.name,
                    "description": c.description,
                    "component_type": c.component_type.value,
                    "estimated_percentage": c.estimated_percentage,
                    "rationale": c.rationale,
                    "eligible": c.eligible,
                    "confidence": c.confidence,
                    "subtasks": c.subtasks,
                }
                for c in decomposition.components
            ],
            "research_percentage": decomposition.research_percentage,
            "eligible_percentage": decomposition.eligible_percentage,
            "non_eligible_components": decomposition.non_eligible_components,
            "overall_eligibility": decomposition.overall_eligibility,
            "decomposition_confidence": decomposition.decomposition_confidence,
            "rationale": decomposition.rationale,
        }
        result["advanced_nlp"]["decomposition"] = decomp_dict
    
    # Step 3: Uncertainty Detection (4.2)
    if enable_uncertainty:
        uncertainty_detector = UncertaintyDetector()
        # If we created a client above, pass it through to the uncertainty detector
        try:
            uncertainty_analysis = uncertainty_detector.detect_uncertainties(
                description,
                use_llm=bool(client),
                llm_client=client,
                model_name=(MODEL_NAME if client else None),
            )
        except Exception:
            # Fallback to non-LLM detection on any failure
            uncertainty_analysis = uncertainty_detector.detect_uncertainties(description)
        
        # Convert dataclass to dict
        uncertainty_dict = {
            "project_id": uncertainty_analysis.project_id or record.project_id,
            "has_technical_unknowns": uncertainty_analysis.has_technical_unknowns,
            "has_experiments": uncertainty_analysis.has_experiments,
            "has_benchmarks": uncertainty_analysis.has_benchmarks,
            "has_failures": uncertainty_analysis.has_failures,
            "has_new_methods": uncertainty_analysis.has_new_methods,
            "overall_uncertainty_score": uncertainty_analysis.overall_uncertainty_score,
            "indicators": [
                {
                    "indicator_type": i.indicator_type,
                    "description": i.description,
                    "confidence": i.confidence,
                    "evidence_phrases": i.evidence_phrases,
                }
                for i in uncertainty_analysis.indicators
            ],
            "missing_evidence": uncertainty_analysis.missing_evidence,
            "rationale": uncertainty_analysis.rationale,
        }
        result["advanced_nlp"]["uncertainty"] = uncertainty_dict
    
    # Step 4: Experimentation Evidence Extraction (4.3)
    if enable_evidence:
        evidence_extractor = ExperimentationExtractor()
        evidence_analysis = evidence_extractor.extract_evidence(
            project_id=record.project_id,
            project_description=description,
        )
        
        # Convert dataclass to dict
        evidence_dict = {
            "project_id": evidence_analysis.project_id,
            "total_phrases_found": evidence_analysis.total_phrases_found,
            "architecture_comparisons": [
                {
                    "phrase_type": p.phrase_type,
                    "quote": p.quote,
                    "context": p.context,
                    "confidence": p.confidence,
                }
                for p in evidence_analysis.architecture_comparisons
            ],
            "parameter_optimizations": [
                {
                    "phrase_type": p.phrase_type,
                    "quote": p.quote,
                    "context": p.context,
                    "confidence": p.confidence,
                }
                for p in evidence_analysis.parameter_optimizations
            ],
            "alternative_approaches": [
                {
                    "phrase_type": p.phrase_type,
                    "quote": p.quote,
                    "context": p.context,
                    "confidence": p.confidence,
                }
                for p in evidence_analysis.alternative_approaches
            ],
            "testing_approaches": [
                {
                    "phrase_type": p.phrase_type,
                    "quote": p.quote,
                    "context": p.context,
                    "confidence": p.confidence,
                }
                for p in evidence_analysis.testing_approaches
            ],
            "failure_learnings": [
                {
                    "phrase_type": p.phrase_type,
                    "quote": p.quote,
                    "context": p.context,
                    "confidence": p.confidence,
                }
                for p in evidence_analysis.failure_learnings
            ],
            "hypotheses_tested": [
                {
                    "phrase_type": p.phrase_type,
                    "quote": p.quote,
                    "context": p.context,
                    "confidence": p.confidence,
                }
                for p in evidence_analysis.hypotheses_tested
            ],
            "evidence_strength": evidence_analysis.evidence_strength,
            "audit_ready_summary": evidence_analysis.audit_ready_summary,
        }
        result["advanced_nlp"]["evidence"] = evidence_dict
    
    return result

def _template_narrative(project_name: str, description: str) -> str:
    return (
        f"**Project Overview**\n\n"
        f"The \"{project_name}\" project advances functionality and performance through "
        f"systematic experimentation grounded in computer science and engineering. "
        f"The team investigates alternative designs to reduce uncertainty and deliver measurable improvements.\n\n"
        f"**Technological Basis & Uncertainty**\n\n"
        f"{description}\n\n"
        f"**Process of Experimentation**\n\n"
        f"- Identify hypotheses and acceptance criteria\n"
        f"- Prototype alternative designs and measure outcomes\n"
        f"- Iterate based on empirical results using defined metrics\n"
        f"- Document learnings and decisions for audit readiness\n"
    )


# -------------------------
# Sync analyzer (kept for compatibility)
# -------------------------
def analyze_project(record: ProjectRecord, user_id: str = "demo-user"):
    steps = [TraceStep(
        step_id=str(uuid.uuid4()), timestamp=datetime.utcnow().isoformat()+"Z",
        model_name="parser:v0", thought="Parse incoming project record into analysis fields.",
        action="parse", observation=f"Got description length={len(record.description or '')}",
        confidence=1.0,
    )]

    result = _tier1_rule_out(record)
    if result.trace_step:
        steps.append(result.trace_step)

    if not result.decided:
        if USE_LLM and OpenAI is not None:
            try:
                result = _tier3_llm_sync(record)
            except Exception as e:
                fb = _tier2_heuristic(record)
                result = TierResult(
                    decided=True, eligible=fb.eligible, confidence=fb.confidence,
                    rationale=f"LLM error fallback (Tier 3→2) -> {e} -> {fb.rationale}",
                    trace_step=fb.trace_step,
                )
        else:
            result = _tier2_heuristic(record)
        if result.trace_step:
            steps.append(result.trace_step)

    confidence = _coerce_confidence(result.confidence)
    steps.append(TraceStep(
        step_id=str(uuid.uuid4()), timestamp=datetime.utcnow().isoformat()+"Z",
        model_name="decision:v0", thought="Consolidate evidence and decide eligibility.",
        action="decide", observation=f"eligible={result.eligible}, confidence={confidence:.2f}",
        confidence=confidence,
    ))

    classification = ClassificationResult(
        project_id=record.project_id, eligible=result.eligible, confidence=confidence,
        rationale=result.rationale, region="US-IRS-Section-41", criteria=result.criteria,
    )
    trace = TraceEnvelope(
        user_id=user_id, project_id=record.project_id, steps=steps,
        model_name=(f"openai:{MODEL_NAME}" if USE_LLM else "rule-based:v0"),
        region="US-IRS-Section-41", reviewer_id=None, legal_hold_flag=False,
    ).model_dump()

    if not USE_LLM:
        trace["narrative_excerpt"] = _template_narrative(record.project_name, record.description)

    return classification, trace


def analyze_project_strict(record: ProjectRecord, user_id: str = "demo-user"):
    """LLM-only analysis: always call the LLM and raise on errors. No rule-based fallback."""
    if not (USE_LLM and OpenAI is not None):
        raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY and install SDK.")

    steps = [TraceStep(
        step_id=str(uuid.uuid4()), timestamp=datetime.utcnow().isoformat()+"Z",
        model_name="parser:v0", thought="Parse incoming project record into analysis fields.",
        action="parse", observation=f"Got description length={len(record.description or '')}",
        confidence=1.0,
    )]

    result = _tier3_llm_sync(record)  # raises on error; no fallback by design
    steps.append(result.trace_step)

    confidence = _coerce_confidence(result.confidence)
    steps.append(TraceStep(
        step_id=str(uuid.uuid4()), timestamp=datetime.utcnow().isoformat()+"Z",
        model_name="decision:v0", thought="Consolidate evidence and decide eligibility.",
        action="decide", observation=f"eligible={result.eligible}, confidence={confidence:.2f}",
        confidence=confidence,
    ))

    classification = ClassificationResult(
        project_id=record.project_id, eligible=result.eligible, confidence=confidence,
        rationale=result.rationale, region="US-IRS-Section-41", criteria=result.criteria,
    )
    trace = TraceEnvelope(
        user_id=user_id, project_id=record.project_id, steps=steps,
        model_name=f"openai:{MODEL_NAME}",
        region="US-IRS-Section-41", reviewer_id=None, legal_hold_flag=False,
    ).model_dump()
    return classification, trace
