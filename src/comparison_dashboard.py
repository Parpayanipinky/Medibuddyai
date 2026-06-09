"""
User-side health progress comparison dashboard for MediBuddy AI.

This module is intentionally self-contained so it can be added to the Gradio
app without disturbing the existing user/admin UI structure.

Updated features:
- Technical lab report comparison explanation instead of only simple wording.
- PDF export helpers for manual comparison and saved-vs-current comparison.
- More technical X-ray comparison summary structure for radiology discussion.
"""

from __future__ import annotations

import base64
import html
import json
import os
import re
import tempfile
import uuid

try:
    from groq import Groq
except Exception:
    try:
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "groq", "-q", "--break-system-packages"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        from groq import Groq
    except Exception:
        Groq = None
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import easyocr
except Exception:  # pragma: no cover - app can still compare PDFs/saved data
    easyocr = None

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except Exception:  # pragma: no cover - app still shows HTML if reportlab missing
    colors = None
    TA_CENTER = TA_LEFT = None
    A4 = landscape = None
    ParagraphStyle = getSampleStyleSheet = None
    inch = None
    Paragraph = SimpleDocTemplate = Spacer = Table = TableStyle = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
REPORT_DIR = DATA_DIR / "reports"
COMPARISON_DIR = DATA_DIR / "comparison"
SNAPSHOT_DIR = DATA_DIR / "saved_lab_reports"
XRAY_HISTORY_DIR = DATA_DIR / "saved_xrays"

for folder in [UPLOAD_DIR, REPORT_DIR, COMPARISON_DIR, SNAPSHOT_DIR, XRAY_HISTORY_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# Optional Groq vision client for actual image-specific X-ray comparison.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
_groq_vision_client = None
if Groq is not None and GROQ_API_KEY:
    try:
        _groq_vision_client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        _groq_vision_client = None

MEDICAL_DISCLAIMER = (
    "This comparison is for educational support only. It does not replace a doctor's diagnosis. "
    "Please consult a qualified healthcare professional for final interpretation."
)

XRAY_DISCLAIMER = (
    "This technical X-ray comparison is for educational support and report-organization only. "
    "It is not a final radiology diagnosis. Please consult a certified radiologist or treating clinician."
)

MOTIVATION_QUOTES = [
    "Small improvements in your reports can reflect meaningful health progress.",
    "Progress is easier to understand when you compare your results step by step.",
    "Your health journey is a timeline, not a single report.",
    "Consistent monitoring helps you ask better questions to your doctor.",
]

REFERENCE_RANGES: Dict[str, Tuple[float, float]] = {
    "hemoglobin": (12.0, 17.5),
    "hb": (12.0, 17.5),
    "wbc": (4000, 11000),
    "white blood cells": (4000, 11000),
    "rbc": (4.2, 5.9),
    "platelets": (150000, 450000),
    "platelet": (150000, 450000),
    "glucose": (70, 140),
    "fasting glucose": (70, 100),
    "hba1c": (4.0, 5.6),
    "cholesterol": (0, 200),
    "total cholesterol": (0, 200),
    "ldl": (0, 100),
    "hdl": (40, 100),
    "triglycerides": (0, 150),
    "creatinine": (0.6, 1.3),
    "urea": (7, 20),
    "uric acid": (3.5, 7.2),
    "tsh": (0.4, 4.5),
    "t3": (0.8, 2.0),
    "t4": (4.5, 12.0),
    "alt": (0, 45),
    "sgpt": (0, 45),
    "ast": (0, 40),
    "sgot": (0, 40),
    "bilirubin": (0.1, 1.2),
    "vitamin d": (30, 100),
    "mcv": (80, 100),
    "mch": (27, 33),
    "mchc": (32, 36),
    "neutrophils": (40, 75),
    "lymphocytes": (20, 45),
    "eosinophils": (1, 6),
    "monocytes": (2, 10),
    "basophils": (0, 2),
}

ALIASES = {
    "hb": "hemoglobin",
    "hgb": "hemoglobin",
    "haemoglobin": "hemoglobin",
    "white blood cell": "wbc",
    "white blood cells": "wbc",
    "total leukocyte count": "wbc",
    "tlc": "wbc",
    "platelet count": "platelets",
    "platelets count": "platelets",
    "total cholesterol": "cholesterol",
    "ldl cholesterol": "ldl",
    "hdl cholesterol": "hdl",
    "sgpt": "alt",
    "sgot": "ast",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _file_path(uploaded_file: Any) -> str:
    if not uploaded_file:
        return ""
    return uploaded_file if isinstance(uploaded_file, str) else getattr(uploaded_file, "name", "")


def _normalize_name(name: str) -> str:
    value = re.sub(r"[^a-z0-9\s]", " ", str(name).lower())
    return re.sub(r"\s+", " ", value).strip()


def _canonical_parameter(name: str) -> str:
    normalized = _normalize_name(name)
    for alias, canonical in sorted(ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return canonical
    for term in sorted(REFERENCE_RANGES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(term)}\b", normalized):
            return ALIASES.get(term, term)
    return normalized


def _status(value: float, low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "Unknown"
    if value < low:
        return "Low"
    if value > high:
        return "High"
    return "Normal"


def _direction(diff: float) -> str:
    if diff > 0:
        return "Increased"
    if diff < 0:
        return "Decreased"
    return "Unchanged"


def _percentage_change(previous: float, current: float) -> str:
    if previous == 0:
        return "N/A"
    return f"{((current - previous) / previous) * 100:.1f}%"


def _clinical_interpretation(item: Dict[str, Any]) -> str:
    prev_status = item.get("previous_status")
    curr_status = item.get("current_status")
    progress = item.get("progress")
    direction = item.get("direction")
    parameter = item.get("parameter")

    if prev_status == "Normal" and curr_status != "Normal":
        return f"{parameter} has newly moved outside the reference interval; clinical correlation is recommended."
    if prev_status != "Normal" and curr_status == "Normal":
        return f"{parameter} has normalized compared with the previous report."
    if curr_status != "Normal" and progress == "Worsened":
        return f"{parameter} remains abnormal and has moved further away from the reference interval."
    if curr_status != "Normal" and progress == "Improved":
        return f"{parameter} remains outside range but is closer to the reference interval than before."
    if curr_status == "Normal" and prev_status == "Normal" and direction != "Unchanged":
        return f"{parameter} changed numerically but remains within the reference interval."
    return f"{parameter} is stable based on the detected reference interval."


def _reference_distance(value: float, low: float | None, high: float | None) -> float:
    if low is None or high is None:
        return 0.0
    if low <= value <= high:
        return 0.0
    return min(abs(value - low), abs(value - high))




COMPARISON_NON_MEDICAL_ERROR = (
    "This file does not look like a medical report. Please upload a valid lab report for comparison."
)
COMPARISON_NON_XRAY_ERROR = "Invalid image uploaded. Please upload a valid X-ray image."
XRAY_BODY_MISMATCH_ERROR = (
    "The uploaded X-rays appear to be from different body areas. "
    "Please upload matching X-rays for comparison."
)

def _comparison_error(message: str) -> str:
    return f"""
    <div class='compare-alert' style='padding:22px;border-radius:18px;background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;font-family:Inter,Arial,sans-serif;'>
      <h3 style='margin:0 0 8px;'>Upload Error</h3>
      <p style='margin:0;'>{_esc(message)}</p>
    </div>
    """

def _medical_text_confidence_score(text: str) -> int:
    text = str(text or '').lower()
    if not text.strip():
        return 0
    medical_terms = [
        'patient', 'doctor', 'physician', 'hospital', 'clinic', 'laboratory', 'diagnostic',
        'report', 'specimen', 'sample', 'collection', 'reference range', 'normal range',
        'hemoglobin', 'haemoglobin', 'wbc', 'rbc', 'platelet', 'cbc', 'glucose', 'hba1c',
        'cholesterol', 'triglycerides', 'hdl', 'ldl', 'creatinine', 'urea', 'bun',
        'bilirubin', 'alt', 'ast', 'sgpt', 'sgot', 'tsh', 't3', 't4', 'urine',
        'x-ray', 'xray', 'radiograph', 'radiology', 'findings', 'impression', 'ct', 'mri',
        'ultrasound', 'prescription', 'medicine', 'diagnosis', 'clinical', 'blood', 'serum'
    ]
    score = sum(1 for term in medical_terms if term in text)
    score += min(5, len(re.findall(r'\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?', text)))
    score += min(5, len(re.findall(r'\d+(?:\.\d+)?\s*(?:mg/dl|g/dl|mmol/l|u/l|iu/l|%|fl|pg|ng/ml|x10)', text, flags=re.I)))
    return score

def _is_likely_medical_report_file(uploaded_file: Any) -> tuple[bool, str]:
    path = _file_path(uploaded_file)
    if not path or not os.path.exists(path):
        return False, 'Please upload a valid file.'
    ext = os.path.splitext(path)[1].lower()
    if ext not in ['.pdf', '.png', '.jpg', '.jpeg', '.webp']:
        return False, 'Unsupported file format. Please upload PDF, PNG, JPG, JPEG, or WEBP medical reports.'
    text = extract_text_from_file(uploaded_file)
    values = parse_lab_values_from_text(text)
    if values or _medical_text_confidence_score(text) >= 3:
        return True, 'OK'
    return False, COMPARISON_NON_MEDICAL_ERROR

def _readable_image_file(uploaded_file: Any) -> tuple[bool, str]:
    """Basic upload checks only. Do not decide body part here."""
    path = _file_path(uploaded_file)
    if not path or not os.path.exists(path):
        return False, COMPARISON_NON_XRAY_ERROR

    ext = os.path.splitext(path)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
        return False, COMPARISON_NON_XRAY_ERROR

    if Image is None:
        return True, "OK"

    try:
        with Image.open(path) as img:
            w, h = img.size
            if w < 80 or h < 80:
                return False, COMPARISON_NON_XRAY_ERROR
            img.verify()
        return True, "OK"
    except Exception:
        return False, COMPARISON_NON_XRAY_ERROR


def _local_xray_image_score(uploaded_file: Any) -> float:
    """
    Forgiving local X-ray likelihood score used only when the vision model is
    unavailable or the validation call fails. It accepts grayscale radiographs
    from different body regions without hard-coding one anatomy type.
    """
    path = _file_path(uploaded_file)
    if not path or Image is None:
        return 0.0
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w < 80 or h < 80:
            return 0.0
        small = img.resize((96, 96))
        pixels = list(small.getdata())
        n = max(len(pixels), 1)

        luma = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pixels]
        mean_luma = sum(luma) / n
        variance = sum((x - mean_luma) ** 2 for x in luma) / n
        contrast = variance ** 0.5

        gray_like = sum(1 for r, g, b in pixels if max(r, g, b) - min(r, g, b) <= 45) / n
        dark_ratio = sum(1 for x in luma if x < 55) / n
        bright_ratio = sum(1 for x in luma if x > 215) / n
        mid_ratio = sum(1 for x in luma if 55 <= x <= 215) / n
        color_spread = sum((abs(r - g) + abs(g - b) + abs(r - b)) / 3 for r, g, b in pixels) / n

        score = 0.0
        if gray_like >= 0.55:
            score += 0.35
        if gray_like >= 0.72:
            score += 0.15
        if contrast >= 16:
            score += 0.25
        if dark_ratio >= 0.10:
            score += 0.15
        if mid_ratio >= 0.12:
            score += 0.10
        if bright_ratio > 0.82 and dark_ratio < 0.08:
            score -= 0.35  # document/screenshot-like white page
        if color_spread > 50 and gray_like < 0.65:
            score -= 0.35  # normal camera photo/selfie/coffee-like image
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def _is_likely_xray_image_file(uploaded_file: Any) -> bool:
    ok, _ = _readable_image_file(uploaded_file)
    if not ok:
        return False
    # Keep this helper permissive so real X-rays are not blocked before the
    # vision model can review them. Non-X-rays are rejected by pair validation.
    if _groq_vision_client is not None:
        return True
    return _local_xray_image_score(uploaded_file) >= 0.45


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _normalise_xray_body_area(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    if not text or text in {"unknown", "unclear", "not clear", "not sure", "other"}:
        return "unknown"

    if any(term in text for term in ["chest", "lung", "lungs", "rib", "ribs", "thorax"]):
        return "chest"
    if any(term in text for term in ["skull", "head", "cranial", "face", "facial"]):
        return "skull_head"
    if "cervical" in text:
        return "cervical_spine"
    if "thoracic" in text:
        return "thoracic_spine"
    if "lumbar" in text or "lumbosacral" in text:
        return "lumbar_spine"
    if "spine" in text or "vertebra" in text:
        return "spine"
    if "shoulder" in text:
        return "shoulder"
    if "elbow" in text:
        return "elbow"
    if "wrist" in text or "hand" in text or "finger" in text:
        return "wrist_hand"
    if "hip" in text or "pelvis" in text:
        return "pelvis_hip"
    if "knee" in text or "patella" in text:
        return "knee"
    if "ankle" in text or "foot" in text or "toe" in text:
        return "ankle_foot"
    if any(term in text for term in ["humerus", "radius", "ulna", "forearm", "arm"]):
        return "arm"
    if any(term in text for term in ["femur", "tibia", "fibula", "leg"]):
        return "leg"
    if "joint" in text:
        return "joint"
    return text.replace(" ", "_")[:40] or "unknown"


def _xray_body_areas_conflict(previous_area: Any, current_area: Any) -> bool:
    prev = _normalise_xray_body_area(previous_area)
    curr = _normalise_xray_body_area(current_area)
    if prev == "unknown" or curr == "unknown":
        return False
    return prev != curr


def _validate_xray_pair_with_vision(previous_xray: Any, current_xray: Any) -> Dict[str, Any] | None:
    """Validate X-ray images using Groq vision. Returns JSON dict or None."""
    if _groq_vision_client is None:
        # No vision client – skip strict validation, let local scorer decide
        return None

    previous_data_url = _image_file_to_data_url(previous_xray)
    current_data_url = _image_file_to_data_url(current_xray)
    if not previous_data_url or not current_data_url:
        return None

    prompt = (
        "You are validating uploads for an educational X-ray comparison dashboard.\n"
        "Classify BOTH images. Accept any genuine medical X-ray/radiograph (knee, chest, skull, "
        "spine, arm, leg, joint, hand, foot, shoulder, pelvis/hip, etc).\n"
        "Reject selfies, food photos, camera photos, screenshots, lab reports, cartoons, "
        "or anything that is NOT a radiographic X-ray.\n\n"
        "Return ONLY valid JSON in this exact shape (no markdown, no explanation):\n"
        '{\n  "previous": {"is_xray": true, "body_area": "chest", "reason": "short reason"},\n'
        '  "current":  {"is_xray": true, "body_area": "chest", "reason": "short reason"}\n}\n\n'
        "Use a specific body_area when visible. Use \"unknown\" only if the body area truly cannot be identified."
    )

    try:
        response = _groq_vision_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": "PREVIOUS uploaded image:"},
                        {"type": "image_url", "image_url": {"url": previous_data_url}},
                        {"type": "text", "text": "CURRENT uploaded image:"},
                        {"type": "image_url", "image_url": {"url": current_data_url}},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=450,
        )
        raw = str(response.choices[0].message.content or "").strip()
        return _extract_json_object(raw)
    except Exception:
        return None




def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value or "").strip().lower()
    return text in {"true", "yes", "y", "1", "valid", "xray", "x-ray", "radiograph"}

def _validate_xray_pair_for_comparison(previous_xray: Any, current_xray: Any) -> tuple[bool, str]:
    prev_ok, prev_msg = _readable_image_file(previous_xray)
    curr_ok, curr_msg = _readable_image_file(current_xray)
    if not prev_ok or not curr_ok:
        return False, COMPARISON_NON_XRAY_ERROR

    vision_result = _validate_xray_pair_with_vision(previous_xray, current_xray)
    if vision_result:
        previous_info = vision_result.get("previous") or {}
        current_info = vision_result.get("current") or {}
        previous_is_xray = _coerce_bool(previous_info.get("is_xray"))
        current_is_xray = _coerce_bool(current_info.get("is_xray"))
        if not previous_is_xray or not current_is_xray:
            return False, COMPARISON_NON_XRAY_ERROR
        if _xray_body_areas_conflict(previous_info.get("body_area"), current_info.get("body_area")):
            return False, XRAY_BODY_MISMATCH_ERROR
        return True, "OK"

    # Fallback when VLM validation is unavailable. This remains intentionally
    # tolerant to avoid blocking valid knee/chest/skull/spine/limb X-rays.
    if _local_xray_image_score(previous_xray) < 0.45 or _local_xray_image_score(current_xray) < 0.45:
        return False, COMPARISON_NON_XRAY_ERROR
    return True, "OK"


# ---------------------------------------------------------------------------
# Extraction and parsing
# ---------------------------------------------------------------------------

def extract_text_from_file(uploaded_file: Any) -> str:
    path = _file_path(uploaded_file)
    if not path or not os.path.exists(path):
        return ""

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            parts: List[str] = []
            with fitz.open(path) as doc:
                for page in doc:
                    parts.append(page.get_text("text"))
            text = "\n".join(parts).strip()
            if text:
                return text

        if ext in [".png", ".jpg", ".jpeg", ".webp"] and easyocr is not None:
            reader = easyocr.Reader(["en"], gpu=False)
            return "\n".join(reader.readtext(path, detail=0)).strip()
    except Exception:
        return ""

    return ""


def _parse_reference_range_from_text(text: str, fallback: Tuple[float | None, float | None]) -> Tuple[float | None, float | None]:
    ref_patterns = [
        r"(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)",
    ]
    for pattern in ref_patterns:
        match = re.search(pattern, str(text), flags=re.I)
        if match:
            try:
                return float(match.group(1)), float(match.group(2))
            except Exception:
                pass
    return fallback


def parse_lab_values_from_records(records: Any) -> Dict[str, Dict[str, Any]]:
    values: Dict[str, Dict[str, Any]] = {}
    if not isinstance(records, list):
        return values

    for row in records:
        if not isinstance(row, dict):
            continue

        name = row.get("Parameter") or row.get("parameter") or row.get("name") or row.get("Test")
        raw_value = row.get("Value") or row.get("value") or row.get("Result")
        if name is None or raw_value is None:
            continue

        match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", str(raw_value))
        if not match:
            continue

        canonical = _canonical_parameter(str(name))
        try:
            value = float(match.group(0))
        except Exception:
            continue

        low, high = REFERENCE_RANGES.get(canonical, (None, None))
        ref = row.get("Reference Range") or row.get("reference_range") or row.get("Normal Range") or ""
        low, high = _parse_reference_range_from_text(ref, (low, high))

        unit = row.get("Unit") or row.get("unit") or ""
        values[canonical] = {
            "name": str(name).strip().title(),
            "value": value,
            "unit": str(unit).strip(),
            "low": low,
            "high": high,
            "source": "records",
        }

    return values


def parse_lab_values_from_text(text: str) -> Dict[str, Dict[str, Any]]:
    text = str(text or "")
    values: Dict[str, Dict[str, Any]] = {}
    terms = sorted(set(list(REFERENCE_RANGES) + list(ALIASES)), key=len, reverse=True)

    for line in text.splitlines():
        clean = re.sub(r"\s+", " ", line.strip())
        if not clean:
            continue

        lower = clean.lower()
        for term in terms:
            if not re.search(rf"\b{re.escape(term)}\b", lower):
                continue

            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", clean)
            if not nums:
                continue

            try:
                value = float(nums[0])
            except Exception:
                continue

            canonical = _canonical_parameter(term)
            low, high = REFERENCE_RANGES.get(canonical, REFERENCE_RANGES.get(term, (None, None)))
            low, high = _parse_reference_range_from_text(clean, (low, high))

            # crude unit detection immediately after first numeric value
            unit = ""
            after_value = clean[clean.find(nums[0]) + len(nums[0]):].strip()
            unit_match = re.match(r"([A-Za-zµμ%/^0-9.]+)", after_value)
            if unit_match:
                unit = unit_match.group(1)

            values[canonical] = {
                "name": canonical.title(),
                "value": value,
                "unit": unit,
                "low": low,
                "high": high,
                "source_line": clean,
            }
            break

    return values


def _values_from_payload(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    values = parse_lab_values_from_records(payload.get("lab_records"))
    if not values:
        values = parse_lab_values_from_text(payload.get("formatted_text") or payload.get("raw_text") or "")
    return values


def _values_from_file(uploaded_file: Any) -> Dict[str, Dict[str, Any]]:
    text = extract_text_from_file(uploaded_file)
    return parse_lab_values_from_text(text)


# ---------------------------------------------------------------------------
# Saved lab reports
# ---------------------------------------------------------------------------

def save_lab_report_snapshot(payload: Dict[str, Any]) -> bool:
    """Save extracted lab values so users can compare later without needing old files."""
    try:
        payload = dict(payload or {})
        values = _values_from_payload(payload)
        if not values:
            return False

        file_name = payload.get("original_filename") or "Lab report"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snapshot = {
            "id": uuid.uuid4().hex,
            "created_at": stamp,
            "file_name": file_name,
            "report_subtype": payload.get("report_subtype", "Laboratory Report"),
            "risk_level": (payload.get("risk_score") or {}).get("level", "Unknown")
            if isinstance(payload.get("risk_score"), dict)
            else "Unknown",
            "values": values,
        }
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(file_name))[:55]
        out_path = SNAPSHOT_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}_{snapshot['id'][:8]}.json"
        out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def get_saved_report_choices() -> List[Tuple[str, str]]:
    choices: List[Tuple[str, str]] = []
    for path in sorted(SNAPSHOT_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            label = f"{data.get('created_at', '')} · {data.get('file_name', 'Saved report')} · {data.get('report_subtype', 'Lab')}"
            choices.append((label, str(path)))
        except Exception:
            continue
    return choices[:30]


def refresh_saved_report_choices():
    try:
        import gradio as gr

        choices = get_saved_report_choices()
        return gr.update(choices=choices, value=choices[0][1] if choices else None)
    except Exception:
        return None


def _load_saved_values(saved_path: str) -> Dict[str, Dict[str, Any]]:
    if not saved_path or not os.path.exists(saved_path):
        return {}
    try:
        data = json.loads(Path(saved_path).read_text(encoding="utf-8"))
        values = data.get("values") or {}
        return values if isinstance(values, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Lab comparison
# ---------------------------------------------------------------------------

def _compare_value(name: str, previous: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    prev = float(previous.get("value", 0))
    curr = float(current.get("value", 0))
    low = current.get("low") if current.get("low") is not None else previous.get("low")
    high = current.get("high") if current.get("high") is not None else previous.get("high")
    unit = current.get("unit") or previous.get("unit") or ""

    prev_status = _status(prev, low, high)
    curr_status = _status(curr, low, high)
    diff = curr - prev

    if prev_status != "Normal" and curr_status == "Normal":
        progress = "Improved"
    elif prev_status == "Normal" and curr_status != "Normal":
        progress = "Needs Attention"
    elif curr_status == "Normal" and prev_status == "Normal":
        progress = "Stable"
    else:
        prev_distance = _reference_distance(prev, low, high)
        curr_distance = _reference_distance(curr, low, high)
        if curr_distance < prev_distance:
            progress = "Improved"
        elif curr_distance > prev_distance:
            progress = "Worsened"
        else:
            progress = "Stable"

    item = {
        "parameter": (current.get("name") or previous.get("name") or name).title(),
        "previous": round(prev, 2),
        "current": round(curr, 2),
        "unit": unit,
        "difference": round(diff, 2),
        "percentage_change": _percentage_change(prev, curr),
        "direction": _direction(diff),
        "normal_range": f"{low} - {high}" if low is not None and high is not None else "Not available",
        "previous_status": prev_status,
        "current_status": curr_status,
        "progress": progress,
    }
    item["clinical_interpretation"] = _clinical_interpretation(item)
    return item


def compare_value_sets(previous_values: Dict[str, Any], current_values: Dict[str, Any]) -> Dict[str, Any]:
    common = sorted(set(previous_values) & set(current_values))
    comparisons = [_compare_value(param, previous_values[param], current_values[param]) for param in common]

    improved = sum(1 for x in comparisons if x["progress"] == "Improved")
    worsened = sum(1 for x in comparisons if x["progress"] in ["Worsened", "Needs Attention"])
    stable = sum(1 for x in comparisons if x["progress"] == "Stable")
    current_abnormal = sum(1 for x in comparisons if x["current_status"] != "Normal")
    previous_abnormal = sum(1 for x in comparisons if x["previous_status"] != "Normal")

    if current_abnormal == 0:
        risk = "Low Risk"
    elif current_abnormal <= 2:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    return {
        "total_compared": len(comparisons),
        "improved": improved,
        "worsened": worsened,
        "stable": stable,
        "previous_abnormal": previous_abnormal,
        "current_abnormal": current_abnormal,
        "risk": risk,
        "comparisons": comparisons,
    }


def _chart_data_uri(comparisons: List[Dict[str, Any]]) -> str:
    if not comparisons:
        return ""

    top = comparisons[:8]
    labels = [x["parameter"] for x in top]
    previous = [x["previous"] for x in top]
    current = [x["current"] for x in top]
    x = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - width / 2 for i in x], previous, width, label="Previous")
    ax.bar([i + width / 2 for i in x], current, width, label="Current")
    ax.set_title("Previous vs Current Lab Values")
    ax.set_ylabel("Detected value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.legend()
    fig.tight_layout()

    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    plt.savefig(temp_path, dpi=165)
    plt.close(fig)

    encoded = base64.b64encode(Path(temp_path).read_bytes()).decode("utf-8")
    try:
        os.remove(temp_path)
    except Exception:
        pass

    return f"data:image/png;base64,{encoded}"


def _technical_lab_points(comparisons: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    abnormal_or_worse = []
    improved = []
    stable = []
    doctor_points = []

    for item in comparisons:
        line = (
            f"{item['parameter']}: {item['previous']} → {item['current']} {item.get('unit', '')} "
            f"({item['direction']}, Î” {item['difference']}, {item['percentage_change']}); "
            f"status {item['previous_status']} → {item['current_status']}. "
            f"{item['clinical_interpretation']}"
        )

        if item["progress"] in ["Worsened", "Needs Attention"] or item["current_status"] != "Normal":
            abnormal_or_worse.append(line)
            doctor_points.append(
                f"Discuss {item['parameter']} because current status is {item['current_status']} and trend is {item['direction'].lower()}."
            )
        elif item["progress"] == "Improved":
            improved.append(line)
        else:
            stable.append(line)

    if not abnormal_or_worse:
        abnormal_or_worse = ["No newly abnormal or worsened matched parameter was detected from the parsed values."]
    if not improved:
        improved = ["No clearly improved matched parameter was detected from the parsed values."]
    if not stable:
        stable = ["No clearly stable matched parameter was detected from the parsed values."]
    if not doctor_points:
        doctor_points = ["Review the full report with a clinician, especially if symptoms are present despite stable numeric values."]

    return {
        "abnormal_or_worse": abnormal_or_worse,
        "improved": improved,
        "stable": stable,
        "doctor_points": doctor_points,
    }


def _comparison_html(result: Dict[str, Any], title: str) -> str:
    comparisons = result.get("comparisons", [])
    if not comparisons:
        return """
        <style>
          .compare-wrap {font-family: Inter, Arial, sans-serif; color:#173b31; max-width:1120px; margin:0 auto;}
          .compare-alert {padding:24px; border-radius:22px; background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; box-shadow:0 14px 34px rgba(154,52,18,.07);}
        </style>
        <div class="compare-wrap">
          <div class="compare-alert">
            <h3 style="margin:0 0 8px;">No matching lab values found</h3>
            <p style="margin:0 0 6px;">Please upload clearer lab reports or select a saved lab report with extracted values.</p>
            <p style="margin:0;">For comparison, each report should contain the same test names with numeric values.</p>
          </div>
        </div>
        """

    chart_src = _chart_data_uri(comparisons)
    sections = _technical_lab_points(comparisons)
    rows = []
    change_cards = []

    priority = {"Needs Attention": 0, "Worsened": 1, "Improved": 2, "Stable": 3}
    highlighted = sorted(comparisons, key=lambda item: priority.get(item.get("progress", "Stable"), 4))[:12]

    for item in highlighted:
        progress = item.get("progress", "")
        badge_class = "good" if progress == "Improved" else ("bad" if progress in ["Worsened", "Needs Attention"] else "stable")
        status_class = "bad" if item.get("current_status") in ["High", "Low"] else ("stable" if item.get("current_status") == "Unknown" else "good")
        change_cards.append(
            f"""
            <div class="change-card">
              <div class="change-topline">
                <h4>{_esc(item.get('parameter', 'Parameter'))}</h4>
                <span class="badge {badge_class}">{_esc(progress)}</span>
              </div>
              <div class="value-grid">
                <div><span>Previous</span><b>{_esc(item.get('previous', ''))} {_esc(item.get('unit', ''))}</b></div>
                <div><span>Current</span><b>{_esc(item.get('current', ''))} {_esc(item.get('unit', ''))}</b></div>
                <div><span>% Change</span><b>{_esc(item.get('percentage_change', ''))}</b></div>
                <div><span>Status</span><b><span class="badge {status_class}">{_esc(item.get('current_status', ''))}</span></b></div>
              </div>
            </div>
            """
        )

    for item in comparisons:
        progress = item.get("progress", "")
        badge_class = "good" if progress == "Improved" else ("bad" if progress in ["Worsened", "Needs Attention"] else "stable")
        prev_status_class = "bad" if item.get("previous_status") in ["High", "Low"] else ("stable" if item.get("previous_status") == "Unknown" else "good")
        curr_status_class = "bad" if item.get("current_status") in ["High", "Low"] else ("stable" if item.get("current_status") == "Unknown" else "good")
        rows.append(
            f"""
            <tr>
              <td class="sticky-param"><b>{_esc(item.get('parameter', ''))}</b></td>
              <td class="value-cell"><b>{_esc(item.get('previous', ''))}</b><small>{_esc(item.get('unit', ''))}</small></td>
              <td class="value-cell current"><b>{_esc(item.get('current', ''))}</b><small>{_esc(item.get('unit', ''))}</small></td>
              <td class="number-cell">{_esc(item.get('difference', ''))}</td>
              <td class="number-cell"><b>{_esc(item.get('percentage_change', ''))}</b></td>
              <td>{_esc(item.get('normal_range', ''))}</td>
              <td><span class="badge {prev_status_class}">{_esc(item.get('previous_status', ''))}</span></td>
              <td><span class="badge {curr_status_class}">{_esc(item.get('current_status', ''))}</span></td>
              <td><span class="badge {badge_class}">{_esc(progress)}</span></td>
              <td class="interpretation-cell">{_esc(item.get('clinical_interpretation', ''))}</td>
            </tr>
            """
        )

    quote = MOTIVATION_QUOTES[result.get("total_compared", 0) % len(MOTIVATION_QUOTES)]

    def list_items(items: List[str]) -> str:
        return "".join(f"<li><span>{_esc(p)}</span></li>" for p in items[:10])

    def section_card(title_text: str, icon: str, items: List[str], card_class: str) -> str:
        return f"""
        <section class="summary-section {card_class}">
          <div class="section-title"><span class="section-icon">{icon}</span><h4>{_esc(title_text)}</h4></div>
          <ul class="technical-list">{list_items(items)}</ul>
        </section>
        """

    abnormal_card = section_card("Abnormal / Worsened Parameters", "⚠️", sections["abnormal_or_worse"], "attention")
    improved_card = section_card("Improved / Normalized Parameters", "✅", sections["improved"], "improved")
    stable_card = section_card("Stable Parameters", "➖", sections["stable"], "stable")
    doctor_card = section_card("Suggested Doctor Discussion Points", "🩺", sections["doctor_points"], "doctor")

    return f"""
    <style>
      .compare-wrap {{
        font-family: Inter, Arial, sans-serif;
        color:#173b31;
        max-width: 1180px;
        margin: 0 auto;
      }}
      .compare-hero {{
        background: radial-gradient(circle at top left, rgba(187,247,208,.7), transparent 34%), linear-gradient(135deg,#ecfdf5,#ffffff 70%);
        border: 1px solid #c9f2df;
        border-radius: 28px;
        padding: 26px;
        margin: 16px 0;
        box-shadow: 0 18px 48px rgba(8,127,91,.09);
      }}
      .compare-hero h1 {{
        font-size: clamp(28px, 4vw, 42px);
        line-height: 1.05;
        color:#173b31;
        letter-spacing:-.03em;
      }}
      .compare-hero p {{max-width:820px; line-height:1.65;}}
      .compare-grid {{
        display:grid;
        grid-template-columns: repeat(4,minmax(0,1fr));
        gap:14px;
        margin:16px 0;
      }}
      .compare-card {{
        background: rgba(255,255,255,.94);
        border: 1px solid #d8f3e6;
        border-radius: 22px;
        padding: 18px;
        box-shadow: 0 14px 34px rgba(19,111,82,.07);
      }}
      .compare-card h3 {{margin:0 0 12px; color:#12372f; font-size:19px; letter-spacing:-.01em;}}
      .compare-number {{
        font-size: 32px;
        font-weight: 950;
        color:#087f5b;
        line-height:1.1;
      }}
      .compare-label {{
        font-size: 11px;
        color:#6b8179;
        font-weight:900;
        text-transform:uppercase;
        letter-spacing:.07em;
      }}
      .compare-chart {{
        width:100%;
        border-radius:18px;
        border:1px solid #d8f3e6;
        margin-top:8px;
        background:#fff;
      }}
      .summary-grid {{
        display:grid;
        grid-template-columns: repeat(2, minmax(0,1fr));
        gap:16px;
        margin-top:12px;
      }}
      .summary-section {{
        border:1px solid #e2e8f0;
        border-radius:20px;
        padding:16px;
        background:#ffffff;
        box-shadow:0 10px 24px rgba(15,23,42,.04);
      }}
      .summary-section.attention {{background:#fffafa; border-color:#fecaca;}}
      .summary-section.improved {{background:#f7fff9; border-color:#bbf7d0;}}
      .summary-section.stable {{background:#fffdf2; border-color:#fde68a;}}
      .summary-section.doctor {{background:#f8fbff; border-color:#bfdbfe;}}
      .section-title {{display:flex; align-items:center; gap:10px; margin-bottom:10px;}}
      .section-title h4 {{margin:0; color:#12372f; font-size:15px;}}
      .section-icon {{
        width:34px; height:34px; border-radius:12px; display:inline-flex; align-items:center; justify-content:center;
        background:#ffffff; border:1px solid rgba(15,23,42,.08); box-shadow:0 6px 14px rgba(15,23,42,.05);
      }}
      .technical-list {{
        line-height:1.65;
        color:#37584d;
        margin:0;
        padding:0;
        list-style:none;
        display:flex;
        flex-direction:column;
        gap:8px;
      }}
      .technical-list li {{
        background:rgba(255,255,255,.72);
        border:1px solid rgba(15,23,42,.06);
        border-radius:14px;
        padding:10px 12px 10px 34px;
        position:relative;
      }}
      .technical-list li::before {{
        content:"";
        width:8px;
        height:8px;
        border-radius:999px;
        background:#087f5b;
        position:absolute;
        left:14px;
        top:18px;
      }}
      .change-card-grid {{
        display:grid;
        grid-template-columns: repeat(3, minmax(0,1fr));
        gap:14px;
        margin-top:12px;
      }}
      .change-card {{
        border:1px solid #dbeafe;
        border-radius:18px;
        padding:14px;
        background:linear-gradient(180deg,#ffffff,#f8fafc);
      }}
      .change-topline {{display:flex; align-items:flex-start; justify-content:space-between; gap:10px; margin-bottom:12px;}}
      .change-topline h4 {{margin:0; font-size:15px; color:#17203a; line-height:1.25;}}
      .value-grid {{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px;}}
      .value-grid div {{background:#ffffff; border:1px solid #e2e8f0; border-radius:14px; padding:9px; min-height:58px;}}
      .value-grid span {{display:block; font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:.06em; font-weight:900; margin-bottom:4px;}}
      .value-grid b {{font-size:14px; color:#12372f; word-break:break-word;}}
      .table-shell {{
        margin-top:16px;
        overflow:hidden;
        border:1px solid #d8f3e6;
        border-radius:20px;
        background:#ffffff;
      }}
      .table-scroll {{
        overflow-x:auto;
        -webkit-overflow-scrolling:touch;
        scrollbar-width:thin;
      }}
      .scroll-hint {{
        font-size:12px;
        color:#64748b;
        background:#f8fafc;
        border-bottom:1px solid #e2e8f0;
        padding:10px 14px;
        font-weight:750;
      }}
      .compare-table {{
        width:100%;
        border-collapse:separate;
        border-spacing:0;
        font-size:13px;
        min-width:1320px;
      }}
      .compare-table th {{
        background:#ecfdf5;
        color:#065f46;
        text-align:left;
        padding:14px 12px;
        border-bottom:1px solid #c9f2df;
        font-weight:950;
        white-space:nowrap;
        position:sticky;
        top:0;
        z-index:1;
      }}
      .compare-table td {{
        padding:13px 12px;
        border-bottom:1px solid #e7f7ef;
        vertical-align:top;
        color:#2d5147;
        background:#ffffff;
      }}
      .compare-table tr:nth-child(even) td {{background:#fbfefc;}}
      .sticky-param {{position:sticky; left:0; z-index:2; min-width:150px; box-shadow:8px 0 16px rgba(15,23,42,.035);}}
      .value-cell b {{display:block; color:#12372f; font-size:14px;}}
      .value-cell small {{display:block; color:#64748b; margin-top:3px;}}
      .value-cell.current {{background:#f8fffb !important;}}
      .number-cell {{font-variant-numeric:tabular-nums; white-space:nowrap;}}
      .interpretation-cell {{min-width:310px; line-height:1.55;}}
      .badge {{
        padding:6px 10px;
        border-radius:999px;
        font-weight:900;
        white-space:nowrap;
        display:inline-block;
        font-size:12px;
      }}
      .badge.good {{background:#dcfce7;color:#166534;}}
      .badge.bad {{background:#fee2e2;color:#991b1b;}}
      .badge.stable {{background:#fef9c3;color:#854d0e;}}
      .quote-box {{
        background:#f0fdf4;
        border:1px solid #bbf7d0;
        color:#166534;
        border-radius:18px;
        padding:16px;
        margin-top:14px;
        font-weight:900;
      }}
      .disclaimer-box {{
        margin-top:16px;
        padding:16px 18px;
        border-radius:18px;
        background:#f8fafc;
        border:1px solid #dbe3f0;
        color:#475569;
        line-height:1.55;
      }}
      @media(max-width:1000px) {{
        .compare-grid {{grid-template-columns:1fr 1fr;}}
        .summary-grid {{grid-template-columns:1fr;}}
        .change-card-grid {{grid-template-columns:1fr 1fr;}}
      }}
      @media(max-width:680px) {{
        .compare-grid, .change-card-grid {{grid-template-columns:1fr;}}
      }}
    </style>

    <div class="compare-wrap">
      <div class="compare-hero">
        <div style="font-weight:900;color:#087f5b;text-transform:uppercase;letter-spacing:.08em;">MediBuddy AI Technical Progress Tracker</div>
        <h1 style="margin:8px 0 8px;">{_esc(title)}</h1>
        <p style="margin:0;color:#5e756d;">A clear comparison of previous and current lab values, showing change direction, percentage change, reference-range status, and doctor discussion points.</p>
      </div>

      <div class="compare-grid">
        <div class="compare-card"><div class="compare-label">Compared Values</div><div class="compare-number">{result['total_compared']}</div></div>
        <div class="compare-card"><div class="compare-label">Improved</div><div class="compare-number">{result['improved']}</div></div>
        <div class="compare-card"><div class="compare-label">Needs Attention</div><div class="compare-number">{result['worsened']}</div></div>
        <div class="compare-card"><div class="compare-label">Current Risk</div><div class="compare-number" style="font-size:24px;">{_esc(result['risk'])}</div></div>
      </div>

      <div class="compare-card">
        <h3>Visual Comparison Graph</h3>
        <img class="compare-chart" src="{chart_src}" />
      </div>

      <div class="compare-card" style="margin-top:16px;">
        <h3>Comparison Summary</h3>
        <div class="summary-grid">
          {abnormal_card}
          {improved_card}
          {stable_card}
          {doctor_card}
        </div>
        <div class="quote-box">🌿 {quote}</div>
      </div>

      <div class="compare-card" style="margin-top:16px;">
        <h3>Key Value Changes</h3>
        <div class="change-card-grid">{''.join(change_cards)}</div>
      </div>

      <div class="compare-card" style="margin-top:16px;">
        <h3>Detailed Parameter-by-Parameter Comparison</h3>
        <div class="table-shell">
          <div class="scroll-hint">Swipe or scroll horizontally to view all comparison columns.</div>
          <div class="table-scroll">
            <table class="compare-table">
              <thead>
                <tr>
                  <th>Parameter</th>
                  <th>Previous Value</th>
                  <th>Current Value</th>
                  <th>Î” Change</th>
                  <th>% Change</th>
                  <th>Reference Range</th>
                  <th>Previous Status</th>
                  <th>Current Status</th>
                  <th>Progress</th>
                  <th>Technical Interpretation</th>
                </tr>
              </thead>
              <tbody>{''.join(rows)}</tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="disclaimer-box">
        <b>Medical Disclaimer:</b> {_esc(MEDICAL_DISCLAIMER)}
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

def _pdf_styles() -> Dict[str, Any]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "MediBuddyCompareTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17203A"),
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "MediBuddyCompareSubtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5F6B85"),
            spaceAfter=14,
        ),
        "heading": ParagraphStyle(
            "MediBuddyCompareHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=16,
            textColor=colors.HexColor("#312E81"),
            spaceBefore=10,
            spaceAfter=7,
        ),
        "box_heading": ParagraphStyle(
            "MediBuddyCompareBoxHeading",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#17203A"),
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "MediBuddyCompareBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=12.3,
            textColor=colors.HexColor("#17203A"),
            alignment=TA_LEFT,
        ),
        "small": ParagraphStyle(
            "MediBuddyCompareSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=9.4,
            textColor=colors.HexColor("#44506A"),
            alignment=TA_LEFT,
        ),
        "table_header": ParagraphStyle(
            "MediBuddyCompareTableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.0,
            leading=8.4,
            textColor=colors.HexColor("#312E81"),
            alignment=TA_LEFT,
        ),
        "table_cell": ParagraphStyle(
            "MediBuddyCompareTableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=6.6,
            leading=8.2,
            textColor=colors.HexColor("#17203A"),
            alignment=TA_LEFT,
        ),
    }


def _p(text: Any, style: Any) -> Any:
    return Paragraph(_esc(text).replace("\n", "<br/>"), style)


def export_lab_comparison_pdf(result: Dict[str, Any], title: str = "Technical Lab Report Comparison") -> str | None:
    """Create a polished downloadable PDF file for a lab comparison result."""
    comparisons = result.get("comparisons", [])
    if not comparisons or SimpleDocTemplate is None:
        return None

    out_path = COMPARISON_DIR / f"lab_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.pdf"
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=landscape(A4),
        leftMargin=0.42 * inch,
        rightMargin=0.42 * inch,
        topMargin=0.42 * inch,
        bottomMargin=0.42 * inch,
        title=title,
    )

    styles = _pdf_styles()
    sections = _technical_lab_points(comparisons)
    story = []

    def markup(text: Any, style: Any) -> Any:
        return Paragraph(str(text), style)

    def esc_text(value: Any) -> str:
        return html.escape(str(value if value is not None else ""))

    def bullet_text(items: List[str], limit: int = 10) -> str:
        selected = items[:limit] if items else ["No item was detected from the matched values."]
        return "<br/>".join(f"• {esc_text(item)}" for item in selected)

    def add_key_value_table(rows: List[Tuple[str, Any]], columns: int = 4) -> None:
        cells = []
        for label, value in rows:
            cell_html = (
                f"<b>{esc_text(label)}</b><br/>"
                f"<font color='#17203A' size='10'>{esc_text(value)}</font>"
            )
            cells.append(markup(cell_html, styles["small"]))
        while len(cells) % columns:
            cells.append(markup("", styles["small"]))
        data = [cells[i:i + columns] for i in range(0, len(cells), columns)]
        table = Table(data, colWidths=[doc.width / columns] * columns, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#DBE3F0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E6EDF7")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(table)
        story.append(Spacer(1, 10))

    def add_section_box(title_text: str, items: List[str], background: str, border: str) -> None:
        data = [
            [markup(f"<b>{esc_text(title_text)}</b>", styles["box_heading"])],
            [markup(bullet_text(items), styles["body"])],
        ]
        box = Table(data, colWidths=[doc.width], hAlign="LEFT")
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(background)),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(border)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(box)
        story.append(Spacer(1, 7))

    story.append(_p(title, styles["title"]))
    story.append(_p("Professional comparison summary · Educational support only · Not a confirmed diagnosis", styles["subtitle"]))

    add_key_value_table([
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Compared Values", result.get("total_compared", 0)),
        ("Improved", result.get("improved", 0)),
        ("Needs Attention", result.get("worsened", 0)),
        ("Stable", result.get("stable", 0)),
        ("Previous Abnormal", result.get("previous_abnormal", 0)),
        ("Current Abnormal", result.get("current_abnormal", 0)),
        ("Risk Category", result.get("risk", "Unknown")),
    ])

    story.append(_p("Comparison Summary", styles["heading"]))
    add_section_box("Abnormal / Worsened Parameters", sections["abnormal_or_worse"], "#FFFAFA", "#FECACA")
    add_section_box("Improved / Normalized Parameters", sections["improved"], "#F7FFF9", "#BBF7D0")
    add_section_box("Stable Parameters", sections["stable"], "#FFFDF2", "#FDE68A")
    add_section_box("Suggested Doctor Discussion Points", sections["doctor_points"], "#F8FBFF", "#BFDBFE")

    story.append(_p("Detailed Parameter-by-Parameter Comparison", styles["heading"]))
    data = [[
        "Parameter", "Previous", "Current", "Delta", "% Change",
        "Reference Range", "Prev Status", "Current Status", "Progress", "Technical Interpretation",
    ]]

    for item in comparisons:
        previous_value = f"{item.get('previous', '')} {item.get('unit', '')}".strip()
        current_value = f"{item.get('current', '')} {item.get('unit', '')}".strip()
        data.append([
            item.get("parameter", ""),
            previous_value,
            current_value,
            item.get("difference", ""),
            item.get("percentage_change", ""),
            item.get("normal_range", ""),
            item.get("previous_status", ""),
            item.get("current_status", ""),
            item.get("progress", ""),
            item.get("clinical_interpretation", ""),
        ])

    table_data = []
    for r_idx, row in enumerate(data):
        style = styles["table_header"] if r_idx == 0 else styles["table_cell"]
        table_data.append([_p(cell, style) for cell in row])

    col_widths = [
        0.92 * inch, 0.66 * inch, 0.66 * inch, 0.52 * inch, 0.62 * inch,
        0.82 * inch, 0.68 * inch, 0.76 * inch, 0.74 * inch, 2.62 * inch,
    ]

    table = Table(table_data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F4EF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#312E81")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#DBE3F0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E6EDF7")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]

    for row_idx, item in enumerate(comparisons, start=1):
        curr_status = str(item.get("current_status", ""))
        prev_status = str(item.get("previous_status", ""))
        progress = str(item.get("progress", ""))
        if prev_status in ["High", "Low"]:
            table_style.append(("BACKGROUND", (6, row_idx), (6, row_idx), colors.HexColor("#FEE2E2")))
        elif prev_status == "Normal":
            table_style.append(("BACKGROUND", (6, row_idx), (6, row_idx), colors.HexColor("#DCFCE7")))
        if curr_status in ["High", "Low"]:
            table_style.append(("BACKGROUND", (7, row_idx), (7, row_idx), colors.HexColor("#FEE2E2")))
        elif curr_status == "Normal":
            table_style.append(("BACKGROUND", (7, row_idx), (7, row_idx), colors.HexColor("#DCFCE7")))
        if progress == "Improved":
            table_style.append(("BACKGROUND", (8, row_idx), (8, row_idx), colors.HexColor("#DCFCE7")))
        elif progress in ["Worsened", "Needs Attention"]:
            table_style.append(("BACKGROUND", (8, row_idx), (8, row_idx), colors.HexColor("#FEE2E2")))
        elif progress == "Stable":
            table_style.append(("BACKGROUND", (8, row_idx), (8, row_idx), colors.HexColor("#FEF9C3")))

    table.setStyle(TableStyle(table_style))
    story.append(table)
    story.append(Spacer(1, 10))

    disclaimer = Table(
        [[markup(f"<b>Important Disclaimer:</b> {esc_text(MEDICAL_DISCLAIMER)}", styles["small"])]],
        colWidths=[doc.width],
    )
    disclaimer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#DBE3F0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(disclaimer)

    doc.build(story)
    return str(out_path)




# ---------------------------------------------------------------------------
# Comparison dashboard landing / option cards
# ---------------------------------------------------------------------------

def build_comparison_dashboard_landing_html() -> str:
    """Reference-matched clean comparison dashboard landing UI."""
    return """
    <style>
      .mb-reference-page {
        font-family: Inter, Arial, sans-serif;
        color: #17203a;
        width: 100%;
        max-width: 1280px;
        margin: 0 auto;
        background: linear-gradient(180deg, #ffffff 0%, #ffffff 86%, #f1fbf6 100%);
        border-radius: 0 0 26px 26px;
        padding: 42px 28px 24px;
      }

      /* Reference-style Page 6 back button from user_app.py */
      #comparison .page-action-row,
      .page-action-row:has(#comparison-back-btn),
      #comparison-back-btn {
        justify-content: flex-start !important;
        align-items: center !important;
        margin: 18px 0 6px 46px !important;
        padding: 0 !important;
        width: fit-content !important;
      }

      #comparison-back-btn,
      #comparison-back-btn button {
        width: 154px !important;
        max-width: 154px !important;
        min-width: 154px !important;
        height: 42px !important;
        min-height: 42px !important;
        padding: 0 16px !important;
        border-radius: 13px !important;
        background: #ffffff !important;
        color: #087057 !important;
        border: 1px solid #b8dcd0 !important;
        box-shadow: 0 10px 24px rgba(17, 62, 50, .08) !important;
        font-size: 14px !important;
        font-weight: 950 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
      }

      #comparison-back-btn:hover,
      #comparison-back-btn button:hover {
        background: #f8fffb !important;
        border-color: #8fcbb8 !important;
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(17, 62, 50, .12) !important;
      }

      .mb-reference-header {
        text-align: center;
        max-width: 760px;
        margin: 0 auto;
      }

      .mb-ref-kicker {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #e9f7f1;
        color: #087057;
        border: 1px solid #d6efe4;
        border-radius: 999px;
        padding: 8px 20px;
        font-size: 12px;
        font-weight: 950;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: 16px;
      }

      .mb-ref-title {
        margin: 0;
        color: #17203a;
        font-size: clamp(36px, 4.2vw, 54px);
        line-height: 1.05;
        font-weight: 950;
        letter-spacing: -.05em;
      }

      .mb-ref-title span { color: #087f5b; }

      .mb-ref-subtitle {
        margin: 18px auto 24px;
        max-width: 700px;
        color: #526079;
        font-size: 16px;
        line-height: 1.55;
        font-weight: 750;
      }

      .mb-ref-tip {
        width: min(620px, 92%);
        margin: 0 auto 30px;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 16px;
        background: #ffffff;
        border: 1px solid #b8dcd0;
        border-radius: 16px;
        padding: 15px 22px;
        color: #17203a;
        font-weight: 900;
        box-shadow: 0 14px 34px rgba(31, 107, 87, .055);
      }

      .mb-ref-tip-icon {
        width: 34px;
        height: 34px;
        border-radius: 12px;
        display: grid;
        place-items: center;
        background: #eaf7f1;
        color: #087057;
        font-size: 21px;
        flex: 0 0 auto;
      }

      .mb-ref-cards {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 30px;
        max-width: 990px;
        margin: 0 auto;
      }

      .mb-ref-card {
        cursor: pointer;
        min-height: 330px;
        background: #ffffff;
        border: 1px solid #e2e8e5;
        border-radius: 22px;
        padding: 30px 24px 24px;
        text-align: center;
        box-shadow: 0 22px 56px rgba(17, 62, 50, .07);
        transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
      }

      .mb-ref-card:hover {
        transform: translateY(-6px);
        border-color: #c7e5da;
        box-shadow: 0 32px 70px rgba(17, 62, 50, .12);
      }

      .mb-ref-icon {
        width: 88px;
        height: 88px;
        border-radius: 26px;
        margin: 0 auto 18px;
        display: grid;
        place-items: center;
        font-size: 44px;
      }

      .mb-ref-icon.green { background: linear-gradient(135deg, #d9f5e8, #f2fbf7); }
      .mb-ref-icon.blue {
        background: linear-gradient(135deg, #dbeafe, #eef6ff);
        color: #105a9b;
        font-size: 41px;
        line-height: .8;
        font-weight: 900;
      }
      .mb-ref-icon.purple { background: linear-gradient(135deg, #ede9fe, #f7f0ff); }

      .mb-ref-badge {
        width: fit-content;
        margin: 0 auto 12px;
        border-radius: 999px;
        padding: 6px 16px;
        font-size: 12px;
        font-weight: 950;
        letter-spacing: .06em;
        text-transform: uppercase;
      }

      .mb-ref-badge.green { background: #dff4eb; color: #087057; }
      .mb-ref-badge.blue { background: #e5f0ff; color: #155c99; }
      .mb-ref-badge.purple { background: #f0e6ff; color: #6c3db8; }

      .mb-ref-card h3 {
        margin: 0 0 13px;
        color: #17203a;
        font-size: 22px;
        font-weight: 950;
        letter-spacing: -.02em;
      }

      .mb-ref-card p {
        min-height: 70px;
        max-width: 250px;
        margin: 0 auto;
        color: #526079;
        font-size: 16px;
        line-height: 1.5;
        font-weight: 700;
      }

      .mb-ref-action {
        margin: 24px auto 0;
        width: 100%;
        max-width: 260px;
        border-radius: 11px;
        padding: 15px 16px;
        color: #ffffff;
        font-size: 16px;
        font-weight: 950;
      }

      .mb-ref-action.green {
        background: linear-gradient(135deg, #07875f, #07986c);
        box-shadow: 0 14px 28px rgba(7, 135, 95, .22);
      }
      .mb-ref-action.blue {
        background: linear-gradient(135deg, #0f4e87, #1264b4);
        box-shadow: 0 14px 28px rgba(15, 78, 135, .22);
      }
      .mb-ref-action.purple {
        background: linear-gradient(135deg, #6c3db8, #7c49d0);
        box-shadow: 0 14px 28px rgba(108, 61, 184, .22);
      }

      .mb-ref-steps {
        max-width: 665px;
        margin: 28px auto 0;
        background: #ffffff;
        border: 1px solid #d5eee3;
        border-radius: 16px;
        box-shadow: 0 18px 38px rgba(17, 62, 50, .06);
        display: grid;
        grid-template-columns: 1fr 44px 1fr 44px 1fr;
        align-items: center;
        padding: 16px 32px;
        text-align: center;
      }

      .mb-ref-step-dot {
        width: 38px;
        height: 38px;
        margin: 0 auto 8px;
        border-radius: 999px;
        display: grid;
        place-items: center;
        background: #087057;
        color: #ffffff;
        font-size: 15px;
        font-weight: 950;
        box-shadow: 0 10px 22px rgba(8,112,87,.18);
      }

      .mb-ref-step-title { color: #17203a; font-size: 14px; font-weight: 950; }
      .mb-ref-step-sub { color: #526079; font-size: 12px; font-weight: 800; margin-top: 3px; }
      .mb-ref-arrow { color: #17203a; font-size: 24px; font-weight: 950; }

      @media(max-width: 950px) {
        .mb-ref-cards { grid-template-columns: 1fr; max-width: 430px; }
        .mb-ref-card { min-height: auto; }
        .mb-ref-card p { min-height: auto; }
      }

      @media(max-width: 650px) {
        #comparison .page-action-row,
        .page-action-row:has(#comparison-back-btn),
        #comparison-back-btn { margin-left: 14px !important; }
        .mb-reference-page { padding: 28px 12px 20px; }
        .mb-ref-tip { width: 100%; align-items: flex-start; }
        .mb-ref-steps { grid-template-columns: 1fr; gap: 12px; }
        .mb-ref-arrow { display: none; }
      }
    </style>

    <div class="mb-reference-page">
      <div class="mb-reference-header">
        <div class="mb-ref-kicker">Step 3 of 3 · Compare Progress</div>
        <h1 class="mb-ref-title">What would you like<br/>to <span>compare?</span></h1>
        <p class="mb-ref-subtitle">
          Choose a comparison mode below. You can use saved reports, upload two reports
          manually, or compare X-ray history side by side.
        </p>

        <div class="mb-ref-tip">
          <div class="mb-ref-tip-icon">💡</div>
          <div>You’re one tap away. Open an option below and compare your health progress.</div>
        </div>
      </div>

      <div class="mb-ref-cards">
        <div class="mb-ref-card" onclick="document.querySelector('#open-saved-compare-btn button, #open-saved-compare-btn')?.click()">
          <div class="mb-ref-icon green">📁</div>
          <div class="mb-ref-badge green">Saved History</div>
          <h3>Saved + Current</h3>
          <p>Select a previous saved lab report and upload a new report to compare progress.</p>
          <div class="mb-ref-action green" onclick="event.stopPropagation(); document.querySelector('#open-saved-compare-btn button, #open-saved-compare-btn')?.click()">Use Saved Report →</div>
        </div>

        <div class="mb-ref-card" onclick="document.querySelector('#open-manual-compare-btn button, #open-manual-compare-btn')?.click()">
          <div class="mb-ref-icon blue">←<br/>→</div>
          <div class="mb-ref-badge blue">Manual</div>
          <h3>Two Lab Reports</h3>
          <p>Upload both old and new lab reports manually for a technical comparison.</p>
          <div class="mb-ref-action blue" onclick="event.stopPropagation(); document.querySelector('#open-manual-compare-btn button, #open-manual-compare-btn')?.click()">Compare PDFs →</div>
        </div>

        <div class="mb-ref-card" onclick="document.querySelector('#open-xray-compare-btn button, #open-xray-compare-btn')?.click()">
          <div class="mb-ref-icon purple">🩻</div>
          <div class="mb-ref-badge purple">Imaging</div>
          <h3>X-ray History</h3>
          <p>View prior and current X-rays side by side with a technical checklist.</p>
          <div class="mb-ref-action purple" onclick="event.stopPropagation(); document.querySelector('#open-xray-compare-btn button, #open-xray-compare-btn')?.click()">Compare X-rays →</div>
        </div>
      </div>

      <div class="mb-ref-steps">
        <div><div class="mb-ref-step-dot">1</div><div class="mb-ref-step-title">Choose</div><div class="mb-ref-step-sub">Pick mode</div></div>
        <div class="mb-ref-arrow">›</div>
        <div><div class="mb-ref-step-dot">2</div><div class="mb-ref-step-title">Upload</div><div class="mb-ref-step-sub">Add files</div></div>
        <div class="mb-ref-arrow">›</div>
        <div><div class="mb-ref-step-dot">3</div><div class="mb-ref-step-title">Compare</div><div class="mb-ref-step-sub">See results</div></div>
      </div>
    </div>
    """

def build_manual_lab_comparison_html(previous_file: Any, current_file: Any) -> str:
    if previous_file is None or current_file is None:
        return _comparison_error("Please upload both previous and current lab reports.")

    previous_ok, previous_message = _is_likely_medical_report_file(previous_file)
    current_ok, current_message = _is_likely_medical_report_file(current_file)
    if not previous_ok:
        return _comparison_error("Previous file: " + previous_message)
    if not current_ok:
        return _comparison_error("Current file: " + current_message)

    previous_values = _values_from_file(previous_file)
    current_values = _values_from_file(current_file)
    result = compare_value_sets(previous_values, current_values)
    return _comparison_html(result, "Manual Lab Report Comparison")


def build_saved_lab_comparison_html(saved_report_path: str, current_file: Any) -> str:
    if not saved_report_path:
        return _comparison_error("No saved previous report selected. Refresh saved reports or use manual upload mode.")
    if current_file is None:
        return _comparison_error("Please upload the current/new lab report.")

    current_ok, current_message = _is_likely_medical_report_file(current_file)
    if not current_ok:
        return _comparison_error("Current file: " + current_message)

    previous_values = _load_saved_values(saved_report_path)
    current_values = _values_from_file(current_file)
    result = compare_value_sets(previous_values, current_values)
    return _comparison_html(result, "Saved Previous Report vs Current Report")


def build_manual_lab_comparison_with_pdf(previous_file: Any, current_file: Any) -> Tuple[str, str | None]:
    """Return (html, pdf_path). Use this when your Gradio event has HTML + File outputs."""
    if previous_file is None or current_file is None:
        return _comparison_error("Please upload both previous and current lab reports."), None

    previous_ok, previous_message = _is_likely_medical_report_file(previous_file)
    current_ok, current_message = _is_likely_medical_report_file(current_file)
    if not previous_ok:
        return _comparison_error("Previous file: " + previous_message), None
    if not current_ok:
        return _comparison_error("Current file: " + current_message), None

    previous_values = _values_from_file(previous_file)
    current_values = _values_from_file(current_file)
    result = compare_value_sets(previous_values, current_values)
    title = "Manual Lab Report Comparison"
    return _comparison_html(result, title), export_lab_comparison_pdf(result, title)


def build_saved_lab_comparison_with_pdf(saved_report_path: str, current_file: Any) -> Tuple[str, str | None]:
    """Return (html, pdf_path). Use this when your Gradio event has HTML + File outputs."""
    if not saved_report_path:
        return _comparison_error("No saved previous report selected. Refresh saved reports or use manual upload mode."), None
    if current_file is None:
        return _comparison_error("Please upload the current/new lab report."), None

    current_ok, current_message = _is_likely_medical_report_file(current_file)
    if not current_ok:
        return _comparison_error("Current file: " + current_message), None

    previous_values = _load_saved_values(saved_report_path)
    current_values = _values_from_file(current_file)
    result = compare_value_sets(previous_values, current_values)
    title = "Saved Previous Report vs Current Report"
    return _comparison_html(result, title), export_lab_comparison_pdf(result, title)



XRAY_IMAGE_SPECIFIC_COMPARISON_PROMPT = """
You are a technical radiology-style X-ray comparison assistant.

Compare the PREVIOUS X-ray image and the CURRENT X-ray image. Do not give a generic checklist. You must describe the actual visible findings from the images.

Important rules:
- This is not a final diagnosis.
- Accept genuine medical X-rays from any body area, including knee, chest, head/skull, spine, joints, arms, legs, hands, feet, shoulder, pelvis/hip, and other radiographs.
- Reject non-X-ray images. If either uploaded image is not a medical X-ray, return exactly: Invalid image uploaded. Please upload a valid X-ray image.
- If both images are valid X-rays but clearly show different body areas, return exactly: The uploaded X-rays appear to be from different body areas. Please upload matching X-rays for comparison.
- Use cautious language such as “appears,” “suggests,” “visible on provided image,” and “radiologist confirmation required.”
- Do not invent findings that are not visible.
- If image quality, projection, or body region is unclear, say so.
- Compare only what can be seen in the uploaded images.
- Focus on technical comparison, not treatment advice.

Return the output in this exact structure:

1. Study / Projection
- Previous image projection:
- Current image projection:
- Image quality limitations:

2. Anatomic Region Reviewed
- Body part / bone region:
- Side marker if visible:
- Adjacent joint involvement if visible:

3. Previous X-ray Findings
- Fracture / abnormality:
- Location:
- Alignment / displacement:
- Cortical or trabecular changes:
- Soft tissue findings:

4. Current X-ray Findings
- Fracture / abnormality:
- Location:
- Alignment / displacement:
- Cortical or trabecular changes:
- Soft tissue findings:

5. Interval Comparison
Choose one:
- Improved
- Worsened
- Unchanged
- Indeterminate

Then explain why using visible image evidence.

6. Key Differences
List 3–5 specific differences between previous and current X-rays.

7. Urgency / Review Flag
Choose one:
- Routine radiologist review
- Prompt radiologist/orthopedic review recommended
- Urgent medical review recommended

Explain the reason briefly.

8. Final Technical Summary
Write a short paragraph comparing the two X-rays in plain language.

Do not output generic instructions like “assess alignment” or “look for cortical break.” Instead, state the actual observed finding, for example:
“Previous image shows a visible fracture line with displacement at the proximal tibia,”
not
“Assess for fracture displacement.”

If you cannot confidently identify actual visual findings from both X-rays, return:
“Insufficient image-specific findings extracted. Please upload clearer AP and lateral X-ray images.”
Do not generate a generic comparison checklist.
""".strip()


def _image_file_to_data_url(uploaded_file: Any) -> str:
    path = _file_path(uploaded_file)
    if not path or not os.path.exists(path):
        return ""
    ext = os.path.splitext(path)[1].lower().replace(".", "")
    mime = "image/png"
    if ext in ["jpg", "jpeg"]:
        mime = "image/jpeg"
    elif ext == "webp":
        mime = "image/webp"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"



def _extract_xray_field(text: str, label: str) -> str:
    """Extract one bullet field from the AI radiology text."""
    pattern = rf"{re.escape(label)}\s*:\s*(.*?)(?=(?:\s+-\s+[A-Z][A-Za-z /]+:)|(?:\n-\s+[A-Z][A-Za-z /]+:)|(?:\n\d+\. )|$)"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        return "Not clearly stated"
    value = re.sub(r"\s+", " ", match.group(1)).strip(" -")
    return value or "Not clearly stated"


def _extract_xray_section(text: str, section_title: str) -> str:
    """Extract a numbered section from the AI output."""
    pattern = rf"{re.escape(section_title)}(.*?)(?=\n?\s*\d+\.\s+[A-Z]|$)"
    match = re.search(pattern, text, flags=re.S | re.I)
    if not match:
        # fallback for single-line output where numbers are not on new lines
        pattern = rf"{re.escape(section_title)}(.*?)(?=\s+\d+\.\s+[A-Z]|$)"
        match = re.search(pattern, text, flags=re.S | re.I)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip(" -")


def _detect_interval_status(text: str) -> tuple[str, str, int]:
    lower = text.lower()
    if "urgent medical review" in lower:
        urgency = "Urgent medical review"
        urgency_score = 90
    elif "prompt radiologist" in lower or "prompt radiologist/orthopedic" in lower:
        urgency = "Prompt specialist review"
        urgency_score = 70
    else:
        urgency = "Routine radiologist review"
        urgency_score = 35

    if re.search(r"\bimproved\b", lower):
        return "Improved", urgency, urgency_score
    if re.search(r"\bworsened\b", lower):
        return "Worsened", urgency, urgency_score
    if re.search(r"\bunchanged\b", lower):
        return "Unchanged", urgency, urgency_score
    return "Indeterminate", urgency, urgency_score


def _extract_key_differences(text: str) -> list[str]:
    section = _extract_xray_section(text, "6. Key Differences")
    if not section:
        return []

    # Split numbered list or semicolon-like long lines.
    pieces = re.split(r"(?:^|\s)(?:\d+[\).]\s+|-+\s+)", section)
    cleaned = []
    for item in pieces:
        item = re.sub(r"\s+", " ", item).strip(" .:-")
        if len(item) > 8 and item.lower() not in ["list 3", "5 specific differences between previous and current x rays"]:
            cleaned.append(item)
    if cleaned:
        return cleaned[:5]

    sentences = re.split(r"(?<=[.!?])\s+", section)
    return [s.strip() for s in sentences if len(s.strip()) > 8][:5]



def _xray_point_key(value: Any) -> str:
    """Compact text key used only to hide repeated comparison points in the UI."""
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    filler = {
        "the", "a", "an", "and", "or", "of", "to", "with", "in", "on", "for", "by", "is", "are",
        "was", "were", "appears", "appear", "visible", "image", "x", "ray", "previous", "current",
    }
    words = [w for w in text.split() if len(w) > 2 and w not in filler]
    return " ".join(words)


def _dedupe_xray_comparison_points(items: List[str], max_items: int = 5) -> List[str]:
    """Remove exact/near repeated X-ray comparison points without changing AI meaning."""
    unique: List[str] = []
    keys: List[str] = []
    for raw in items or []:
        item = re.sub(r"\s+", " ", str(raw or "")).strip(" .:-")
        if len(item) < 8:
            continue
        key = _xray_point_key(item)
        if not key:
            continue
        duplicate = False
        key_words = set(key.split())
        for old_key in keys:
            old_words = set(old_key.split())
            overlap = len(key_words & old_words) / max(1, min(len(key_words), len(old_words)))
            if key == old_key or overlap >= 0.82:
                duplicate = True
                break
        if duplicate:
            continue
        keys.append(key)
        unique.append(item)
        if len(unique) >= max_items:
            break
    return unique


def _clean_xray_summary_repetition(summary_text: str, repeated_points: List[str]) -> str:
    """Keep AI explanation readable by removing only sentences that repeat a shown key point."""
    summary = re.sub(r"\s+", " ", str(summary_text or "")).strip()
    if not summary or not repeated_points:
        return summary
    point_keys = [_xray_point_key(p) for p in repeated_points if _xray_point_key(p)]
    sentences = re.split(r"(?<=[.!?])\s+", summary)
    kept: List[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        s_key = _xray_point_key(sentence)
        # Remove only very close/full repeats. This avoids changing the clinical meaning.
        repeated = False
        for p_key in point_keys:
            if not p_key or not s_key:
                continue
            p_words = set(p_key.split())
            s_words = set(s_key.split())
            overlap = len(p_words & s_words) / max(1, min(len(p_words), len(s_words)))
            if (p_key in s_key or s_key in p_key) and overlap >= 0.80:
                repeated = True
                break
        if not repeated:
            kept.append(sentence)
    return " ".join(kept).strip() or summary


def _xray_quote_for_status(status: str) -> str:
    status = status.lower()
    if status == "improved":
        return "Positive sign: the current image appears to show interval improvement, but radiologist confirmation is still important."
    if status == "worsened":
        return "Attention matters: early specialist review can help clarify the next safe step."
    if status == "unchanged":
        return "Stability is useful information — keeping prior images makes clinical review much stronger."
    return "Clear images and side-by-side comparison help doctors see the story, not just one snapshot."


def _status_badge_class(status: str) -> str:
    status = status.lower()
    if status == "improved":
        return "xray-good"
    if status == "worsened":
        return "xray-bad"
    if status == "unchanged":
        return "xray-stable"
    return "xray-neutral"


def _urgency_badge_class(urgency: str) -> str:
    lower = urgency.lower()
    if "urgent" in lower:
        return "xray-bad"
    if "prompt" in lower:
        return "xray-warn"
    return "xray-good"


def _format_xray_ai_result(text: str, previous_xray: Any, current_xray: Any) -> str:
    prev_src = _image_to_data_uri(previous_xray)
    curr_src = _image_to_data_uri(current_xray)
    clean_text = re.sub(r"\s+", " ", str(text or "")).strip()

    status, urgency, urgency_score = _detect_interval_status(clean_text)
    quote = _xray_quote_for_status(status)

    prev_projection = _extract_xray_field(clean_text, "Previous image projection")
    curr_projection = _extract_xray_field(clean_text, "Current image projection")
    limitations = _extract_xray_field(clean_text, "Image quality limitations")
    body_region = _extract_xray_field(clean_text, "Body part / bone region")
    side_marker = _extract_xray_field(clean_text, "Side marker if visible")

    prev_fracture = _extract_xray_field(clean_text, "Fracture / abnormality")
    # For current, the repeated labels make exact extraction hard; use section extraction too.
    prev_section = _extract_xray_section(clean_text, "3. Previous X-ray Findings")
    curr_section = _extract_xray_section(clean_text, "4. Current X-ray Findings")
    interval_section = _extract_xray_section(clean_text, "5. Interval Comparison")
    final_summary = _extract_xray_section(clean_text, "8. Final Technical Summary")

    prev_location = _extract_xray_field(prev_section, "Location")
    prev_alignment = _extract_xray_field(prev_section, "Alignment / displacement")
    prev_cortical = _extract_xray_field(prev_section, "Cortical or trabecular changes")
    prev_soft = _extract_xray_field(prev_section, "Soft tissue findings")

    curr_abnormality = _extract_xray_field(curr_section, "Fracture / abnormality")
    curr_location = _extract_xray_field(curr_section, "Location")
    curr_alignment = _extract_xray_field(curr_section, "Alignment / displacement")
    curr_cortical = _extract_xray_field(curr_section, "Cortical or trabecular changes")
    curr_soft = _extract_xray_field(curr_section, "Soft tissue findings")

    differences = _dedupe_xray_comparison_points(_extract_key_differences(clean_text), max_items=5)
    if not differences:
        differences = [
            "The AI output did not separate key differences clearly.",
            "Review previous and current alignment with a radiologist.",
            "Confirm healing or displacement on original AP and lateral views.",
        ]

    difference_rows = "".join(
        f"<tr><td>{i}</td><td>{_esc(item)}</td></tr>"
        for i, item in enumerate(differences[:5], start=1)
    )

    final_summary_clean = _clean_xray_summary_repetition(final_summary or interval_section or clean_text, differences[:3])
    final_summary_html = _esc(final_summary_clean).replace("\\n", "<br/>")

    # ── scan type shorthand ──────────────────────────────────────────────────
    prev_scan_type = _esc(prev_projection) if prev_projection else "N/A"
    curr_scan_type = _esc(curr_projection) if curr_projection else "N/A"
    scan_type_changed = prev_scan_type != curr_scan_type

    # ── condition bar position ───────────────────────────────────────────────
    prog_pct = {"Improved": 18, "Unchanged": 48, "Indeterminate": 48, "Worsened": 82}.get(status, 48)

    # ── status colour (green-theme compatible) ───────────────────────────────
    status_color = "#087057"          # default green
    status_bg    = "#ecfdf5"
    status_border= "#d8f3e6"
    if status == "Worsened":
        status_color = "#b91c1c"; status_bg = "#fff1f2"; status_border = "#fecaca"
    elif status == "Unchanged":
        status_color = "#1d4ed8"; status_bg = "#eff6ff"; status_border = "#bfdbfe"

    urgency_color  = "#087057"
    urgency_bg     = "#ecfdf5"
    urgency_border = "#d8f3e6"
    if "urgent" in urgency.lower():
        urgency_color = "#b91c1c"; urgency_bg = "#fff1f2"; urgency_border = "#fecaca"
    elif "prompt" in urgency.lower():
        urgency_color = "#b45309"; urgency_bg = "#fffbeb"; urgency_border = "#fde68a"

    # ── key-action cards – full informative text from AI differences ─────────
    action_num_colors = ["#087057", "#0f5f4b", "#166534"]
    action_cards_html = ""
    for i, diff in enumerate(differences[:3], start=1):
        bg = action_num_colors[(i - 1) % len(action_num_colors)]
        action_cards_html += f"""
        <div class="rx-action-card">
          <div class="rx-action-num" style="background:{bg};">{i}</div>
          <p>{_esc(diff)}</p>
        </div>"""

    # ── detailed findings rows ───────────────────────────────────────────────
    curr_soft_alert = status == "Worsened"
    findings_rows = f"""
      <tr>
        <td class="rx-find-name">Scan type
          {"<span class='rx-tag rx-tag-changed'>Changed</span>" if scan_type_changed else ""}
        </td>
        <td class="rx-find-prev">{prev_scan_type}</td>
        <td class="rx-find-curr">{curr_scan_type}</td>
      </tr>
      <tr>
        <td class="rx-find-name">Fracture / abnormality</td>
        <td class="rx-find-prev">{_esc(prev_fracture) or "Not clearly stated"}</td>
        <td class="rx-find-curr">{_esc(curr_abnormality) or "Not clearly stated"}</td>
      </tr>
      <tr>
        <td class="rx-find-name">Location</td>
        <td class="rx-find-prev">{_esc(prev_location) or "Not clearly stated"}</td>
        <td class="rx-find-curr">{_esc(curr_location) or "Not clearly stated"}</td>
      </tr>
      <tr>
        <td class="rx-find-name">Alignment / displacement</td>
        <td class="rx-find-prev">{_esc(prev_alignment) or "Not clearly stated"}</td>
        <td class="rx-find-curr">{_esc(curr_alignment) or "Not clearly stated"}</td>
      </tr>
      <tr>
        <td class="rx-find-name">Bone &amp; structure</td>
        <td class="rx-find-prev">{_esc(prev_cortical) or "No acute changes"}</td>
        <td class="rx-find-curr">{_esc(curr_cortical) or "No acute changes"}</td>
      </tr>
      <tr>
        <td class="rx-find-name">Soft tissue / other findings
          {"<span class='rx-tag rx-tag-warn'>Worsened</span>" if curr_soft_alert else ""}
        </td>
        <td class="rx-find-prev">{_esc(prev_soft) or "No specific soft tissue findings are visible."}</td>
        <td class="rx-find-curr{"  rx-find-curr-alert" if curr_soft_alert else ""}">{_esc(curr_soft) or "No significant findings"}</td>
      </tr>"""

    return f"""
    <style>
      .rx-wrap {{
        font-family: "Segoe UI", Inter, Arial, sans-serif;
        color: #173b31;
        max-width: 1120px;
        margin: 0 auto;
        padding: 16px;
        border-radius: 28px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fffb 100%);
        border: 1px solid #d8f3e6;
        box-shadow: 0 18px 44px rgba(8,112,87,.07);
      }}

      /* ── Header ── */
      .rx-hero {{
        background: radial-gradient(circle at top left, rgba(187,247,208,.72), transparent 34%),
                    linear-gradient(135deg, #ecfdf5, #ffffff 74%);
        border: 1px solid #c9f2df;
        border-radius: 24px;
        padding: 24px 26px;
        margin-bottom: 16px;
        box-shadow: 0 14px 34px rgba(8,112,87,.08);
      }}
      .rx-hero-eyebrow {{
        font-size: 10px;
        font-weight: 900;
        letter-spacing: .14em;
        text-transform: uppercase;
        color: #6b8179;
        margin-bottom: 4px;
      }}
      .rx-hero-title {{
        font-size: clamp(22px, 3vw, 30px);
        font-weight: 950;
        color: #0f3d2e;
        margin: 0 0 14px;
        letter-spacing: -.02em;
      }}
      .rx-pill-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
      }}
      .rx-pill {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 5px 13px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 800;
        border: 1.5px solid;
      }}

      /* ── Condition bar ── */
      .rx-bar-card {{
        background: #ffffff;
        border: 1px solid #d8f3e6;
        border-radius: 20px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 10px 28px rgba(8,112,87,.06);
      }}
      .rx-bar-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
      }}
      .rx-bar-label {{
        font-size: 10px;
        font-weight: 900;
        letter-spacing: .12em;
        text-transform: uppercase;
        color: #6b8179;
      }}
      .rx-bar-status {{
        font-size: 11px;
        font-weight: 800;
        color: {status_color};
      }}
      .rx-bar-track {{
        height: 8px;
        background: linear-gradient(90deg, #22c55e 0%, #f59e0b 55%, #ef4444 100%);
        border-radius: 999px;
        position: relative;
      }}
      .rx-bar-dot {{
        position: absolute;
        top: 50%;
        left: {prog_pct}%;
        transform: translate(-50%, -50%);
        width: 16px;
        height: 16px;
        background: #fff;
        border: 3px solid {status_color};
        border-radius: 50%;
        box-shadow: 0 2px 8px rgba(0,0,0,.18);
      }}

      /* ── X-ray images ── */
      .rx-image-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-bottom: 16px;
      }}
      .rx-img-card {{
        background: #ffffff;
        border: 1px solid #d8f3e6;
        border-radius: 22px;
        padding: 16px;
        box-shadow: 0 12px 30px rgba(8,112,87,.06);
      }}
      .rx-img-card h3 {{
        margin: 0 0 12px;
        font-size: 15px;
        font-weight: 950;
        color: #0f3d2e;
      }}
      .rx-img-card img {{
        width: 100%;
        max-height: 420px;
        object-fit: contain;
        border-radius: 14px;
        background: #f0fdf4;
      }}

      /* ── Prev vs Current ── */
      .rx-section-title {{
        font-size: 17px;
        font-weight: 900;
        color: #0f3d2e;
        margin: 0 0 10px;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: -.01em;
      }}
      .rx-section-subtitle {{
        margin: -4px 0 12px;
        color: #5f756d;
        font-size: 13px;
        line-height: 1.55;
        font-weight: 650;
      }}
      .rx-vs-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
        margin-bottom: 16px;
      }}
      .rx-vs-col {{
        background: #ffffff;
        border: 1px solid #cfeee1;
        border-radius: 20px;
        padding: 18px 18px;
        box-shadow: 0 10px 24px rgba(8,112,87,.045);
      }}
      .rx-vs-col-label {{
        font-size: 10px;
        font-weight: 900;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid #eaf7f1;
        display: flex;
        align-items: center;
        gap: 8px;
      }}
      .rx-vs-col-label.prev {{ color: #087057; }}
      .rx-vs-col-label.curr {{ color: #b91c1c; }}
      .rx-vs-badge {{
        font-size: 9px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 999px;
        background: #eaf7f1;
        color: #087057;
        letter-spacing: .05em;
      }}
      .rx-vs-badge.latest {{
        background: #fff1f2;
        color: #b91c1c;
      }}
      .rx-vs-row {{
        margin-bottom: 12px;
      }}
      .rx-vs-row:last-child {{ margin-bottom: 0; }}
      .rx-vs-row-lbl {{
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: #6b8179;
        font-weight: 800;
        margin-bottom: 3px;
      }}
      .rx-vs-row-val {{
        font-size: 13px;
        color: #173b31;
        font-weight: 700;
        line-height: 1.45;
      }}
      .rx-vs-row-val.alert {{ color: #b91c1c; font-weight: 800; }}

      /* ── Key Comparison Points ── */
      .rx-actions-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 16px;
      }}
      .rx-action-card {{
        background: linear-gradient(180deg, #ffffff, #f8fffb);
        border: 1px solid #cfeee1;
        border-radius: 18px;
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 10px;
        box-shadow: 0 9px 22px rgba(8,112,87,.045);
        min-height: 128px;
      }}
      .rx-action-num {{
        width: 28px;
        height: 28px;
        border-radius: 50%;
        color: #fff;
        font-size: 13px;
        font-weight: 900;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
      }}
      .rx-action-card p {{
        margin: 0;
        font-size: 13px;
        color: #2d5a4a;
        line-height: 1.6;
        font-weight: 700;
      }}

      /* ── AI Explanation ── */
      .rx-ai-card {{
        background: linear-gradient(135deg, #f8fffb, #ffffff);
        border: 1px solid #cfeee1;
        border-radius: 22px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 12px 30px rgba(8,112,87,.055);
      }}
      .rx-ai-card-title {{
        font-size: 11px;
        font-weight: 900;
        letter-spacing: .12em;
        text-transform: uppercase;
        color: #087057;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
      }}
      .rx-ai-body {{
        background: #ecfdf5;
        border-left: 4px solid #087057;
        border-radius: 0 16px 16px 0;
        padding: 15px 17px;
        font-size: 14px;
        line-height: 1.75;
        color: #173b31;
        font-weight: 720;
      }}

      /* ── Findings table ── */
      .rx-findings-wrap {{
        background: #ffffff;
        border: 1px solid #d8f3e6;
        border-radius: 20px;
        overflow: hidden;
        margin-bottom: 16px;
        box-shadow: 0 10px 24px rgba(8,112,87,.05);
      }}
      .rx-findings-table {{
        width: 100%;
        border-collapse: collapse;
      }}
      .rx-findings-table thead tr {{
        background: #eaf7f1;
        border-bottom: 1.5px solid #d8f3e6;
      }}
      .rx-findings-table th {{
        padding: 13px 16px;
        text-align: left;
        font-size: 11.5px;
        font-weight: 850;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #087057;
      }}
      .rx-findings-table tr + tr {{ border-top: 1px solid #eaf7f1; }}
      .rx-findings-table tbody tr:nth-child(even) {{ background: #fbfefc; }}
      .rx-find-name {{
        padding: 14px 16px;
        font-size: 13.5px;
        font-weight: 780;
        color: #173b31;
        width: 24%;
      }}
      .rx-find-prev, .rx-find-curr {{
        padding: 14px 16px;
        font-size: 13.5px;
        color: #38594f;
        font-weight: 620;
        line-height: 1.55;
      }}
      .rx-find-curr-alert {{ color: #b91c1c !important; font-weight: 800 !important; }}
      .rx-tag {{
        display: inline-block;
        font-size: 10px;
        font-weight: 800;
        padding: 2px 8px;
        border-radius: 999px;
        letter-spacing: .06em;
        text-transform: uppercase;
        margin-left: 6px;
      }}
      .rx-tag-changed {{ background: #eff6ff; color: #1d4ed8; }}
      .rx-tag-warn    {{ background: #fff1f2; color: #b91c1c; }}

      /* ── Disclaimer ── */
      .rx-disclaimer {{
        font-size: 12px;
        color: #6b8179;
        line-height: 1.6;
        font-weight: 700;
        display: flex;
        gap: 8px;
        align-items: flex-start;
        background: #f0fdf4;
        border: 1px solid #d8f3e6;
        border-radius: 14px;
        padding: 12px 16px;
      }}

      @media(max-width: 720px) {{
        .rx-vs-grid, .rx-image-grid, .rx-actions-grid {{ grid-template-columns: 1fr; }}
      }}
    </style>

    <div class="rx-wrap">

      <!-- Header -->
      <div class="rx-hero">
        <div class="rx-hero-eyebrow">Radiology Report</div>
        <h2 class="rx-hero-title">X-Ray Comparison Dashboard</h2>
        <div class="rx-pill-row">
          <span class="rx-pill" style="color:{status_color};border-color:{status_border};background:{status_bg};">
            ↑ Status: {_esc(status)}
          </span>
          <span class="rx-pill" style="color:{urgency_color};border-color:{urgency_border};background:{urgency_bg};">
            ⚑ {_esc(urgency)}
          </span>
          <span class="rx-pill" style="color:#087057;border-color:#d8f3e6;background:#ecfdf5;">
            Region: {_esc(body_region) if body_region else "Chest"}
          </span>
        </div>
      </div>

      <!-- 1. Condition Progression -->
      <div class="rx-bar-card">
        <div class="rx-bar-header">
          <span class="rx-bar-label">Condition Progression</span>
          <span class="rx-bar-status">{_esc(status)}</span>
        </div>
        <div class="rx-bar-track">
          <div class="rx-bar-dot"></div>
        </div>
      </div>

      <!-- 2. X-ray Images -->
      <div class="rx-image-grid">
        <div class="rx-img-card">
          <h3>Previous X-ray</h3>
          <img src="{prev_src}" alt="Previous X-ray" />
        </div>
        <div class="rx-img-card">
          <h3>Current X-ray</h3>
          <img src="{curr_src}" alt="Current X-ray" />
        </div>
      </div>

      <!-- 3. Previous vs Current -->
      <div class="rx-section-title">⇄ Previous vs. Current</div>
      <div class="rx-vs-grid">
        <div class="rx-vs-col">
          <div class="rx-vs-col-label prev">
            ● Previous Scan <span class="rx-vs-badge">Baseline</span>
          </div>
          <div class="rx-vs-row">
            <div class="rx-vs-row-lbl">Scan Type</div>
            <div class="rx-vs-row-val">{prev_scan_type}</div>
          </div>
          <div class="rx-vs-row">
            <div class="rx-vs-row-lbl">Bone &amp; Structure</div>
            <div class="rx-vs-row-val">{_esc(prev_cortical) or "No acute changes"}</div>
          </div>
          <div class="rx-vs-row">
            <div class="rx-vs-row-lbl">Soft Tissue / Other Findings</div>
            <div class="rx-vs-row-val">{_esc(prev_soft) or "No specific soft tissue findings are visible."}</div>
          </div>
        </div>
        <div class="rx-vs-col">
          <div class="rx-vs-col-label curr">
            ● Current Scan <span class="rx-vs-badge latest">Latest</span>
          </div>
          <div class="rx-vs-row">
            <div class="rx-vs-row-lbl">Scan Type</div>
            <div class="rx-vs-row-val">{curr_scan_type}</div>
          </div>
          <div class="rx-vs-row">
            <div class="rx-vs-row-lbl">Bone &amp; Structure</div>
            <div class="rx-vs-row-val">{_esc(curr_cortical) or "No acute changes"}</div>
          </div>
          <div class="rx-vs-row">
            <div class="rx-vs-row-lbl">Soft Tissue / Other Findings</div>
            <div class="rx-vs-row-val{"  alert" if curr_soft_alert else ""}">{_esc(curr_soft) or "No significant findings"}</div>
          </div>
        </div>
      </div>

      <!-- 4. Key Comparison Points -->
      <div class="rx-section-title">✦ Key Comparison Points</div>
      <div class="rx-actions-grid">
        {action_cards_html}
      </div>

      <!-- 5. AI Explanation -->
      <div class="rx-ai-card">
        <div class="rx-ai-card-title">🤖 AI Explanation</div>
        <div class="rx-ai-body">{final_summary_html or "The previous and current X-rays show notable differences. Please consult a radiologist for full interpretation."}</div>
      </div>

      <!-- Detailed Findings Table -->
      <div class="rx-section-title">Detailed X-ray Comparison Findings</div>
      <div class="rx-section-subtitle">Side-by-side technical details from the previous and current X-ray review.</div>
      <div class="rx-findings-wrap">
        <table class="rx-findings-table">
          <thead>
            <tr>
              <th>Finding</th>
              <th>Previous</th>
              <th>Current</th>
            </tr>
          </thead>
          <tbody>{findings_rows}</tbody>
        </table>
      </div>

      <!-- Disclaimer -->
      <div class="rx-disclaimer">
        ℹ️ {_esc(XRAY_DISCLAIMER)}
      </div>

    </div>
    """

def _image_to_data_uri(uploaded_file: Any) -> str:
    path = _file_path(uploaded_file)
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(path)[1].lower().replace(".", "") or "png"
        if ext == "jpg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{encoded}"
    except Exception:
        return ""


def _groq_xray_call(previous_xray: Any, current_xray: Any) -> str | None:
    """Call Groq vision model to compare two X-ray images.
    Returns raw model text or None if the call fails.
    """
    if _groq_vision_client is None:
        return None

    previous_data_url = _image_file_to_data_url(previous_xray)
    current_data_url = _image_file_to_data_url(current_xray)
    if not previous_data_url or not current_data_url:
        return None

    try:
        response = _groq_vision_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": XRAY_IMAGE_SPECIFIC_COMPARISON_PROMPT},
                        {"type": "text", "text": "PREVIOUS X-ray image:"},
                        {"type": "image_url", "image_url": {"url": previous_data_url}},
                        {"type": "text", "text": "CURRENT X-ray image:"},
                        {"type": "image_url", "image_url": {"url": current_data_url}},
                    ],
                }
            ],
            temperature=0.05,
            max_tokens=1600,
        )
        return str(response.choices[0].message.content or "").strip()
    except Exception:
        return None


def build_xray_comparison_html(previous_xray: Any, current_xray: Any) -> str:
    """Compare two valid X-ray images with actual visible, image-specific findings."""
    if previous_xray is None or current_xray is None:
        return _comparison_error("Please upload both previous and current X-ray images.")

    xray_ok, xray_message = _validate_xray_pair_for_comparison(previous_xray, current_xray)
    if not xray_ok:
        return _comparison_error(xray_message)

    # ── Call Groq vision ────────────────────────────────────────────────────
    result = _groq_xray_call(previous_xray, current_xray)

    # ── No result means Groq client missing or call failed ──────────────────
    if not result:
        return _comparison_error(
            "X-ray comparison requires a GROQ_API_KEY. "
            "Please set the GROQ_API_KEY environment variable and restart the app. "
            "Get a free key at https://console.groq.com"
        )

    # ── Validate model output ───────────────────────────────────────────────
    generic_forbidden = [
        "assess alignment",
        "look for cortical break",
        "assess for fracture",
        "evaluate soft tissue",
        "compare the visible projection",
        "look for",
    ]
    if COMPARISON_NON_XRAY_ERROR.lower() in result.lower():
        return _comparison_error(COMPARISON_NON_XRAY_ERROR)
    if XRAY_BODY_MISMATCH_ERROR.lower() in result.lower():
        return _comparison_error(XRAY_BODY_MISMATCH_ERROR)
    if any(term in result.lower() for term in generic_forbidden):
        result = "Insufficient image-specific findings extracted. Please upload clearer AP and lateral X-ray images."

    return _format_xray_ai_result(result, previous_xray, current_xray)

def save_xray_comparison_snapshot(previous_xray: Any, current_xray: Any) -> str | None:
    """Optional helper to save uploaded X-ray pair paths/copies for later history review."""
    prev_path = _file_path(previous_xray)
    curr_path = _file_path(current_xray)
    if not prev_path or not curr_path or not os.path.exists(prev_path) or not os.path.exists(curr_path):
        return None

    snapshot_id = uuid.uuid4().hex[:10]
    folder = XRAY_HISTORY_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{snapshot_id}"
    folder.mkdir(parents=True, exist_ok=True)

    prev_out = folder / f"previous{Path(prev_path).suffix.lower() or '.png'}"
    curr_out = folder / f"current{Path(curr_path).suffix.lower() or '.png'}"
    try:
        prev_out.write_bytes(Path(prev_path).read_bytes())
        curr_out.write_bytes(Path(curr_path).read_bytes())
        meta = {
            "id": snapshot_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "previous": str(prev_out),
            "current": str(curr_out),
        }
        (folder / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(folder)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Ready-to-use full Gradio comparison dashboard section
# ---------------------------------------------------------------------------

def build_full_comparison_dashboard_ui():
    """Build complete comparison dashboard as a screen-style flow.

    Behavior:
    - First screen shows only the cards.
    - Clicking card button hides the card screen.
    - The selected upload/comparison interface opens like the next page.
    - Back button returns to the card screen.
    """
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError("Gradio is required to build the comparison dashboard UI.") from exc

    def open_saved_screen():
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def open_manual_screen():
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    def open_xray_screen():
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    def back_to_cards():
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    gr.HTML("""
    <style>
      .mb-hidden-trigger-row {
        display:none !important;
        height:0 !important;
        overflow:hidden !important;
        margin:0 !important;
        padding:0 !important;
      }
      .mb-screen-panel {
        max-width: 1050px;
        margin: 18px auto 0;
        background: rgba(255,255,255,.88);
        border: 1px solid #d8f3e6;
        border-radius: 24px;
        padding: 18px;
        box-shadow: 0 18px 42px rgba(17, 62, 50, .08);
      }
      .mb-panel-title {
        background:#f8fffb;
        border:1px solid #d8f3e6;
        border-radius:18px;
        padding:16px;
        margin:0 0 14px;
        color:#173b31;
        font-weight:850;
      }
      .mb-panel-title b {
        color:#0f5f4b;
      }
      .mb-panel-back {
        margin-bottom: 12px;
      }
      .mb-panel-back button,
      .mb-panel-back {
        background:#ffffff !important;
        color:#173b31 !important;
        border:1px solid #bddbd1 !important;
        border-radius:16px !important;
        font-weight:950 !important;
        box-shadow:0 10px 24px rgba(17,62,50,.06) !important;
      }
    </style>
    """)

    # Screen 1: cards only.
    with gr.Column(visible=True) as cards_screen:
        gr.HTML(build_comparison_dashboard_landing_html())

        # Hidden real Gradio buttons. Card HTML triggers these via JS.
        with gr.Row(elem_classes=["mb-hidden-trigger-row"]):
            open_saved_btn = gr.Button("Use Saved Report", elem_id="open-saved-compare-btn")
            open_manual_btn = gr.Button("Compare Two PDFs", elem_id="open-manual-compare-btn")
            open_xray_btn = gr.Button("Compare X-rays", elem_id="open-xray-compare-btn")

    # Screen 2A: saved report comparison.
    with gr.Column(visible=False, elem_classes=["mb-screen-panel"]) as saved_panel:
        close_saved_btn = gr.Button("← Back to comparison choices", elem_classes=["mb-panel-back"])
        gr.HTML("""
        <div class="mb-panel-title">
          <b>Saved comparison:</b>
          Select a previous saved lab report, then upload your latest/current report.
        </div>
        """)

        with gr.Row():
            saved_report_dropdown = gr.Dropdown(
                choices=get_saved_report_choices(),
                label="Select Previous Saved Lab Report",
                interactive=True,
            )
            refresh_saved_btn = gr.Button("Refresh Saved Reports")

        current_saved_file = gr.File(
            label="Upload Current / New Lab Report",
            file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"],
        )

        compare_saved_btn = gr.Button("Compare Saved Previous + Current Report", variant="primary")
        saved_result_html = gr.HTML()
        saved_pdf_file = gr.File(label="Download Technical Comparison PDF")

        refresh_saved_btn.click(
            fn=refresh_saved_report_choices,
            inputs=[],
            outputs=[saved_report_dropdown],
        )

        compare_saved_btn.click(
            fn=build_saved_lab_comparison_with_pdf,
            inputs=[saved_report_dropdown, current_saved_file],
            outputs=[saved_result_html, saved_pdf_file],
        )

    # Screen 2B: manual two-report comparison.
    with gr.Column(visible=False, elem_classes=["mb-screen-panel"]) as manual_panel:
        close_manual_btn = gr.Button("← Back to comparison choices", elem_classes=["mb-panel-back"])
        gr.HTML("""
        <div class="mb-panel-title">
          <b>Manual comparison:</b>
          Upload both old and new lab reports manually for technical comparison.
        </div>
        """)

        with gr.Row():
            previous_manual_file = gr.File(
                label="Upload Previous / Old Lab Report",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"],
            )
            current_manual_file = gr.File(
                label="Upload Current / New Lab Report",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"],
            )

        compare_manual_btn = gr.Button("Compare Both Uploaded Lab Reports", variant="primary")
        manual_result_html = gr.HTML()
        manual_pdf_file = gr.File(label="Download Technical Comparison PDF")

        compare_manual_btn.click(
            fn=build_manual_lab_comparison_with_pdf,
            inputs=[previous_manual_file, current_manual_file],
            outputs=[manual_result_html, manual_pdf_file],
        )

    # Screen 2C: x-ray comparison.
    with gr.Column(visible=False, elem_classes=["mb-screen-panel"]) as xray_panel:
        close_xray_btn = gr.Button("← Back to comparison choices", elem_classes=["mb-panel-back"])
        gr.HTML("""
        <div class="mb-panel-title">
          <b>X-ray comparison:</b>
          Upload previous and current X-ray images for side-by-side technical comparison guidance.
        </div>
        """)

        with gr.Row():
            previous_xray_file = gr.File(
                label="Upload Previous X-ray",
                file_types=[".png", ".jpg", ".jpeg", ".webp"],
            )
            current_xray_file = gr.File(
                label="Upload Current X-ray",
                file_types=[".png", ".jpg", ".jpeg", ".webp"],
            )

        compare_xray_btn = gr.Button("Compare X-ray History", variant="primary")
        xray_result_html = gr.HTML()

        compare_xray_btn.click(
            fn=build_xray_comparison_html,
            inputs=[previous_xray_file, current_xray_file],
            outputs=[xray_result_html],
        )

    screen_outputs = [cards_screen, saved_panel, manual_panel, xray_panel]

    open_saved_btn.click(fn=open_saved_screen, inputs=[], outputs=screen_outputs)
    open_manual_btn.click(fn=open_manual_screen, inputs=[], outputs=screen_outputs)
    open_xray_btn.click(fn=open_xray_screen, inputs=[], outputs=screen_outputs)

    close_saved_btn.click(fn=back_to_cards, inputs=[], outputs=screen_outputs)
    close_manual_btn.click(fn=back_to_cards, inputs=[], outputs=screen_outputs)
    close_xray_btn.click(fn=back_to_cards, inputs=[], outputs=screen_outputs)

