# 
# CELL 2 OF 5 IMPORTS
# 

import os
import sys
import subprocess
import importlib
import base64
import mimetypes
import re
import io
import json
import uuid
import math
import time
import tempfile
import zipfile
import warnings
import html
import hashlib
from collections import Counter

warnings.filterwarnings("ignore")

# Load .env automatically for local VS Code runs without requiring python-dotenv.
def _load_local_env_file():
  try:
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
      return
    with open(env_path, "r", encoding="utf-8") as f:
      for raw_line in f:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
          continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
          os.environ[key] = value
  except Exception:
    pass

_load_local_env_file()

# Repair sympy in the active runtime if needed before importing easyocr/torch.
try:
  import sympy
  if not hasattr(sympy, "core"):
    raise AttributeError("sympy.core missing")
except Exception:
  subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--force-reinstall", "sympy==1.13.1", "mpmath==1.3.0"],
    check=True,
  )
  importlib.invalidate_caches()
  for mod in [m for m in list(sys.modules) if m.startswith("sympy")]:
    del sys.modules[mod]
  import sympy

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fitz
import gradio as gr
try:
  import easyocr
except Exception:
  easyocr = None
try:
  import pytesseract
except Exception:
  pytesseract = None

from PIL import Image
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.platypus import Image as RLImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
try:
  import arabic_reshaper
  from bidi.algorithm import get_display
except Exception:
  arabic_reshaper = None
  get_display = None



# Admin panel event logging (safe: app still works if admin module is unavailable)
try:
  from src.admin_storage import log_analysis_event
except Exception:
  def log_analysis_event(payload):
    return False

try:
  from src.comparison_dashboard import (
    build_full_comparison_dashboard_ui,
    build_manual_lab_comparison_html,
    build_saved_lab_comparison_html,
    build_xray_comparison_html,
    get_saved_report_choices,
    refresh_saved_report_choices,
    save_lab_report_snapshot,
  )
except Exception:
  build_full_comparison_dashboard_ui = None
  build_manual_lab_comparison_html = None
  build_saved_lab_comparison_html = None
  build_xray_comparison_html = None
  get_saved_report_choices = lambda: []
  refresh_saved_report_choices = lambda: None
  save_lab_report_snapshot = lambda payload: False

print("All imports loaded successfully")
print("sympy:", getattr(sympy, "__version__", "unknown"))
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("gradio:", gr.__version__)


# 
# CELL 3 OF 5 CONFIG + HELPERS
# 

# ============================================================
# GROQ CONFIG
# ============================================================
# Recommended:
# os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Text model for report explanation / Q&A
TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")

# Vision-capable model for X-ray review
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# Fast demo mode keeps the app responsive during presentation/demo.
# It skips optional LLM calls for lab parsing/explanation and uses the app's local
# summary logic instead, so Analyze does not wait for a slow network/API response.
MEDIBUDDY_FAST_MODE = os.getenv("MEDIBUDDY_FAST_MODE", "0").strip().lower() not in {"0", "false", "no", "off"}

client = None
if GROQ_API_KEY:
  try:
    client = Groq(api_key=GROQ_API_KEY)
    print("Groq client initialized successfully")
    print("Text model:", TEXT_MODEL)
    print("Vision model:", VISION_MODEL)
  except Exception as e:
    print("Groq client init failed:", e)
else:
  print("Groq API key not set.")
  print("OCR, parsing, charts, PDF export, and evaluation will still work.")
  print("AI explanation, chat, and X-ray analysis require a valid GROQ_API_KEY.")

ocr_reader = None

def get_easyocr_reader():
  """Load EasyOCR only when Tesseract/native extraction is not enough."""
  global ocr_reader
  if easyocr is None:
    return None
  if ocr_reader is None:
    ocr_reader = easyocr.Reader(['en'], gpu=False)
  return ocr_reader

CURRENT_REPORT_CONTEXT = {
  "raw_text": "",
  "formatted_text": "",
  "report_category": "",
  "report_subtype": "",
  "patient_info": {},
  "lab_records": [],
  "radiology_sections": {},
  "ai_explanation": "",
  "summary": "",
  "ocr_quality": {},
  "risk_score": {},
  "health_suggestions": [],
  "ml_classifier": {},
}

LAB_KEYWORDS = [
  "hemoglobin", "haemoglobin", "platelet", "wbc", "rbc", "cbc", "glucose", "hba1c", "cholesterol",
  "triglycerides", "hdl", "ldl", "creatinine", "urea", "bun", "uric acid",
  "bilirubin", "alt", "ast", "alkaline phosphatase", "tsh", "t3", "t4"
]

RADIOLOGY_KEYWORDS = [
  "findings", "impression", "technique", "exam", "examination", "radiology",
  "x-ray", "xray", "ct", "mri", "ultrasound", "sonography", "no focal consolidation",
  "cardiomediastinal", "pleural", "pneumothorax", "opacities"
]

XRAY_KEYWORDS = [
  "chest x ray", "chest x-ray", "cxr", "x ray", "x-ray", "radiograph"
]

LAB_PATTERNS = [
  r"([A-Za-z][A-Za-z0-9\s\/\-\(\)%\+]{1,45})\s+([<>]?\s*\d+\.?\d*)\s*([A-Za-z%\/\-\^0-9\.]*)\s+(\d+\.?\d*\s*[-]\s*\d+\.?\d*)",
  r"([A-Za-z][A-Za-z0-9\s\/\-\(\)%\+]{1,45})\s*[:\-]?\s*([<>]?\s*\d+\.?\d*)\s*([A-Za-z%\/\-\^0-9\.]*)\s*\(?\s*(\d+\.?\d*\s*[-]\s*\d+\.?\d*)\s*\)?",
]


LAB_DISPLAY_PATTERNS = [
  ("Hemoglobin", "HGB", [r"\bhemoglobin\b", r"\bhaemoglobin\b", r"\bhgb\b", r"\bhb\b"]),
  ("WBC", "WBC", [r"\bwbc\b", r"white\s*blood\s*cell", r"white\s*blood\s*cells"]),
  ("Platelets", "PLT", [r"\bplatelet\b", r"\bplatelets\b", r"\bplt\b"]),
  ("Glucose", "Gluc", [r"\bglucose\b", r"fasting\s*glucose", r"blood\s*sugar"]),
  ("Creatinine", "Creat", [r"\bcreatinine\b", r"serum\s*creatinine"]),
  ("RBC", "RBC", [r"\brbc\b", r"red\s*blood\s*cell"]),
  ("Hematocrit", "HCT", [r"\bhematocrit\b", r"\bhaematocrit\b", r"\bhct\b"]),
  ("MCV", "MCV", [r"\bmcv\b"]),
  ("MCH", "MCH", [r"\bmch\b"]),
  ("MCHC", "MCHC", [r"\bmchc\b"]),
  ("Neutrophils", "Neut", [r"\bneutrophil"]),
  ("Lymphocytes", "Lymph", [r"\blymphocyte"]),
  ("Urea", "Urea", [r"\burea\b", r"\bbun\b"]),
  ("Uric Acid", "Uric", [r"uric\s*acid"]),
  ("ALT", "ALT", [r"\balt\b"]),
  ("AST", "AST", [r"\bast\b"]),
  ("Bilirubin", "Bili", [r"bilirubin"]),
  ("TSH", "TSH", [r"\btsh\b"]),
  ("T3", "T3", [r"\bt3\b"]),
  ("T4", "T4", [r"\bt4\b"]),
  ("Cholesterol", "Chol", [r"cholesterol"]),
  ("Triglycerides", "TG", [r"triglyceride"]),
  ("HDL", "HDL", [r"\bhdl\b"]),
  ("LDL", "LDL", [r"\bldl\b"]),
  ("NLR", "NLR", [r"\bnlr\b", r"neutrophils?\s*lymphocytes?\s*ratio"]),
  ("RDW", "RDW", [r"\brdw\b", r"red\s*cell\s*distribution"]),
  ("MPV", "MPV", [r"\bmpv\b", r"mean\s*platelet\s*volume"]),
  ("PDW", "PDW", [r"\bpdw\b"]),
  ("PCV", "PCV", [r"\bpcv\b", r"packed\s*cell\s*volume"]),
  ("ESR", "ESR", [r"\besr\b"]),
  ("CRP", "CRP", [r"\bcrp\b"]),
  ("HbA1c", "HbA1c", [r"\bhba1c\b", r"glycated\s*hemoglobin", r"glycosylated\s*hemoglobin"]),
]

FILLER_LABEL_WORDS = {
  "which", "is", "lower", "than", "the", "normal", "range", "of", "higher",
  "parameter", "value", "test", "level", "levels", "result", "results",
  "below", "above", "within", "and", "or", "to", "for"
}

STRUCTURED_BLOCKLIST_WORDS = {
  "aga", "khan", "hospital", "laboratory", "lab", "road", "street", "avenue", "block",
  "po", "box", "karachi", "pakistan", "phone", "fax", "email", "website", "www", "com",
  "patient", "name", "gender", "age", "date", "doctor", "consultant", "specimen", "sample",
  "collection", "received", "reported", "registration", "invoice", "center", "centre", "diagnostic",
  "branch", "location", "address", "stadium", "lobbessi", "lobbesi"
}

MEDICAL_UNIT_REGEX = re.compile(
  r"^(?:g/dl|gm/dl|mg/dl|mmol/l|meq/l|iu/l|u/l|pg|fl|%|x10\^?\d+/?(?:l|ul|l|ml)?|x10\^?\d+|10\^?\d+/?(?:l|ul|l|ml)?|10\*\d+/?(?:l|ul|l|ml)?|cells/?(?:cmm|ul|l|ml)|/hpf|/lpf|ng/ml|pg/ml|miu/l|iu/ml|u/ml|?g/dl|?mol/l|mm/hr|ratio|sec|seconds?)$",
  flags=re.IGNORECASE,
)

GENERAL_MEDICAL_PARAM_PATTERNS = [
  r"\bsodium\b", r"\bpotassium\b", r"\bchloride\b", r"\bcalcium\b", r"\bmagnesium\b",
  r"\bphosph(?:orus|ate)?\b", r"\balbumin\b", r"\bglobulin\b", r"\bprotein\b", r"\bbilirubin\b",
  r"\balkaline\s*phosphatase\b", r"\besr\b", r"\bcrp\b", r"\bhba1c\b", r"\bsgpt\b",
  r"\bsgot\b", r"\bheart\s*rate\b", r"\bpulse\b", r"\btemperature\b", r"\bspo2\b",
  r"\boxygen\s*saturation\b", r"\bblood\s*pressure\b", r"\burea\b", r"\bcreatinine\b",
  r"\bglucose\b", r"\bcholesterol\b", r"\btriglycerides\b", r"\bhdl\b", r"\bldl\b",
  r"\bhemoglobin\b", r"\bhaemoglobin\b", r"\bhematocrit\b", r"\bhaematocrit\b", r"\bwbc\b", r"\brbc\b", r"\bplatelets?\b", r"\bmcv\b", r"\bmchc?\b", r"\bnlr\b",
]

STRUCTURED_ALLOWED_SINGLE_WORD = {
  "wbc", "rbc", "mcv", "mch", "mchc", "hgb", "hb", "plt", "esr", "crp", "tsh", "t3", "t4",
  "alt", "ast", "hdl", "ldl", "bun", "rdw", "mpv", "pdw", "pcv", "pct", "hct", "hba1c", "nlr"
}


def extract_numeric_with_optional_unit(text):
  raw = safe_str(text).strip()
  if not raw:
    return "", ""
  raw = raw.replace(",", " ")
  m = re.search(r"([<>]?\d+(?:\.\d+)?)\s*([A-Za-z%/^xX0-9\.-]+)?", raw)
  if not m:
    return "", ""
  value = clean_value_text(m.group(1))
  unit = safe_str(m.group(2)).strip(" .,:;()[]{}")
  return value, unit


def clean_unit_text(unit_text):
  unit = safe_str(unit_text).strip()
  if not unit:
    return ""
  unit = unit.replace("", "")
  unit = re.split(r"\s|,|;|:|\)|\]", unit)[0]
  unit = unit.strip(" .,:;()[]{}")
  if not unit:
    return ""
  if any(bad in unit.lower() for bad in ["karachi", "road", "box", "hospital", "report", "normal", "range"]):
    return ""
  return unit


def is_plausible_medical_unit(unit):
  unit = clean_unit_text(unit)
  if not unit:
    return True
  if MEDICAL_UNIT_REGEX.fullmatch(unit):
    return True
  low = unit.lower()
  if any(tok in low for tok in ["g/dl", "mg/dl", "mmol", "meq", "iu/l", "u/l", "%", "pg", "fl", "ng/ml", "pg/ml", "mol/l", "/hpf", "/lpf", "ratio"]):
    return True
  if re.fullmatch(r"(?:x?10\^?\d+|\d+)/?(?:l|ul|ml)?", low):
    return True
  return False


def looks_like_real_medical_parameter(param):
  param = safe_str(param).strip()
  if not param:
    return False
  low = param.lower()
  if any(word in low for word in STRUCTURED_BLOCKLIST_WORDS):
    return False
  if re.search(r"\d", param):
    return False
  words = re.findall(r"[A-Za-z]+", param)
  if not words:
    return False
  if len(words) > 4:
    return False
  if len(words) == 1 and words[0].lower() not in STRUCTURED_ALLOWED_SINGLE_WORD:
    if len(words[0]) < 3:
      return False
  for _, _, patterns in LAB_DISPLAY_PATTERNS:
    for pattern in patterns:
      if re.search(pattern, low):
        return True
  for pattern in GENERAL_MEDICAL_PARAM_PATTERNS:
    if re.search(pattern, low):
      return True
  title_words = [w for w in words if len(w) > 1]
  if 1 <= len(title_words) <= 3 and all(w.lower() not in STRUCTURED_BLOCKLIST_WORDS for w in title_words):
    return True
  return False


def normalize_structured_record(param, value, unit, ref):
  param = clean_parameter_name(param)
  value, value_unit = extract_numeric_with_optional_unit(value)
  unit = clean_unit_text(unit) or clean_unit_text(value_unit)
  ref = normalize_range_text(ref)
  low, high = parse_reference_bounds(ref)

  if not param or not looks_like_real_medical_parameter(param):
    return None
  if not value or not re.fullmatch(r"[<>]?\d+(?:\.\d+)?", value.replace(",", "")):
    return None
  if low is None or high is None or low >= high:
    return None
  if high > 100000 or low < 0:
    return None

  # OCR sometimes drops decimal points in percentages, e.g. 70.4% becomes 704%.
  # If the reference range is percentage-like and the value is impossible, repair it safely.
  try:
    numeric_value = float(value.replace(",", "").replace("<", "").replace(">", ""))
    unit_low = safe_str(unit).lower().replace(" ", "")
    if unit_low == "%" and high <= 100 and 100 < numeric_value < 1000:
      repaired = numeric_value / 10.0
      if low <= repaired <= max(high * 1.8, high + 20):
        value = (f"{repaired:.1f}" if repaired % 1 else f"{int(repaired)}")
  except Exception:
    pass

  if unit and not is_plausible_medical_unit(unit):
    return None

  return {
    "Parameter": param,
    "Value": value,
    "Unit": unit,
    "Reference Range": ref,
  }


def shorten_parameter_for_card(full_label):
  full_label = safe_str(full_label).strip()
  if not full_label:
    return ""
  manual = {
    "Hemoglobin": "HGB",
    "WBC": "WBC",
    "Platelets": "PLT",
    "Glucose": "Gluc",
    "Creatinine": "Creat",
    "RBC": "RBC",
    "Hematocrit": "HCT",
    "Neutrophils": "Neut",
    "Lymphocytes": "Lymph",
    "Triglycerides": "TG",
  }
  if full_label in manual:
    return manual[full_label]
  words = full_label.split()
  if len(words) == 1:
    return words[0][:10]
  return " ".join(w[:6] for w in words[:2])

def clean_parameter_name(raw_name):
  raw = safe_str(raw_name).strip()
  if not raw:
    return ""

  lowered = raw.lower()
  lowered = re.sub(r"[_\-]+", " ", lowered)
  lowered = re.sub(r"\s+", " ", lowered).strip()

  # Normalize common OCR forms from CBC reports:
  # M.C.V. -> mcv, R.D.W. -> rdw, HCT/haematocrit -> Hematocrit.
  compact = re.sub(r"[^a-z0-9]+", "", lowered)
  abbreviation_map = [
    ("haemoglobin", "Hemoglobin"), ("hemoglobin", "Hemoglobin"),
    ("haematocrit", "Hematocrit"), ("hematocrit", "Hematocrit"),
    ("platelets", "Platelets"), ("platelet", "Platelets"),
    ("neutrophilslymphocytesratio", "NLR"), ("nlr", "NLR"),
    ("mchc", "MCHC"), ("mcv", "MCV"), ("mch", "MCH"),
    ("rdw", "RDW"), ("rbc", "RBC"), ("wbc", "WBC"), ("hct", "Hematocrit"),
    ("hgb", "Hemoglobin"), ("hb", "Hemoglobin"),
    ("neutrophils", "Neutrophils"), ("lymphocytes", "Lymphocytes"),
    ("eosinophils", "Eosinophils"), ("monocytes", "Monocytes"), ("basophils", "Basophils"),
  ]
  for token, label in abbreviation_map:
    if compact.startswith(token) or token in compact[:40]:
      return label

  for full_label, short_label, patterns in LAB_DISPLAY_PATTERNS:
    for pattern in patterns:
      if re.search(pattern, lowered):
        return full_label

  words = re.findall(r"[A-Za-z][A-Za-z0-9\+\-/()%]*", raw)
  cleaned_words = [w for w in words if w.lower() not in FILLER_LABEL_WORDS]

  if cleaned_words:
    label = " ".join(cleaned_words[:4]).strip(" :-")
    label = re.sub(r"\s+", " ", label)
    return label.title()

  return raw.strip(" :-").title()

def get_display_parameter_labels(name, fallback_idx=None):
  full_label = clean_parameter_name(name)
  if not full_label:
    return "Parameter", "Param"
  short_label = shorten_parameter_for_card(full_label)
  return full_label, short_label

def normalize_param_name(name):
  return clean_parameter_name(name)

def safe_str(x):

  return "" if x is None else str(x)

def clean_text(text):
  if not text:
    return ""
  text = text.replace("\x0c", " ")
  text = text.replace("", "- ")
  text = re.sub(r"[ \t]+", " ", text)
  text = re.sub(r"\n{3,}", "\n\n", text)
  return text.strip()

def normalize_text_for_compare(text):
  text = safe_str(text).lower()
  text = re.sub(r"[^a-z0-9\s]", " ", text)
  text = re.sub(r"\s+", " ", text).strip()
  return text

def _write_temp_ocr_image(image_array):
  """Write an OCR candidate image to a temporary PNG file."""
  temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
  temp_path = temp_img.name
  temp_img.close()
  cv2.imwrite(temp_path, image_array)
  return temp_path


def _ocr_candidate_score(text):
  """Score extracted text so the clearest OCR pass can be selected."""
  text = clean_text(text)
  if not text:
    return 0

  words = re.findall(r"[A-Za-z0-9]+", text)
  numbers = re.findall(r"[<>]?\d+(?:,\d{3})*(?:\.\d+)?", text)
  ranges = re.findall(
    r"\d+(?:,\d{3})*(?:\.\d+)?\s*(?:-|||to)\s*\d+(?:,\d{3})*(?:\.\d+)?",
    text,
    flags=re.IGNORECASE,
  )
  units = re.findall(
    r"\b(?:mg/dl|g/dl|gm/dl|mmol/l|u/l|iu/l|x10\^?\d+/?(?:l|ul|l|ml)?|10\^?\d+/?(?:l|ul|l|ml)?|%|fl|pg|ng/ml|miu/l|iu/ml)\b",
    text,
    flags=re.IGNORECASE,
  )

  try:
    medical_score = medical_text_confidence_score(text)
  except Exception:
    medical_score = 0

  # Prefer candidates that contain complete medical-value patterns, not just lots of noisy text.
  return (
    len(words)
    + len(numbers) * 3
    + len(ranges) * 18
    + len(units) * 12
    + medical_score * 25
  )



def _dynamic_lab_row_score(text):
  """Score OCR candidates by how many complete lab rows they preserve.

  This is dynamic: it does not contain report values. It only checks whether
  the OCR text preserves a medical parameter + patient value + reference range
  in the same nearby row/window. The actual values still come from the upload.
  """
  text = clean_text(text)
  if not text:
    return 0

  label_patterns = [
    r"ha?emoglobin", r"ha?ematocrit",
    r"r\s*\.?\s*b\s*\.?\s*c\.?", r"w\s*\.?\s*b\s*\.?\s*c\.?",
    r"m\s*\.?\s*c\s*\.?\s*v\.?", r"m\s*\.?\s*c\s*\.?\s*h\.?",
    r"m\s*\.?\s*c\s*\.?\s*h\s*\.?\s*c\.?", r"r\s*\.?\s*d\s*\.?\s*w\.?",
    r"neutrophils?", r"lymphocytes?", r"eosinophils?", r"monocytes?",
    r"basophils?", r"platelets?", r"glucose", r"creatinine", r"urea",
    r"cholesterol", r"triglycerides", r"hba1c", r"tsh", r"alt", r"ast",
    r"bilirubin", r"troponin", r"ck\s*-?\s*mb", r"sodium", r"potassium", r"chloride",
  ]
  number_re = r"[<>]?\d+(?:,\d{3})*(?:\.\d+)?"
  range_re = r"\d+(?:,\d{3})*(?:\.\d+)?\s*(?:-|||to)\s*\d+(?:,\d{3})*(?:\.\d+)?"
  unit_re = r"(?:g/dl|gm/dl|mg/dl|mmol/l|u/l|iu/l|x10\^?\d+/?[a-z]*|10\^?\d+/?[a-z]*|%|fl|pg|ng/ml|miu/l|ratio)"

  lines = [re.sub(r"\s+", " ", safe_str(x)).strip() for x in text.splitlines() if safe_str(x).strip()]
  hits = 0
  seen_labels = set()
  for i in range(len(lines)):
    for size in (1, 2, 3):
      chunk = " ".join(lines[i:i + size])
      if not re.search(number_re, chunk) or not re.search(range_re, chunk, flags=re.I):
        continue
      for pat in label_patterns:
        if re.search(pat, chunk, flags=re.I):
          if pat not in seen_labels:
            seen_labels.add(pat)
            hits += 1
          if re.search(unit_re, chunk, flags=re.I):
            hits += 0.25
          break
  return hits


def _local_structured_row_count_for_ocr_candidate(text):
  """Count rows recoverable by the local parser without using hardcoded values.

  Used only to choose the clearest OCR pass. It does not create output values;
  it simply prefers the OCR text where more real rows can be parsed.
  """
  try:
    parser = globals().get("parse_noisy_cbc_records")
    if callable(parser):
      return len(parser(text))
  except Exception:
    pass
  try:
    return int(_dynamic_lab_row_score(text))
  except Exception:
    return 0

def _merge_text_blocks(*blocks):
  """Merge native PDF text and OCR text without repeating identical lines."""
  merged_lines = []
  seen = set()

  for block in blocks:
    block = safe_str(block)
    if not block.strip():
      continue

    # Keep row boundaries when available because lab parsing depends on rows.
    for line in block.splitlines():
      line = re.sub(r"\s+", " ", safe_str(line)).strip()
      if not line:
        continue
      key = re.sub(r"[^a-z0-9]+", "", line.lower())
      if not key or key in seen:
        continue
      seen.add(key)
      merged_lines.append(line)

  return clean_text("\n".join(merged_lines))


def _looks_like_strong_page_text(page_text):
  """Decide whether embedded PDF text is strong enough or OCR fallback is needed."""
  text = clean_text(page_text)
  if not text:
    return False

  words = len(re.findall(r"[A-Za-z0-9]+", text))
  numbers = len(re.findall(r"[<>]?\d+(?:,\d{3})*(?:\.\d+)?", text))
  ranges = len(re.findall(
    r"\d+(?:,\d{3})*(?:\.\d+)?\s*(?:-|||to)\s*\d+(?:,\d{3})*(?:\.\d+)?",
    text,
    flags=re.IGNORECASE,
  ))

  try:
    medical_score = medical_text_confidence_score(text)
  except Exception:
    medical_score = 0

  return words >= 70 and medical_score >= 3 and (ranges >= 1 or numbers >= 5)


def _build_ocr_image_variants(image_path):
  """Create fast OCR variants for blurry, small, or scanned reports.

  SPEED OPTIMISATION: Only 2 variants are generated by default (original +
  denoised grayscale). The adaptive-threshold and crop variants are added only
  when the image is a poor-quality scan (low resolution or very noisy), which
  keeps the common-case fast while still handling difficult scans well.
  """
  variants = []
  temp_paths = []

  img = cv2.imread(image_path)
  if img is None:
    return [image_path], temp_paths

  try:
    variants.append(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    scale = 1.0
    if max(h, w) < 1800:
      scale = 2.0
    if max(h, w) < 1000:
      scale = 3.0

    if scale != 1.0:
      gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Variant 1: denoised grayscale always included, good for printed reports.
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    temp_paths.append(_write_temp_ocr_image(denoised))

    # Quick quality check: only add expensive variants for difficult scans.
    noise_ratio = float(np.std(gray)) / max(float(np.mean(gray)), 1)
    is_difficult_scan = scale >= 2.0 or noise_ratio > 0.55

    if is_difficult_scan:
      # Variant 2: adaptive threshold, good for scanned/photographed paper.
      blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
      thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
      )
      temp_paths.append(_write_temp_ocr_image(thresh))

      # Variant 3: centre table crop for scanned lab reports with headers/footers.
      if h > 900 and w > 600:
        y1, y2 = int(h * 0.24), int(h * 0.78)
        x1, x2 = int(w * 0.02), int(w * 0.98)
        crop = gray[y1:y2, x1:x2]
        if crop.size > 0:
          crop = cv2.resize(crop, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
          crop = cv2.fastNlMeansDenoising(crop, None, 8, 7, 21)
          crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)
          temp_paths.append(_write_temp_ocr_image(crop))

    variants.extend(temp_paths)
  except Exception:
    for temp_path in temp_paths:
      if os.path.exists(temp_path):
        os.unlink(temp_path)
    return [image_path], []

  return variants, temp_paths


def _build_tesseract_ocr_variants(image_path):
  """Create OCR variants tuned for scanned/table-heavy lab reports.

  Tesseract keeps CBC table rows much better than EasyOCR for reports with
  dotted leaders such as "HAEMOGLOBIN .... 10.4 g/dl .... (11-14.5)".
  Full-page variants keep patient/header text, while centre-crop variants
  recover rows hidden by watermarks or footer noise.
  """
  variants = []
  temp_paths = []
  img = cv2.imread(image_path)
  if img is None:
    return [image_path], []
  try:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Enlarge only genuinely small uploads. Large A4 scans already OCR well
    # at original size; over-enlarging them can make Tesseract very slow.
    scale = 1.0
    longest = max(h, w)
    if longest < 900:
      scale = 2.5
    elif longest < 1500:
      scale = 1.6
    if scale != 1.0:
      gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
      h, w = gray.shape[:2]

    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    temp_paths.append(_write_temp_ocr_image(denoised))

    otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    temp_paths.append(_write_temp_ocr_image(otsu))

    # Lab reports usually keep result tables in the middle of the page.
    # This crop is especially important for Aga Khan style CBC reports with
    # large watermark text across the table.
    if h > 900 and w > 600:
      y1, y2 = int(h * 0.27), int(h * 0.79)
      x1, x2 = int(w * 0.02), int(w * 0.98)
      crop = denoised[y1:y2, x1:x2]
      if crop.size > 0:
        temp_paths.append(_write_temp_ocr_image(crop))
        crop_otsu = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        temp_paths.append(_write_temp_ocr_image(crop_otsu))

    variants.extend(temp_paths)
  except Exception:
    for temp_path in temp_paths:
      if os.path.exists(temp_path):
        os.unlink(temp_path)
    return [image_path], []
  return variants or [image_path], temp_paths



def _has_clean_cbc_rows(text):
  """Return True when OCR text contains enough same-line CBC rows.

  This protects the parser from merged OCR variants that can shift values from
  one CBC parameter to the next.
  """
  text = clean_text(text)
  if not text:
    return False
  row_patterns = [
    r"\bHA?EMOGLOBIN\b.*?\d+(?:\.\d+)?\s*(?:g/dl|gm/dl).*?\(?\d+(?:\.\d+)?\s*[-]\s*\d+(?:\.\d+)?\)?",
    r"\bHA?EMATOCRIT\b.*?\d+(?:\.\d+)?\s*%.*?\(?\d+(?:\.\d+)?\s*[-]\s*\d+(?:\.\d+)?\)?",
    r"\bR\s*\.?\s*B\s*\.?\s*C\.?\b.*?\d+(?:\.\d+)?.*?\(?\d+(?:\.\d+)?\s*[-]\s*\d+(?:\.\d+)?\)?",
    r"\bM\s*\.?\s*C\s*\.?\s*V\.?\b.*?\d+(?:\.\d+)?.*?\(?\d+(?:\.\d+)?\s*[-]\s*\d+(?:\.\d+)?\)?",
    r"\bM\s*\.?\s*C\s*\.?\s*H\.?\b.*?\d+(?:\.\d+)?.*?\(?\d+(?:\.\d+)?\s*[-]\s*\d+(?:\.\d+)?\)?",
    r"\bM\s*\.?\s*C\s*\.?\s*H\s*\.?\s*C\.?\b.*?\d+(?:\.\d+)?.*?\(?\d+(?:\.\d+)?\s*[-]\s*\d+(?:\.\d+)?\)?",
    r"\bW\s*\.?\s*B\s*\.?\s*C\.?\b.*?\d+(?:\.\d+)?.*?\(?\d+(?:\.\d+)?\s*[-]\s*\d+(?:\.\d+)?\)?",
    r"\bNEUTROPHILS?\b.*?\d+(?:\.\d+)?\s*%.*?\(?\d+(?:\.\d+)?\s*[-]\s*\d+(?:\.\d+)?\)?",
    r"\bLYMPHOCYTES?\b.*?\d+(?:\.\d+)?\s*%.*?\(?\d+(?:\.\d+)?\s*[-]\s*\d+(?:\.\d+)?\)?",
    r"\bPLATELETS?\b.*?\d+(?:\.\d+)?.*?\(?\d+(?:\.\d+)?\s*[-]\s*\d+(?:\.\d+)?\)?",
  ]
  hits = 0
  for line in text.splitlines():
    line = re.sub(r"\s+", " ", safe_str(line)).strip()
    for pat in row_patterns:
      if re.search(pat, line, flags=re.I):
        hits += 1
        break
  return hits >= 7

def extract_text_from_image_with_tesseract(image_path):
  """OCR image/PDF page with Tesseract when available.

  Returns an empty string when Tesseract/pytesseract is unavailable so the app
  can safely fall back to EasyOCR.
  """
  if pytesseract is None:
    return ""
  temp_paths = []
  try:
    variants, temp_paths = _build_tesseract_ocr_variants(image_path)
    candidates = []
    # PSM 4 preserves lab-table rows better; PSM 6/11 are kept as backups.
    configs = ["--oem 3 --psm 11", "--oem 3 --psm 4", "--oem 3 --psm 6"]
    for candidate_path in variants:
      for config in configs:
        try:
          text = pytesseract.image_to_string(candidate_path, config=config, timeout=25)
          text = clean_text(text)
          if text:
            candidates.append(text)
            # Early return when this OCR pass already preserves many
            # complete structured rows. This keeps demo/runtime faster
            # and still uses only values found in the uploaded report.
            try:
              if _local_structured_row_count_for_ocr_candidate(text) >= 14:
                return text
            except Exception:
              pass
        except RuntimeError:
          # Skip a slow/bad OCR variant instead of freezing the app.
          pass
        except Exception:
          pass
    if not candidates:
      return ""
    ranked = sorted(
      candidates,
      key=lambda cand: (
        _local_structured_row_count_for_ocr_candidate(cand),
        _dynamic_lab_row_score(cand),
        _ocr_candidate_score(cand),
      ),
      reverse=True,
    )

    # For table-heavy lab scans, prefer the OCR pass that allows the local
    # parser to recover the most rows. This avoids value shifting and does
    # not hardcode any report values.
    for cand in ranked:
      try:
        if _local_structured_row_count_for_ocr_candidate(cand) >= 8 or _has_clean_cbc_rows(cand):
          return clean_text(cand)
      except Exception:
        pass

    return _merge_text_blocks(*ranked[:3])
  finally:
    for temp_path in temp_paths:
      if temp_path and temp_path != image_path and os.path.exists(temp_path):
        os.unlink(temp_path)

def preprocess_image_for_ocr(image_path):
  """Create the main cleaned temporary image to improve OCR quality."""
  variants, temp_paths = _build_ocr_image_variants(image_path)
  # Keep backward compatibility with the older code path.
  if len(variants) > 1:
    return variants[1], True
  return image_path, False



def extract_text_from_image_with_vision_model(image_path):
  """Use the vision model as a document OCR fallback for image-based reports.

  Tesseract/EasyOCR are still used first because they are local. Vision OCR is
  used in high-accuracy mode for difficult screenshots/scans where row order is
  important.
  """
  if client is None or MEDIBUDDY_FAST_MODE:
    return ""
  if not image_path or not os.path.exists(image_path):
    return ""
  try:
    image_data_url = encode_image_to_data_url(image_path)
    response = client.chat.completions.create(
      model=VISION_MODEL,
      messages=[
        {"role": "system", "content": "You are a medical document OCR engine. Return plain text only. Do not explain."},
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": (
                "Read this medical document image and transcribe the visible report text. "
                "Preserve table rows as much as possible in this format: Test | Result | Unit | Reference Range. "
                "Do not invent missing values. If text is unreadable, write [unreadable]. Return plain text only."
              ),
            },
            {"type": "image_url", "image_url": {"url": image_data_url}},
          ],
        },
      ],
      temperature=0.0,
      max_tokens=2200,
    )
    return clean_text(response.choices[0].message.content)
  except Exception:
    return ""


def extract_text_from_image(image_path):
  """Extract text from image using Tesseract first, then EasyOCR fallback.

  Tesseract is better for scanned/table-heavy lab reports because it preserves
  line order and dotted leader rows. EasyOCR remains as a backup for images
  where Tesseract returns weak text.
  """
  try:
    tesseract_text = extract_text_from_image_with_tesseract(image_path)
    if _has_clean_cbc_rows(tesseract_text) or _ocr_candidate_score(tesseract_text) >= 80:
      return tesseract_text
  except Exception:
    tesseract_text = ""

  # In high-accuracy mode, let the vision model transcribe difficult report images.
  # This improves non-standard layouts and random viva/demo uploads.
  vision_text = extract_text_from_image_with_vision_model(image_path)
  if _ocr_candidate_score(vision_text) > max(25, _ocr_candidate_score(tesseract_text) * 0.9):
    return _merge_text_blocks(vision_text, tesseract_text)

  temp_paths = []
  try:
    reader = get_easyocr_reader()
    if reader is None:
      return tesseract_text or ""

    variants, temp_paths = _build_ocr_image_variants(image_path)
    candidates = []
    if tesseract_text:
      candidates.append(tesseract_text)

    for idx, candidate_path in enumerate(variants):
      # Row-based OCR: fast, preserves table structure.
      try:
        results = reader.readtext(candidate_path, detail=0, paragraph=False)
        candidate_text = clean_text("\n".join([safe_str(x) for x in results if safe_str(x).strip()]))
        if candidate_text:
          candidates.append(candidate_text)
      except Exception:
        pass

      # Paragraph mode only on the original image (idx==0) to catch wrapped narrative text.
      if idx == 0:
        try:
          results = reader.readtext(candidate_path, detail=0, paragraph=True)
          candidate_text = clean_text("\n".join([safe_str(x) for x in results if safe_str(x).strip()]))
          if candidate_text:
            candidates.append(candidate_text)
        except Exception:
          pass

    if not candidates:
      return ""

    ranked = sorted(candidates, key=_ocr_candidate_score, reverse=True)
    return _merge_text_blocks(*ranked[:3])
  except Exception as e:
    return f"OCR Error: {e}"
  finally:
    for temp_path in temp_paths:
      if temp_path and temp_path != image_path and os.path.exists(temp_path):
        os.unlink(temp_path)

def extract_text_from_pdf(pdf_path):
  """Extract PDF text with hybrid native-text + OCR fallback per page.

  Some scanned PDFs contain tiny embedded header text. The old logic returned
  that native text immediately and never OCR'd the scanned lab table. This
  hybrid approach OCRs pages when embedded text is too short or does not look
  strong enough for lab parsing.
  """
  doc = None
  try:
    native_blocks = []
    ocr_blocks = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
      page = doc.load_page(page_num)
      page_text = page.get_text("text").strip()
      if page_text:
        native_blocks.append(page_text)

      if not _looks_like_strong_page_text(page_text):
        temp_img_path = None
        try:
          # Higher resolution improves small lab-table values.
          pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
          temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
          temp_img_path = temp_img.name
          temp_img.close()
          pix.save(temp_img_path)

          page_ocr = extract_text_from_image(temp_img_path)
          if page_ocr and not safe_str(page_ocr).startswith("OCR Error"):
            ocr_blocks.append(page_ocr)
        finally:
          if temp_img_path and os.path.exists(temp_img_path):
            os.unlink(temp_img_path)

    native_text = clean_text("\n".join(native_blocks))
    ocr_text = clean_text("\n".join(ocr_blocks))

    if not native_text and not ocr_text:
      return ""

    # Use both when OCR adds meaningful table/numeric content.
    if _ocr_candidate_score(ocr_text) > (_ocr_candidate_score(native_text) * 0.65):
      return _merge_text_blocks(native_text, ocr_text)

    return clean_text(native_text or ocr_text)
  except Exception as e:
    return f"PDF Extraction Error: {e}"
  finally:
    if doc is not None:
      try:
        doc.close()
      except Exception:
        pass

def detect_report_type(text, file_path=None):
  t = safe_str(text).lower()
  ext = os.path.splitext(file_path)[1].lower() if file_path else ""

  structured_lab_rows = len(parse_lab_records(text))
  lab_hits = sum(1 for k in LAB_KEYWORDS if re.search(rf"\b{re.escape(k)}\b", t))
  radiology_hits = sum(1 for k in RADIOLOGY_KEYWORDS if re.search(rf"\b{re.escape(k)}\b", t))
  narrative_hits = sum(1 for k in ["diagnosis", "history", "medication", "advice", "recommendation", "clinical", "complaint", "summary", "notes"] if re.search(rf"\b{re.escape(k)}\b", t))
  range_hits = len(re.findall(r"\d+(?:\.\d+)?\s*[-]\s*\d+(?:\.\d+)?", t))

  if ext in [".png", ".jpg", ".jpeg"] and any(re.search(rf"\b{re.escape(k)}\b", t) for k in XRAY_KEYWORDS):
    return {
      "category": "X-ray Image",
      "subtype": "Image-Based X-ray",
      "reason": "Image file and X-ray keywords detected"
    }

  if structured_lab_rows >= 2 or (structured_lab_rows >= 1 and (lab_hits >= 1 or range_hits >= 2)) or lab_hits >= 3:
    subtype = "General Laboratory Report"
    if any(re.search(rf"\b{re.escape(k)}\b", t) for k in ["wbc", "rbc", "hemoglobin", "haemoglobin", "platelet", "cbc"]):
      subtype = "CBC / Hematology Report"
    elif any(re.search(rf"\b{re.escape(k)}\b", t) for k in ["glucose", "hba1c", "cholesterol", "triglycerides", "hdl", "ldl"]):
      subtype = "Biochemistry / Diabetes / Lipid Report"
    elif any(re.search(rf"\b{re.escape(k)}\b", t) for k in ["creatinine", "urea", "bun", "uric acid"]):
      subtype = "Renal Function Report"
    elif any(re.search(rf"\b{re.escape(k)}\b", t) for k in ["alt", "ast", "bilirubin", "alkaline phosphatase"]):
      subtype = "Liver Function Report"
    elif any(re.search(rf"\b{re.escape(k)}\b", t) for k in ["tsh", "t3", "t4", "thyroid"]):
      subtype = "Thyroid / Hormone Report"
    return {
      "category": "Laboratory Report",
      "subtype": subtype,
      "reason": f"Structured lab rows: {structured_lab_rows}; lab keywords: {lab_hits}; ranges: {range_hits}"
    }

  if radiology_hits >= 2 and structured_lab_rows == 0:
    subtype = "General Radiology Report"
    if re.search(r"\bct\b", t):
      subtype = "CT Report"
    elif re.search(r"\bmri\b", t):
      subtype = "MRI Report"
    elif re.search(r"\b(?:ultrasound|sonography)\b", t):
      subtype = "Ultrasound Report"
    elif re.search(r"\b(?:x ray|x-ray|radiograph)\b", t):
      subtype = "X-ray Report"
    return {
      "category": "Radiology Report",
      "subtype": subtype,
      "reason": f"Radiology-style keywords detected: {radiology_hits}"
    }

  subtype = "General Clinical Report"
  if re.search(r"\bprescription\b|\brx\b", t):
    subtype = "Prescription / Medication Report"
  elif re.search(r"\bdischarge\b", t):
    subtype = "Discharge Summary"
  elif re.search(r"\b(?:histopathology|biopsy)\b", t):
    subtype = "Pathology / Biopsy Report"
  elif re.search(r"\b(?:ecg|ekg)\b", t):
    subtype = "Cardiac Test Report"
  elif narrative_hits == 0 and len(t.split()) < 80:
    subtype = "Unclassified Medical Document"

  return {
    "category": "General Medical Report",
    "subtype": subtype,
    "reason": "General medical report detected; no strong structured lab table found"
  }

def extract_patient_info(text):
  info = {}
  patterns = {
    "Patient Name": [
      r"(?:Patient Name|Name)\s*[:\-]?\s*([A-Za-z][A-Za-z .]{1,60})",
    ],
    "Age": [
      r"(?:Age)\s*[:\-]?\s*(\d{1,3})",
      r"(\d{1,3})\s*(?:years|yrs|y/o)"
    ],
    "Gender": [
      r"(?:Gender|Sex)\s*[:\-]?\s*(Male|Female|M|F)",
    ],
    "Date": [
      r"(?:Date|Reported On|Report Date|Collection Date)\s*[:\-]?\s*([A-Za-z0-9\/\-.]+)",
    ],
    "Report ID": [
      r"(?:Report ID|Lab No|Sample ID|Accession)\s*[:\-]?\s*([A-Za-z0-9\-_\/]+)",
    ],
  }
  for key, pattern_list in patterns.items():
    value = "Not Found"
    for pattern in pattern_list:
      match = re.search(pattern, text, flags=re.IGNORECASE)
      if match:
        value = match.group(1).strip()
        break
    info[key] = value
  return info

def format_patient_info_md(info):
  lines = ["**Patient Information**", ""]
  for key in ["Patient Name", "Age", "Gender", "Date", "Report ID"]:
    lines.append(f"- **{key}:** {safe_str(info.get(key, 'Not Found'))}")
  return "\n".join(lines)

def normalize_param_name(name):
  return re.sub(r"\s+", " ", safe_str(name)).strip(" :-").title()


def looks_numeric_token(token):
  token = safe_str(token).strip().replace(",", "")
  return bool(re.fullmatch(r"[<>]?\d+(?:\.\d+)?", token))


def looks_unit_token(token):
  token = safe_str(token).strip()
  if not token:
    return False
  return bool(re.fullmatch(r"[A-Za-z%/\^xX0-9\.-]+", token))


def is_noise_or_heading(line):
  line_l = safe_str(line).lower().strip()
  if not line_l:
    return True
  bad_starts = [
    "reference range", "normal range", "parameter", "value", "status", "visual",
    "name", "patient", "doctor", "specimen", "collection", "received", "reported",
    "ai medical report", "summary", "interpretation", "remarks", "comment"
  ]
  if any(line_l.startswith(x) for x in bad_starts):
    return True
  if len(line_l.split()) <= 2 and not re.search(r"\d", line_l):
    return True
  return False


def normalize_range_text(ref_text):
  ref_text = safe_str(ref_text).replace("", "-").replace("", "-").replace("", "-")
  ref_text = ref_text.replace(",", "")
  ref_text = re.sub(r"\s*(?:to|||)\s*", "-", ref_text, flags=re.IGNORECASE)
  ref_text = re.sub(r"\s+", "", ref_text)
  return ref_text


def clean_value_text(value_text):
  value_text = safe_str(value_text).strip()
  value_text = value_text.replace(",", "")
  value_text = re.sub(r"\s+", "", value_text)
  return value_text


def build_candidate_lines(text):
  raw_lines = [re.sub(r"\s+", " ", ln).strip() for ln in safe_str(text).splitlines() if safe_str(ln).strip()]
  candidates = []
  for i, line in enumerate(raw_lines):
    candidates.append(line)
    if i + 1 < len(raw_lines):
      nxt = raw_lines[i + 1]
      if re.search(r"[A-Za-z]", line) and (re.search(r"\d", nxt) or re.search(r"\d+(?:\.\d+)?\s*[-]\s*\d+(?:\.\d+)?", nxt)):
        candidates.append(f"{line} {nxt}")
  seen = set()
  ordered = []
  for line in candidates:
    key = line.lower()
    if key not in seen:
      seen.add(key)
      ordered.append(line)
  return ordered


def parse_lab_records_with_llm(text):
  """Universal AI-based lab parser.

  This is used as the high-accuracy parser for mixed lab report formats
  (CBC, LFT, KFT/RFT, thyroid, lipid, urine chemistry, cardiac enzymes,
  diabetes tests, etc.). It is intentionally strict: uncertain rows are
  skipped instead of creating a wrong structured table.
  """
  if client is None:
    return []

  prompt = f"""
You are a strict medical laboratory report parser.

Task: Extract ONLY structured laboratory result rows from the OCR/native text.
The report may be CBC, LFT, KFT/RFT, lipid profile, thyroid, urine lab report,
diabetes report, cardiac enzymes, electrolytes, or another lab table.

Return JSON only as a list of objects with exactly these keys:
Parameter, Value, Unit, Reference Range

Very important safety rules:
- Extract only rows where the patient value and reference range clearly belong to the same parameter row.
- Do not shift values from one row to another.
- Do not guess missing values.
- Do not use example values, default values, memory, or hardcoded values.
- Every Value and Reference Range must come from the OCR/native text given below.
- Do not include a row if alignment is unclear.
- Do not include units as parameters. Examples of invalid parameters: g/dL, gm/dL, Mil/Cmm, Cu Micron, Picograms, fL, %, mg/dL, Unit, Result, Normal Range.
- Parameter must be the medical test/analyte name, e.g. Hemoglobin, RBC, WBC, Platelets, Glucose, Creatinine, ALT, AST, Bilirubin, TSH, Cholesterol, Troponin.
- Value must contain only the patient result number or comparator, e.g. 10.4, >200, <0.01.
- Unit must contain only the unit, e.g. g/dL, %, mg/dL, mmol/L, U/L, x10^9/L. Leave blank only if no unit is visible.
- Reference Range must be a numeric low-high range if visible, e.g. 11-14.5. If no numeric reference range is visible, skip the row.
- For narrative reports such as X-ray, CT, MRI, prescription, discharge summary, or handwritten note, return [].
- Return [] if fewer than two reliable lab rows are visible.

OCR/native text:
{safe_str(text)[:8000]}
"""
  try:
    response = client.chat.completions.create(
      model=TEXT_MODEL,
      messages=[
        {"role": "system", "content": "You are a strict medical lab table extraction engine. Return valid JSON only. Never invent or guess values."},
        {"role": "user", "content": prompt},
      ],
      temperature=0.0,
      max_tokens=1800,
    )
    content = safe_str(response.choices[0].message.content).strip()
    m = re.search(r"\[.*\]", content, flags=re.S)
    if m:
      content = m.group(0)
    data = json.loads(content)
    records = []
    if isinstance(data, list):
      for item in data:
        if not isinstance(item, dict):
          continue
        rec = normalize_structured_record(
          item.get("Parameter", ""),
          item.get("Value", ""),
          item.get("Unit", ""),
          item.get("Reference Range", ""),
        )
        if rec and is_plausible_lab_record(rec):
          records.append(rec)
    return records
  except Exception:
    return []


UNIT_ONLY_PARAMETER_NAMES = {
  "g/dl", "gm/dl", "mg/dl", "mmol/l", "u/l", "iu/l", "miu/l", "ng/ml", "pg/ml",
  "mil/cmm", "cu micron", "picograms", "fl", "%", "ratio", "unit", "units", "result",
  "normal range", "reference range", "test", "parameter", "value", "visual", "status"
}


def is_plausible_lab_record(record):
  """Final guardrail for extracted rows.

  It blocks OCR/parser mistakes such as using units as parameters or assigning
  Hemoglobin units to RBC/WBC rows. The goal is not to hide valid data; it is to
  prevent unsafe wrong matches in the structured table.
  """
  if not isinstance(record, dict):
    return False

  param = clean_parameter_name(record.get("Parameter", ""))
  value = clean_value_text(record.get("Value", ""))
  unit = safe_str(record.get("Unit", "")).strip()
  ref = normalize_range_text(record.get("Reference Range", ""))
  p = param.lower().strip()
  u = unit.lower().replace(" ", "")

  if not p or len(re.findall(r"[a-z]", p)) < 2:
    return False
  if p in UNIT_ONLY_PARAMETER_NAMES:
    return False
  if re.fullmatch(r"[/%a-z0-9\^\.]+", p) and any(ch in p for ch in ["/", "%", "^", "10"]):
    return False
  if any(x == p for x in ["g dl", "gm dl", "mg dl", "mil cmm", "cu micron", "picograms"]):
    return False
  if not re.fullmatch(r"[<>]?\d+(?:\.\d+)?", value):
    return False
  if not re.fullmatch(r"\d+(?:\.\d+)?-\d+(?:\.\d+)?", ref):
    return False

  # Common unit sanity checks for CBC-style rows. These block shifted values.
  compact = re.sub(r"[^a-z0-9]", "", p)
  if compact in {"rbc", "redbloodcell", "redbloodcells", "wbc", "whitebloodcell", "whitebloodcells", "platelet", "platelets", "plt"}:
    if "g/dl" in u or "gmdl" in u or "mg/dl" in u:
      return False
  if compact in {"neutrophils", "lymphocytes", "eosinophils", "monocytes", "basophils"}:
    if u and "%" not in u and "ratio" not in u:
      return False
  if compact in {"hemoglobin", "haemoglobin", "hb", "hgb"}:
    if u and not any(x in u for x in ["g/dl", "gm/dl", "gdl"]):
      return False
  if compact in {"hematocrit", "haematocrit", "hct", "pcv"}:
    if u and "%" not in u:
      return False

  return True


def parse_noisy_cbc_records(text):
  """Recover CBC rows from noisy scanned OCR.

  Example supported OCR row:
  HAEMOGLOBIN Peery orca or 10.4 g/dl ATES (11-14.5)
  M.C.H.C. seseeneeesees 29.4 g/dL covssenneneane (30.3-34.4)
  """
  text = safe_str(text)
  if not text.strip():
    return []

  alias_patterns = [
    ("Hemoglobin", r"HA?EMOGLOBIN|\bHGB\b|\bHB\b"),
    ("Hematocrit", r"HA?EMATOCRIT|\bHCT\b|\bPCV\b"),
    ("RBC", r"R\s*\.?\s*B\s*\.?\s*C\.?|\bRBC\b"),
    ("MCV", r"M\s*\.?\s*C\s*\.?\s*V\.?|\bMCV\b"),
    ("MCHC", r"M\s*\.?\s*C\s*\.?\s*H\s*\.?\s*C\.?|\bMCHC\b"),
    ("MCH", r"M\s*\.?\s*C\s*\.?\s*H\.?|\bMCH\b"),
    ("RDW", r"R\s*\.?\s*D\s*\.?\s*W\.?|\bRDW\b"),
    ("WBC", r"W\s*\.?\s*B\s*\.?\s*C\.?|\bWBC\b"),
    ("Neutrophils", r"NEUTROPHILS?|\bNEUT\b"),
    ("Lymphocytes", r"LYMPHOCYTES?|\bLYMPH\b"),
    ("Eosinophils", r"EOSINOPHILS?|\bEOS\b"),
    ("Monocytes", r"MONOCYTES?|\bMONO\b"),
    ("Basophils", r"BASOPHILS?|\bBASO\b"),
    ("NLR", r"NEUTROPHILS?\s+LYMPHOCYTES?\s+RATIO\s*\(?\s*NLR\s*\)?|\bNLR\b"),
    ("Platelets", r"PLATELETS?|\bPLT\b"),
  ]

  unit_pattern = r"(?:x\s*10\s*E?\s*\d+\s*/?\s*[A-Za-z]*|x10E?\d+/?[A-Za-z]*|x10\^?\d+/?[A-Za-z]*|10\^?\d+/?[A-Za-z]*|g\s*/\s*dL|gm\s*/\s*dL|mg\s*/\s*dL|mmol\s*/\s*L|IU\s*/\s*L|U\s*/\s*L|fL|pg|%|ratio)"
  number_pattern = r"[<>]?\d+(?:,\d{3})*(?:\.\d+)?"
  range_pattern = rf"\(?\s*(?P<low>\d+(?:\.\d+)?)\s*(?:-|||to)\s*(?P<high>\d+(?:\.\d+)?)\s*\)?"

  def normalize_line(line):
    line = safe_str(line)
    line = line.replace("", "-").replace("", "-").replace("", "-")
    line = re.sub(r"\s+", " ", line).strip()
    return line

  lines = [normalize_line(x) for x in text.splitlines() if safe_str(x).strip()]
  candidates = []
  for i in range(len(lines)):
    for size in (1, 2, 3):
      if i + size <= len(lines):
        chunk = normalize_line(" ".join(lines[i:i + size]))
        if re.search(r"\d", chunk) and any(re.search(pat, chunk, flags=re.I) for _, pat in alias_patterns):
          candidates.append(chunk)

  found = {}
  for label, alias_pat in alias_patterns:
    best = None
    for chunk in candidates:
      m_label = re.search(alias_pat, chunk, flags=re.I)
      if not m_label:
        continue

      # Work only from this label onward and stop before the next CBC label if a joined OCR row exists.
      segment = chunk[m_label.start():]
      next_starts = []
      for other_label, other_pat in alias_patterns:
        if other_label == label:
          continue
        m_other = re.search(other_pat, segment[len(m_label.group(0)):], flags=re.I)
        if m_other:
          next_starts.append(len(m_label.group(0)) + m_other.start())
      if next_starts:
        segment = segment[:min(next_starts)]

      ranges = list(re.finditer(range_pattern, segment, flags=re.I))
      if not ranges:
        continue
      range_match = ranges[-1]
      before_range = segment[:range_match.start()]

      numbers = []
      for nm in re.finditer(number_pattern, before_range):
        left = before_range[nm.start() - 1:nm.start()].lower() if nm.start() > 0 else ""
        right = before_range[nm.end():nm.end() + 1].lower()
        if left in {"x", "e", "^"} or right in {"e", "^"}:
          continue
        numbers.append(nm)
      if not numbers:
        continue
      value = clean_value_text(numbers[0].group(0))
      unit_match = re.search(unit_pattern, before_range[numbers[0].end():], flags=re.I)
      unit = unit_match.group(0) if unit_match else ""
      unit = re.sub(r"\s+", "", unit)
      unit = unit.replace("x10E", "x10^").replace("x10e", "x10^").replace("g/dl", "g/dL").replace("g/dL", "g/dL")
      rec = normalize_structured_record(label, value, unit, f"{range_match.group('low')}-{range_match.group('high')}")
      if rec:
        score = 0
        if rec.get("Unit"):
          score += 2
        if "." in rec.get("Value", ""):
          score += 1
        # Prefer standalone rows over joined chunks.
        score -= max(0, len(segment.split()) - 10) * 0.05
        if best is None or score > best[0]:
          best = (score, rec)
    if best:
      found[label] = best[1]

  return list(found.values())


def parse_lab_records(text):
  """Parse structured lab rows from OCR/native text.

  This parser is intentionally tolerant because real lab reports vary:
  - value/unit/range can appear in different order
  - ranges may use hyphen, en dash, or "to"
  - OCR may split one row across two or three lines
  - status words such as High/Low may appear after the range
  """
  records = []
  ai_records = []

  # High-accuracy mode: with API available, use the universal AI parser,
  # but DO NOT return immediately. We combine AI rows with local/OCR rows
  # so rows such as R.B.C., W.B.C. and PLATELETS are not missed.
  # Actual values still come only from the uploaded report text.
  if client is not None and not MEDIBUDDY_FAST_MODE:
    ai_records = parse_lab_records_with_llm(text)
    if ai_records:
      records.extend(ai_records)

  def fix_lab_ocr_line(line):
    line = safe_str(line)
    line = line.replace("", "-").replace("", "-").replace("", "-")
    line = line.replace("", '"').replace("", '"').replace("", "'")
    line = re.sub(r"[|]+", " ", line)

    # Common OCR unit mistakes.
    replacements = {
      "mgldl": "mg/dL",
      "mg/dl": "mg/dL",
      "gldl": "g/dL",
      "gm/dl": "g/dL",
      "mmolll": "mmol/L",
      "mmol/l": "mmol/L",
      "iu/l": "IU/L",
      "u/l": "U/L",
      "x10^9/l": "x10^9/L",
      "x10 9/l": "x10^9/L",
      "x 10 9/l": "x10^9/L",
      "10^3/ul": "10^3/uL",
      "10*3/ul": "10^3/uL",
      "ul": "uL",
    }
    for wrong, right in replacements.items():
      line = re.sub(rf"\b{re.escape(wrong)}\b", right, line, flags=re.IGNORECASE)

    # Fix OCR O/I/l mistakes only inside numbers.
    line = re.sub(r"(?<=\d)[Oo](?=\d)", "0", line)
    line = re.sub(r"(?<=\d)[Il](?=\d)", "1", line)

    line = re.sub(r"\s+", " ", line).strip(" |:\t")
    return line

  def candidate_lines_for_lab(text):
    base_lines = [fix_lab_ocr_line(x) for x in build_candidate_lines(text)]
    raw_lines = [
      fix_lab_ocr_line(ln)
      for ln in safe_str(text).splitlines()
      if safe_str(ln).strip()
    ]

    # Rebuild rows that OCR split as:
    # Hemoglobin
    # 10.2 g/dL
    # 12.0 - 15.0
    extra = []
    for i in range(len(raw_lines)):
      window = raw_lines[i:i + 4]
      for size in (2, 3, 4):
        if len(window) >= size:
          joined = fix_lab_ocr_line(" ".join(window[:size]))
          if re.search(r"[A-Za-z]", joined) and len(re.findall(r"\d", joined)) >= 2:
            extra.append(joined)

    ordered = []
    seen = set()
    for line in base_lines + extra:
      key = re.sub(r"[^a-z0-9]+", "", line.lower())
      if key and key not in seen:
        seen.add(key)
        ordered.append(line)
    return ordered

  num = r"[<>]?\d+(?:,\d{3})*(?:\.\d+)?"
  ref_num = r"\d+(?:,\d{3})*(?:\.\d+)?"
  unit = r"[A-Za-z%/^xX0-9\.\-\*]+"
  sep = r"(?:-|to)"
  trailing_status = r"(?:\s*(?:H|L|N|High|Low|Normal|Abnormal|Flag|Result))?"

  explicit_patterns = [
    # Hemoglobin 10.2 g/dL 12.0-15.0
    re.compile(
      rf"^(?P<param>[A-Za-z][A-Za-z0-9\s/\-()%,+\.]{{1,80}}?)\s+"
      rf"(?P<value>{num})\s*"
      rf"(?P<unit>{unit})?\s+"
      rf"(?P<low>{ref_num})\s*{sep}\s*(?P<high>{ref_num}){trailing_status}\s*$",
      flags=re.IGNORECASE,
    ),
    # Hemoglobin: 10.2 g/dL (12.0 - 15.0)
    re.compile(
      rf"^(?P<param>[A-Za-z][A-Za-z0-9\s/\-()%,+\.]{{1,80}}?)\s*[:=]?\s*"
      rf"(?P<value>{num})\s*"
      rf"(?P<unit>{unit})?\s*"
      rf"\(?\s*(?P<low>{ref_num})\s*{sep}\s*(?P<high>{ref_num})\s*\)?{trailing_status}\s*$",
      flags=re.IGNORECASE,
    ),
    # Hemoglobin 10.2 12.0-15.0 g/dL
    re.compile(
      rf"^(?P<param>[A-Za-z][A-Za-z0-9\s/\-()%,+\.]{{1,80}}?)\s+"
      rf"(?P<value>{num})\s+"
      rf"(?P<low>{ref_num})\s*{sep}\s*(?P<high>{ref_num})\s*"
      rf"(?P<unit>{unit})?{trailing_status}\s*$",
      flags=re.IGNORECASE,
    ),
    # Hemoglobin g/dL 10.2 12.0-15.0
    re.compile(
      rf"^(?P<param>[A-Za-z][A-Za-z0-9\s/\-()%,+\.]{{1,80}}?)\s+"
      rf"(?P<unit>{unit})\s+"
      rf"(?P<value>{num})\s+"
      rf"(?P<low>{ref_num})\s*{sep}\s*(?P<high>{ref_num}){trailing_status}\s*$",
      flags=re.IGNORECASE,
    ),
    # Hemoglobin 10.2 g/dL 12.0 15.0 Low
    re.compile(
      rf"^(?P<param>[A-Za-z][A-Za-z0-9\s/\-()%,+\.]{{1,80}}?)\s+"
      rf"(?P<value>{num})\s+"
      rf"(?P<unit>{unit})?\s*"
      rf"(?:H|L|N|High|Low|Normal)?\s*"
      rf"(?P<low>{ref_num})\s+(?P<high>{ref_num}){trailing_status}\s*$",
      flags=re.IGNORECASE,
    ),
  ]

  sentence_patterns = [
    re.compile(
      rf"(?P<param>[A-Za-z][A-Za-z0-9\s/\-()%,+\.]{{1,60}}?)\s*(?:is|:|=)?\s*"
      rf"(?P<value>{num})\s*(?P<unit>{unit})?"
      rf".*?(?:normal|reference)\s*range\s*(?:of|:|=)?\s*"
      rf"(?P<low>{ref_num})\s*{sep}\s*(?P<high>{ref_num})",
      flags=re.IGNORECASE,
    ),
    re.compile(
      rf"(?P<param>[A-Za-z][A-Za-z0-9\s/\-()%,+\.]{{1,60}}?)\s*(?:result|value)?\s*[:=]?\s*"
      rf"(?P<value>{num})\s*(?P<unit>{unit})?"
      rf".*?\b(?:range|normal|reference)\b\s*[:=]?\s*"
      rf"(?P<low>{ref_num})\s*{sep}\s*(?P<high>{ref_num})",
      flags=re.IGNORECASE,
    ),
  ]

  bad_labels = {
    "Reference Range", "Normal Range", "Range", "Value", "Status", "Result",
    "Parameter", "Parameter Value", "Visual", "Test", "Tests", "Unit", "Flag"
  }

  known_lab_labels = {
    "Hemoglobin", "Hematocrit", "RBC", "MCV", "MCH", "MCHC", "RDW",
    "WBC", "Neutrophils", "Lymphocytes", "Eosinophils", "Monocytes",
    "Basophils", "NLR", "Platelets", "Glucose", "Creatinine", "HbA1c"
  }

  def parse_known_noisy_lab_line(line):
    """Handle scanned-report OCR rows such as:
    HAEMOGLOBIN noisy words 10.4 g/dl noisy words (11-14.5)
    R.B.C. noisy words 4.50 x10E12/L noisy words (3.61-5.2)
    This prevents the generic regex from accidentally using digits from the reference range as the result.
    """
    cleaned_name = clean_parameter_name(line)
    if cleaned_name not in known_lab_labels:
      return None
    # Use the first reference range in the row. OCR candidate rebuilding may join
    # two neighbouring rows; using the first range keeps the current row correct.
    range_match = re.search(
      rf"(?<![\d.])\(?\s*(?P<low>{ref_num})\s*{sep}\s*(?P<high>{ref_num})\s*\)?",
      line,
      flags=re.IGNORECASE,
    )
    if not range_match:
      return None
    before_range = line[:range_match.start()]

    # Ignore numbers embedded in scientific-style units such as x10E12/L or 10^9/L.
    # Do NOT reject a value just because the previous word contains the letter "e"
    # (for example OCR noise like "rere 35.4" in this Aga Khan report).
    number_matches = []
    for m in re.finditer(num, before_range):
      left_char = before_range[m.start() - 1:m.start()].lower() if m.start() > 0 else ""
      right_char = before_range[m.end():m.end() + 1].lower()
      left_context = before_range[max(0, m.start() - 4):m.start()].lower()
      right_context = before_range[m.end():m.end() + 4].lower()
      embedded_in_unit = (
        left_char in {"x", "e", "^"}
        or right_char in {"e", "^"}
        or re.search(r"x\s*$", left_context) is not None
        or re.search(r"^\s*[e^]", right_context) is not None
      )
      if embedded_in_unit:
        continue
      number_matches.append(m)
    if not number_matches:
      return None

    # For a known lab row, the first meaningful number after the label is usually the patient value.
    value_match = number_matches[0]
    after_value = before_range[value_match.end():]
    unit_match = re.search(r"(x\s*10\s*e?\d+/?[A-Za-z]*|x10E?\d+/?[A-Za-z]*|10\^?\d+/?[A-Za-z]*|g/dl|gm/dl|mg/dl|mmol/l|iu/l|u/l|fl|pg|%|ratio)", after_value, flags=re.IGNORECASE)
    unit_text = unit_match.group(1) if unit_match else ""
    unit_text = unit_text.replace(" ", "").replace("x10E", "x10^").replace("x10e", "x10^")
    return {
      "param": cleaned_name,
      "value": clean_value_text(value_match.group(0)),
      "unit": unit_text,
      "ref": f"{range_match.group('low')}-{range_match.group('high')}"
    }

  for raw_line in candidate_lines_for_lab(text):
    line = fix_lab_ocr_line(raw_line)
    if len(line) < 4 or is_noise_or_heading(line):
      continue
    if not re.search(r"\d", line):
      continue

    matched = parse_known_noisy_lab_line(line)

    for pattern in explicit_patterns if matched is None else []:
      m = pattern.search(line)
      if m:
        matched = {
          "param": safe_str(m.group("param")).strip(),
          "value": clean_value_text(m.group("value")),
          "unit": safe_str(m.groupdict().get("unit", "") or "").strip(),
          "ref": f"{m.group('low')}-{m.group('high')}"
        }
        break

    if matched is None:
      for pattern in sentence_patterns:
        m = pattern.search(line)
        if m:
          matched = {
            "param": safe_str(m.group("param")).strip(),
            "value": clean_value_text(m.group("value")),
            "unit": safe_str(m.groupdict().get("unit", "") or "").strip(),
            "ref": f"{m.group('low')}-{m.group('high')}"
          }
          break

    if matched is None:
      # Last-resort: row ends with a range, optional status after range.
      range_match = re.search(
        rf"(?P<low>{ref_num})\s*{sep}\s*(?P<high>{ref_num})(?:\s*(?:H|L|N|High|Low|Normal|Abnormal))?\s*$",
        line,
        flags=re.IGNORECASE,
      )
      if range_match:
        left = line[:range_match.start()].strip()
        value_matches = list(re.finditer(num, left))
        if value_matches:
          value_match = value_matches[-1]
          param = left[:value_match.start()].strip(" :-=")
          unit_text = left[value_match.end():].strip(" :-()")
          matched = {
            "param": param,
            "value": clean_value_text(value_match.group(0)),
            "unit": unit_text,
            "ref": f"{range_match.group('low')}-{range_match.group('high')}"
          }

    if matched is None:
      # Token fallback: Parameter Value Unit Low High
      tokens = line.split()
      numeric_positions = [i for i, tok in enumerate(tokens) if looks_numeric_token(tok.replace(",", ""))]
      if len(numeric_positions) >= 3:
        for j in range(len(numeric_positions) - 2):
          value_pos, low_pos, high_pos = numeric_positions[j], numeric_positions[j + 1], numeric_positions[j + 2]
          if value_pos < low_pos < high_pos and low_pos - value_pos <= 4:
            param = " ".join(tokens[:value_pos]).strip(" :-=")
            unit_text = " ".join(tok for tok in tokens[value_pos + 1:low_pos] if looks_unit_token(tok)).strip()
            matched = {
              "param": param,
              "value": clean_value_text(tokens[value_pos]),
              "unit": unit_text,
              "ref": f"{tokens[low_pos]}-{tokens[high_pos]}"
            }
            break

    if not matched:
      continue

    param = safe_str(matched["param"]).strip()
    cleaned_param = clean_parameter_name(param)
    if not param or cleaned_param in bad_labels:
      continue

    rec = normalize_structured_record(
      param,
      matched["value"],
      matched["unit"],
      matched["ref"],
    )
    if rec:
      records.append(rec)

  # Extra safety net for scanned CBC reports where OCR creates noisy filler words
  # between parameter, result and reference range. Always evaluate this for CBC
  # reports, because a generic parser can produce enough rows but with shifted
  # values after OCR merges table columns. Prefer the CBC-specific result when
  # it finds many known CBC rows.
  fallback_records = parse_noisy_cbc_records(text)
  if len(fallback_records) >= 8:
    records = fallback_records
  elif len(fallback_records) > len(records):
    records = fallback_records

  if len(records) < 2 and client is not None and not MEDIBUDDY_FAST_MODE:
    llm_records = parse_lab_records_with_llm(text)
    if len(llm_records) > len(records):
      records = llm_records

  # Prefer the best row per parameter name, which avoids fake duplicates created by broken OCR joins.
  best_by_param = {}
  param_order = []
  for r in records:
    param_key = r["Parameter"].lower()
    score = 0
    if safe_str(r.get("Unit", "")).strip():
      score += 2
    if "." in safe_str(r.get("Value", "")) or len(safe_str(r.get("Value", ""))) > 1:
      score += 1
    if re.fullmatch(r"\d+(?:\.\d+)?-\d+(?:\.\d+)?", safe_str(r.get("Reference Range", ""))):
      score += 1

    # Prefer known lab parameter names over generic OCR fragments.
    if any(re.search(pattern, r["Parameter"].lower()) for _, _, patterns in LAB_DISPLAY_PATTERNS for pattern in patterns):
      score += 2

    if param_key not in best_by_param:
      best_by_param[param_key] = (score, r)
      param_order.append(param_key)
    elif score > best_by_param[param_key][0]:
      best_by_param[param_key] = (score, r)

  unique_records = [best_by_param[k][1] for k in param_order]
  unique_records = [r for r in unique_records if is_plausible_lab_record(r)]
  return unique_records

def evaluate_record(record):
  value_raw = safe_str(record.get("Value", "")).strip()
  comparison = None
  if value_raw.startswith("<"):
    comparison = "<"
  elif value_raw.startswith(">"):
    comparison = ">"

  value_str = value_raw.replace("<", "").replace(">", "").strip()
  ref = safe_str(record.get("Reference Range", "")).replace("", "-")
  unit = safe_str(record.get("Unit", ""))

  status = "Unknown"
  severity = "Unknown"
  pct_in_range = None

  try:
    value = float(value_str)
    parts = [p.strip() for p in ref.split("-")]
    if len(parts) == 2:
      low = float(parts[0])
      high = float(parts[1])

      if comparison == "<":
        status = "Low" if value <= low else "Normal"
      elif comparison == ">":
        status = "High" if value >= high else "Normal"
      elif low <= value <= high:
        status = "Normal"
      elif value < low:
        status = "Low"
      elif value > high:
        status = "High"

      if status == "Normal":
        severity = "Normal"
        span = high - low if high != low else 1
        pct_in_range = round(((value - low) / span) * 100, 2)
      elif status == "Low":
        diff = max(low - value, 0)
        if diff <= (abs(low) * 0.10 if low != 0 else 1):
          severity = "Mild Low"
        elif diff <= (abs(low) * 0.25 if low != 0 else 2):
          severity = "Moderate Low"
        else:
          severity = "Critical Low"
        pct_in_range = 0
      elif status == "High":
        diff = max(value - high, 0)
        if diff <= (abs(high) * 0.10 if high != 0 else 1):
          severity = "Mild High"
        elif diff <= (abs(high) * 0.25 if high != 0 else 2):
          severity = "Moderate High"
        else:
          severity = "Critical High"
        pct_in_range = 100
  except Exception:
    pass

  return {
    "Parameter": record.get("Parameter", ""),
    "Value": record.get("Value", ""),
    "Unit": unit,
    "Reference Range": record.get("Reference Range", ""),
    "Status": status,
    "% in Range": pct_in_range,
    "Severity": severity
  }

def build_lab_dataframe(records):
  columns = ["Parameter", "Value", "Unit", "Reference Range", "Status", "% in Range", "Severity"]
  if not records:
    return pd.DataFrame(columns=columns)
  evaluated = [evaluate_record(r) for r in records]
  return pd.DataFrame(evaluated, columns=columns)

def parse_radiology_sections(text):
  cleaned = clean_text(text)
  patterns = {
    "Exam": r"(?:exam|examination)\s*[:\-]\s*(.*?)(?=\n[A-Z][A-Za-z ]{2,24}\s*[:\-]|\Z)",
    "Technique": r"(?:technique)\s*[:\-]\s*(.*?)(?=\n[A-Z][A-Za-z ]{2,24}\s*[:\-]|\Z)",
    "Findings": r"(?:findings)\s*[:\-]\s*(.*?)(?=\n(?:impression|conclusion|recommendation)\s*[:\-]|\Z)",
    "Impression": r"(?:impression|conclusion)\s*[:\-]\s*(.*?)(?=\n[A-Z][A-Za-z ]{2,24}\s*[:\-]|\Z)",
  }
  sections = {}
  for key, pattern in patterns.items():
    match = re.search(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL)
    if match:
      sections[key] = clean_text(match.group(1))

  if not sections:
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    if paragraphs:
      sections["Narrative"] = paragraphs[0][:1500]
  return sections


def parse_general_medical_sections(text):
  cleaned = clean_text(text)
  section_patterns = {
    "Clinical Notes": r"(?:clinical\s*details|clinical\s*notes|history|complaint|indication)\s*[:\-]\s*(.*?)(?=\n[A-Z][A-Za-z ]{2,28}\s*[:\-]|\Z)",
    "Findings": r"(?:findings|observation|observations|result|results)\s*[:\-]\s*(.*?)(?=\n(?:impression|conclusion|advice|recommendation)\s*[:\-]|\Z)",
    "Impression": r"(?:impression|conclusion|summary|diagnosis)\s*[:\-]\s*(.*?)(?=\n[A-Z][A-Za-z ]{2,28}\s*[:\-]|\Z)",
    "Advice": r"(?:advice|recommendation|plan)\s*[:\-]\s*(.*?)(?=\n[A-Z][A-Za-z ]{2,28}\s*[:\-]|\Z)",
  }
  sections = {}
  for key, pattern in section_patterns.items():
    match = re.search(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL)
    if match:
      sections[key] = clean_text(match.group(1))[:1500]

  if not sections:
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    if paragraphs:
      sections["Summary"] = paragraphs[0][:1200]
    if len(paragraphs) > 1:
      sections["Details"] = paragraphs[1][:1200]
  return sections


def format_detected_sections_md(sections, report_category):
  title = "Detected sections / findings"
  if report_category == "Radiology Report":
    title = "Radiology sections"
  elif report_category == "General Medical Report":
    title = "Detected report sections"

  if not sections:
    return f"**{title.title()}**\n\n- No structured sections were detected."

  lines = [f"**{title.title()}**", ""]
  for key, value in sections.items():
    lines.append(f"### {key}")
    lines.append(safe_str(value)[:2000])
    lines.append("")
  return "\n".join(lines).strip()

def parse_reference_bounds(ref_text):
  ref_text = safe_str(ref_text).replace("", "-").strip()
  parts = [p.strip() for p in ref_text.split("-")]
  if len(parts) != 2:
    return None, None
  try:
    return float(parts[0]), float(parts[1])
  except Exception:
    return None, None


def to_float_safe(value):
  try:
    return float(safe_str(value).replace("<", "").replace(">", "").strip())
  except Exception:
    return None


def visual_percent_for_row(row):
  status = safe_str(row.get("Status", "Unknown"))
  pct = row.get("% in Range", None)
  severity = safe_str(row.get("Severity", ""))

  if status == "Normal" and pct is not None and not pd.isna(pct):
    return int(max(18, min(92, float(pct))))

  if status == "Low":
    if "Critical" in severity:
      return 8
    elif "Moderate" in severity:
      return 18
    else:
      return 28

  if status == "High":
    if "Critical" in severity:
      return 92
    elif "Moderate" in severity:
      return 82
    else:
      return 72

  return 50

def status_classes(status):
  status = safe_str(status)
  if status == "High":
    return "status-high", "bar-high", "mini-high"
  if status == "Low":
    return "status-low", "bar-low", "mini-low"
  if status == "Normal":
    return "status-normal", "bar-normal", "mini-normal"
  return "status-unknown", "bar-unknown", "mini-unknown"


def compact_value_label(row):
  value = safe_str(row.get("Value", "")).strip()
  unit = safe_str(row.get("Unit", "")).strip()
  label = f"{value} {unit}".strip()
  return label if label else "-"

def build_lab_visual_html(df, report_category="General Medical Report"):
  if df is None or df.empty:
    empty_message = "This report does not contain a structured lab table. See the extracted sections and summary below."
    if report_category == "Laboratory Report":
      empty_message = "No valid medical test rows were extracted from this report. The system hid the graph to avoid showing non-medical text as results."
    return f"""
    <div class="result-card">
      <div class="result-card-head"><h3>Structured values overview</h3></div>
      <div class="empty-state">{html.escape(empty_message)}</div>
    </div>
    """

  rows_html = []
  status_counts = {"Normal": 0, "Low": 0, "High": 0, "Unknown": 0}
  # Show every dynamically extracted value from this upload.
  # No fixed 12-row cap, so rows like RBC, WBC and Platelets are not hidden.
  for idx, (_, row) in enumerate(df.iterrows(), start=1):
    full_label, short_label = get_display_parameter_labels(row.get("Parameter", ""), idx)
    parameter = html.escape(full_label)
    value_label = html.escape(compact_value_label(row))
    ref_label = html.escape(safe_str(row.get("Reference Range", "")))
    status = safe_str(row.get("Status", "Unknown")).title()
    status_cls, bar_cls, mini_cls = status_classes(status)
    visual_pct = visual_percent_for_row(row)
    short_label = html.escape(short_label)
    status_text = html.escape(status)

    rows_html.append(f"""
    <div class="metric-row">
      <div class="metric-cell metric-param">{parameter}</div>
      <div class="metric-cell metric-value">{value_label}</div>
      <div class="metric-cell metric-range">{ref_label}</div>
      <div class="metric-cell metric-status"><span class="status-pill {status_cls}">{status_text}</span></div>
      <div class="metric-cell metric-visual">
        <div class="visual-wrap">
          <div class="visual-track"><div class="visual-fill {bar_cls}" style="width:{visual_pct}%"></div></div>
          <div class="visual-pct">{visual_pct}%</div>
        </div>
      </div>
    </div>
    """)
    status_counts[status if status in status_counts else "Unknown"] += 1

  return f"""
  <div class="result-card">
    <div class="result-card-head">
      <h3>Structured values overview</h3>
      <div class="mini-note">Dynamic values extracted from this uploaded report</div>
    </div>
    <div class="metric-table">
      <div class="metric-row metric-header">
        <div class="metric-cell metric-param">Parameter</div>
        <div class="metric-cell metric-value">Value</div>
        <div class="metric-cell metric-range">Normal range</div>
        <div class="metric-cell metric-status">Status</div>
        <div class="metric-cell metric-visual">Visual</div>
      </div>
      {''.join(rows_html)}
    </div>
    <div class="chart-wrap">
      <div class="glance-title">Result status chart</div>
      <div class="status-chart">
        <div class="chart-row"><span>Normal</span><div class="chart-track"><div class="chart-fill chart-normal" style="width:{min(100, status_counts['Normal']*100/max(1,sum(status_counts.values())))}%"></div></div><b>{status_counts['Normal']}</b></div>
        <div class="chart-row"><span>Low</span><div class="chart-track"><div class="chart-fill chart-low" style="width:{min(100, status_counts['Low']*100/max(1,sum(status_counts.values())))}%"></div></div><b>{status_counts['Low']}</b></div>
        <div class="chart-row"><span>High</span><div class="chart-track"><div class="chart-fill chart-high" style="width:{min(100, status_counts['High']*100/max(1,sum(status_counts.values())))}%"></div></div><b>{status_counts['High']}</b></div>
      </div>
    </div>
  </div>
  """

def summarize_names(sub_df, fallback):
  if sub_df is None or sub_df.empty:
    return fallback
  cleaned = []
  for idx, raw_name in enumerate(sub_df["Parameter"].tolist()[:3], start=1):
    full_label, _ = get_display_parameter_labels(raw_name, idx)
    if full_label:
      cleaned.append(full_label)
  if not cleaned:
    return fallback
  if len(cleaned) == 1:
    return cleaned[0]
  if len(cleaned) == 2:
    return f"{cleaned[0]} and {cleaned[1]}"
  return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"

def build_summary_card_html(report_category, report_subtype, patient_info, df, radiology_sections):
  patient_name = html.escape(safe_str(patient_info.get("Patient Name", "Not Found")))
  subtype = html.escape(safe_str(report_subtype))
  category = html.escape(safe_str(report_category))

  if df is not None and not df.empty:
    high_df = df[df["Status"] == "High"]
    low_df = df[df["Status"] == "Low"]
    normal_df = df[df["Status"] == "Normal"]

    high_names = html.escape(summarize_names(high_df, "No high value detected"))
    low_names = html.escape(summarize_names(low_df, "No low value detected"))
    normal_names = html.escape(summarize_names(normal_df, "No normal value identified"))

    return f"""
    <div class="summary-card">
      <div class="summary-head">
        <div class="summary-title">Final report summary</div>
        <div class="complete-badge">Complete</div>
      </div>
      <div class="summary-meta">
        <span class="meta-pill">{category}</span>
        <span class="meta-pill">{subtype}</span>
        <span class="meta-pill">Patient: {patient_name}</span>
        <span class="meta-pill">Total values: {len(df)}</span>
      </div>
      <div class="summary-item summary-alert">
        <div class="summary-icon">!</div>
        <div>
          <div class="summary-item-title">{len(high_df)} high values</div>
          <div class="summary-item-text">{high_names}</div>
        </div>
      </div>
      <div class="summary-item summary-low">
        <div class="summary-icon"></div>
        <div>
          <div class="summary-item-title">{len(low_df)} low values</div>
          <div class="summary-item-text">{low_names}</div>
        </div>
      </div>
      <div class="summary-item summary-normal">
        <div class="summary-icon"></div>
        <div>
          <div class="summary-item-title">{len(normal_df)} normal values</div>
          <div class="summary-item-text">{normal_names}</div>
        </div>
      </div>
      <div class="summary-footnote">AI-generated for educational use only.</div>
    </div>
    """

  if radiology_sections:
    sec_items = list(radiology_sections.items())[:2]
    primary_title = html.escape(sec_items[0][0]) if sec_items else "Summary"
    primary_text = html.escape(safe_str(sec_items[0][1])[:220]) if sec_items else "No key section extracted."
    secondary_title = html.escape(sec_items[1][0]) if len(sec_items) > 1 else "Review note"
    secondary_text = html.escape(safe_str(sec_items[1][1])[:220]) if len(sec_items) > 1 else "This report contains narrative text rather than a numeric lab table."
    return f"""
    <div class="summary-card">
      <div class="summary-head">
        <div class="summary-title">Final report summary</div>
        <div class="complete-badge">Complete</div>
      </div>
      <div class="summary-meta">
        <span class="meta-pill">{category}</span>
        <span class="meta-pill">{subtype}</span>
        <span class="meta-pill">Patient: {patient_name}</span>
      </div>
      <div class="summary-item summary-alert">
        <div class="summary-icon">!</div>
        <div>
          <div class="summary-item-title">{primary_title}</div>
          <div class="summary-item-text">{primary_text}</div>
        </div>
      </div>
      <div class="summary-item summary-normal">
        <div class="summary-icon"></div>
        <div>
          <div class="summary-item-title">{secondary_title}</div>
          <div class="summary-item-text">{secondary_text}</div>
        </div>
      </div>
      <div class="summary-footnote">AI-generated for educational use only.</div>
    </div>
    """

  return f"""
  <div class="summary-card">
    <div class="summary-head">
      <div class="summary-title">Final report summary</div>
      <div class="complete-badge">Complete</div>
    </div>
    <div class="summary-meta">
      <span class="meta-pill">{category}</span>
      <span class="meta-pill">{subtype}</span>
      <span class="meta-pill">Patient: {patient_name}</span>
    </div>
    <div class="summary-item summary-low">
      <div class="summary-icon">~</div>
      <div>
        <div class="summary-item-title">No structured table detected</div>
        <div class="summary-item-text">The uploaded document looks more like narrative medical text than a lab-values table.</div>
      </div>
    </div>
    <div class="summary-footnote">AI-generated for educational use only.</div>
  </div>
  """

def generate_ai_explanation(report_text, report_category, report_subtype, lab_df, radiology_sections):
  try:
    if client is None or MEDIBUDDY_FAST_MODE:
      if lab_df is not None and not lab_df.empty:
        high_df = lab_df[lab_df["Status"] == "High"]
        low_df = lab_df[lab_df["Status"] == "Low"]
        normal_df = lab_df[lab_df["Status"] == "Normal"]
        return (
          f"This report was detected as {report_subtype.lower()}. "
          f"High values: {summarize_names(high_df, 'none found')}. "
          f"Low values: {summarize_names(low_df, 'none found')}. "
          f"Normal values: {summarize_names(normal_df, 'none found')}. "
          f"This summary is for educational use only and is not a confirmed diagnosis."
        )

      if radiology_sections:
        first_items = list(radiology_sections.items())[:2]
        joined = " ".join(f"{k}: {safe_str(v)}" for k, v in first_items)
        return (
          f"This report was detected as {report_subtype.lower()}. "
          f"Main extracted details: {joined[:600]}. "
          f"This is a simplified educational explanation and not a confirmed diagnosis."
        )

      return (
        f"This report was detected as {report_subtype.lower()}. "
        f"The system could read the report text, but it did not find a structured lab table. "
        f"Please review the extracted text and summary below. This is for educational use only."
      )

    lab_context = ""
    if lab_df is not None and not lab_df.empty:
      preview_cols = ["Parameter", "Value", "Unit", "Reference Range", "Status"]
      preview_df = lab_df[[c for c in preview_cols if c in lab_df.columns]].head(20)
      lab_context = preview_df.to_string(index=False)

    section_context = ""
    if radiology_sections:
      section_context = "\n".join([f"{k}: {safe_str(v)}" for k, v in radiology_sections.items() if safe_str(v).strip()])

    prompt = f"""
You are a careful medical report explanation assistant for educational purposes only.

Report category: {report_category}
Report subtype: {report_subtype}

Raw report text:
{safe_str(report_text)[:5000]}

Structured values found:
{lab_context}

Detected sections:
{section_context}

Write a short, easy-to-read explanation for a non-medical user.

Rules:
1. Keep it simple and readable.
2. If numeric values were found, mention the important high or low values first.
3. If this is narrative medical text, summarize the main findings or impression in plain English.
4. Do not give a confirmed diagnosis.
5. Use cautious wording such as "may suggest", "appears", or "shows".
6. End with a short disclaimer that this is educational only.

Output in 1 short paragraph or 3 to 5 bullet points.
"""

    response = client.chat.completions.create(
      model=TEXT_MODEL,
      messages=[
        {"role": "system", "content": "You explain medical reports in simple language for educational use only."},
        {"role": "user", "content": prompt},
      ],
      temperature=0.2,
    )
    return safe_str(response.choices[0].message.content).strip()

  except Exception as e:
    return f"AI explanation could not be generated: {str(e)}"

def generate_final_report_summary(report_category, report_subtype, patient_info, lab_df, radiology_sections):
  patient_name = safe_str(patient_info.get("Patient Name", "Not Found"))

  if lab_df is not None and not lab_df.empty:
    total_tests = len(lab_df)
    high_df = lab_df[lab_df["Status"] == "High"]
    low_df = lab_df[lab_df["Status"] == "Low"]
    normal_df = lab_df[lab_df["Status"] == "Normal"]

    lines = [
      f"Report Category: {report_category}",
      f"Report Type: {report_subtype}",
      f"Patient Name: {patient_name}",
      f"Total Extracted Values: {total_tests}",
      f"High Values: {len(high_df)}",
      f"Low Values: {len(low_df)}",
      f"Normal Values: {len(normal_df)}",
      "",
      "Key Findings:",
      f"High: {summarize_names(high_df, 'No high value detected')}",
      f"Low: {summarize_names(low_df, 'No low value detected')}",
      f"Normal: {summarize_names(normal_df, 'No normal value identified')}",
      "",
      "Important: This summary is AI-generated for educational purposes only and is not a confirmed medical diagnosis."
    ]
    return "\n".join(lines)

  if radiology_sections:
    lines = [
      f"Report Category: {report_category}",
      f"Report Type: {report_subtype}",
      f"Patient Name: {patient_name}",
      "",
      "Extracted Sections:"
    ]
    for key, value in list(radiology_sections.items())[:3]:
      lines.append(f"{key}: {safe_str(value)[:400]}")
    lines.extend([
      "",
      "Important: This summary is AI-generated for educational purposes only and is not a confirmed medical diagnosis."
    ])
    return "\n".join(lines)

  return "\n".join([
    f"Report Category: {report_category}",
    f"Report Type: {report_subtype}",
    f"Patient Name: {patient_name}",
    "",
    "No structured values were extracted from this report.",
    "The report may contain narrative medical text rather than a lab-values table.",
    "",
    "Important: This summary is AI-generated for educational purposes only and is not a confirmed medical diagnosis."
  ])



# 
# 2-DAY POLISH FEATURES: practical, stable improvements for presentation
# 
PARSER_VERSION = "dynamic-v5-merge-ai-and-ocr-all-rows"
REPORT_ANALYSIS_CACHE = {}
XRAY_RESULT_CACHE = {}
XRAY_STABLE_RESULT_VERSION = "xray-stable-v5-filmphoto-cache"
MAX_UPLOAD_SIZE_MB = 12
MEDICAL_DISCLAIMER_TEXT = (
  "This AI analysis is for educational support only. It is not a medical diagnosis. "
  "Please consult a qualified doctor for final interpretation."
)

def calculate_file_hash(file_path):
  """Return a stable hash so repeated uploads can reuse the same analysis."""
  try:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
      for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()
  except Exception:
    return ""

def validate_uploaded_medical_file(file_path):
  """Friendly validation before OCR/AI work starts."""
  if not file_path or not os.path.exists(file_path):
    return False, "Uploaded file path is invalid. Please upload the file again."
  ext = os.path.splitext(file_path)[1].lower()
  if ext not in [".pdf", ".png", ".jpg", ".jpeg"]:
    return False, "Unsupported file format. Please upload a PDF, PNG, JPG, or JPEG medical report."
  try:
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
      return False, f"File is too large ({size_mb:.1f} MB). Please upload a file under {MAX_UPLOAD_SIZE_MB} MB."
    if size_mb <= 0:
      return False, "This file appears to be empty. Please upload a valid medical report."
  except Exception:
    pass
  return True, "OK"

NON_MEDICAL_REPORT_ERROR = (
  "This file does not look like a medical report. Please upload a valid lab report, "
  "radiology report, prescription, or X-ray image."
)
NON_XRAY_IMAGE_ERROR = "This image does not look like a medical X-ray. Please upload a valid X-ray image."
BLURRY_XRAY_IMAGE_ERROR = "This X-ray image looks too blurred or unclear for a reliable review. Please upload a sharper, well-focused X-ray image."
ANNOTATED_XRAY_IMAGE_ERROR = "This image looks annotated, labeled, circled, or combined from more than one X-ray panel. Please upload the original single, clear, unmarked X-ray image for a better educational review."

def medical_text_confidence_score(text):
  """Lightweight safety check so random files are not analyzed as medical reports."""
  text = safe_str(text).lower()
  if not text.strip():
    return 0
  medical_terms = [
    "patient", "doctor", "physician", "hospital", "clinic", "laboratory", "diagnostic",
    "report", "specimen", "sample", "collection", "reference range", "normal range",
    "hemoglobin", "haemoglobin", "hematocrit", "haematocrit", "wbc", "rbc", "platelet", "cbc", "glucose", "hba1c",
    "cholesterol", "triglycerides", "hdl", "ldl", "creatinine", "urea", "bun",
    "bilirubin", "alt", "ast", "sgpt", "sgot", "tsh", "t3", "t4", "urine",
    "x-ray", "xray", "radiograph", "radiology", "findings", "impression", "ct", "mri",
    "ultrasound", "prescription", "medicine", "diagnosis", "clinical", "blood", "serum"
  ]
  score = 0
  score += sum(1 for term in medical_terms if term in text)
  score += min(5, len(re.findall(r"\d+(?:\.\d+)?\s*[-]\s*\d+(?:\.\d+)?", text)))
  score += min(5, len(re.findall(r"\d+(?:\.\d+)?\s*(?:mg/dl|g/dl|mmol/l|u/l|iu/l|%|fl|pg|ng/ml|x10)", text, flags=re.I)))
  return score

def looks_like_medical_report_text(text, parsed_records=None, report_category=""):
  """Return True only when the extracted content looks medical enough."""
  if parsed_records and len(parsed_records) >= 1:
    return True
  category = safe_str(report_category).lower()
  if any(word in category for word in ["laboratory", "radiology", "x-ray"]):
    return True
  return medical_text_confidence_score(text) >= 3

def looks_like_xray_image_file(file_path):
  """Strict local validation for X-ray tab before calling the vision model.

  This is only a quick safety filter. The stronger check is done with the
  vision model when GROQ_API_KEY is available.
  """
  if not file_path or not os.path.exists(file_path):
    return False
  ext = os.path.splitext(file_path)[1].lower()
  if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
    return False
  try:
    img = cv2.imread(file_path)
    if img is None:
      return False
    h, w = img.shape[:2]
    if h < 160 or w < 160:
      return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    gray_like_ratio = float((saturation < 28).mean())

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())
    mean_intensity = float(gray.mean())

    # X-rays are usually near-grayscale, have meaningful contrast, and are not
    # dominated by very colorful natural-photo pixels. This blocks most random photos.
    is_grayscale_medical_style = gray_like_ratio >= 0.86 and contrast >= 22

    # Extra radiograph-like clue: noticeable dark and bright regions together.
    dark_ratio = float((gray < 55).mean())
    bright_ratio = float((gray > 185).mean())
    has_radiograph_range = dark_ratio >= 0.05 and bright_ratio >= 0.03 and 35 <= mean_intensity <= 220

    return bool(is_grayscale_medical_style and has_radiograph_range)
  except Exception:
    return False


def validate_xray_with_vision_model(file_path):
  """Use the vision model to reject random/non-medical images in the X-ray tab.

  Returns True for a medical X-ray, False for a non-X-ray, and None if the
  validation call cannot run.
  """
  if client is None or not file_path or not os.path.exists(file_path):
    return None
  try:
    image_data_url = encode_image_to_data_url(file_path)
    response = client.chat.completions.create(
      model=VISION_MODEL,
      messages=[
        {
          "role": "system",
          "content": "You are a strict medical image validator. Return JSON only."
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": (
                "Decide if this uploaded image is a real medical X-ray/radiograph image. "
                "Return strict JSON only: {\"is_xray\": true/false, \"reason\": \"short reason\"}. "
                "Mark false for selfies, documents, screenshots, food, scenery, cartoons, normal photos, "
                "or any non-radiology image. Mark true only when the image visibly looks like a medical X-ray/radiograph."
              ),
            },
            {"type": "image_url", "image_url": {"url": image_data_url}},
          ],
        },
      ],
      temperature=0.0,
      max_tokens=120,
    )
    raw = safe_str(response.choices[0].message.content).strip()
    parsed = extract_json_object(raw)
    if isinstance(parsed, dict) and "is_xray" in parsed:
      return bool(parsed.get("is_xray"))
  except Exception:
    return None
  return None

def build_validation_error_html(message):
  return f"""
  <div class="xray-card">
    <div class="xray-head">
      <div>
        <div class="xray-title">Upload Error</div>
        <div class="xray-subtitle">Please upload the correct medical file type.</div>
      </div>
      <div class="xray-status-badge xray-red">! Invalid file</div>
    </div>
    <div class="xray-disclaimer-box"> {html.escape(safe_str(message))}</div>
  </div>
  """


def compute_ocr_quality_score(raw_text, file_path=None):
  """Estimate OCR/text extraction quality without needing ground truth."""
  text = safe_str(raw_text)
  chars = len(text)
  words = re.findall(r"[A-Za-z0-9]+", text)
  word_count = len(words)
  noisy_chars = len(re.findall(r"[^\w\s.,:;/%()\-+\[\]]", text))
  numeric_hits = len(re.findall(r"\d+(?:\.\d+)?", text))
  common_medical_hits = len(re.findall(
    r"hemoglobin|wbc|rbc|platelet|cholesterol|glucose|creatinine|tsh|bilirubin|report|patient|result|range|unit|x[- ]?ray|impression|findings",
    text.lower()
  ))
  score = 35
  if chars > 300: score += 15
  if chars > 900: score += 10
  if word_count > 60: score += 10
  if numeric_hits >= 3: score += 10
  if common_medical_hits >= 2: score += 10
  if common_medical_hits >= 6: score += 5
  noise_ratio = noisy_chars / max(chars, 1)
  if noise_ratio > 0.08: score -= 15
  if noise_ratio > 0.15: score -= 15
  if chars < 80: score -= 25
  score = max(5, min(98, int(score)))
  if score >= 80:
    label = "Excellent"
  elif score >= 65:
    label = "Good"
  elif score >= 45:
    label = "Readable but needs review"
  else:
    label = "Poor / unclear scan"
  return {
    "score": score,
    "label": label,
    "characters": chars,
    "words": word_count,
    "numbers_detected": numeric_hits,
    "noise_ratio": round(noise_ratio, 3),
    "note": "OCR quality is estimated automatically from extracted text clarity and medical-value patterns."
  }

def classify_report_type_ml_style(text):
  """Lightweight ML-style classifier using weighted keyword scoring."""
  lower = safe_str(text).lower()
  report_types = {
    "CBC / Hematology Report": ["hemoglobin", "hb", "wbc", "rbc", "platelet", "mcv", "mch", "mchc", "neutrophil", "lymphocyte", "monocyte", "eosinophil"],
    "Lipid Profile": ["cholesterol", "ldl", "hdl", "triglyceride", "vldl", "cholesterol ratio", "non hdl"],
    "Liver Function Test": ["sgpt", "sgot", "alt", "ast", "bilirubin", "alkaline phosphatase", "albumin", "globulin", "liver"],
    "Kidney Function Test": ["creatinine", "urea", "bun", "egfr", "uric acid", "sodium", "potassium", "chloride", "kidney"],
    "Thyroid Function Test": ["tsh", "t3", "t4", "free t3", "free t4", "thyroid"],
    "Diabetes / Glucose Report": ["glucose", "fasting blood sugar", "random blood sugar", "hba1c", "diabetes", "blood sugar"],
    "Urine Test": ["urine", "pus cells", "epithelial cells", "specific gravity", "ketone", "bacteria", "protein"],
    "Radiology / X-ray Report": ["x-ray", "xray", "radiograph", "impression", "findings", "fracture", "lungs", "chest", "opacity"],
  }
  scores = {}
  matched = {}
  for name, keywords in report_types.items():
    hits = [kw for kw in keywords if kw in lower]
    scores[name] = len(hits)
    matched[name] = hits
  best = max(scores, key=scores.get)
  best_score = scores[best]
  if best_score == 0:
    return {"report_type": "General Medical Report", "confidence": 30, "matched_keywords": []}
  confidence = min(95, 45 + best_score * 10)
  return {"report_type": best, "confidence": confidence, "matched_keywords": matched[best]}

def build_risk_score(lab_records):
  """Simple presentation-safe risk score based on abnormal lab statuses."""
  records = lab_records or []
  total = len(records)
  abnormal = []
  high = []
  low = []
  critical_words = ["critical", "danger", "panic", "urgent", "severe"]
  for r in records:
    status = safe_str(r.get("Status", "")).strip().lower()
    blob = " ".join([safe_str(r.get(k, "")) for k in ["Parameter", "Value", "Unit", "Reference Range", "Status", "Severity"]]).lower()
    if status in ["high", "low", "abnormal"] or any(w in blob for w in critical_words):
      abnormal.append(r)
    if status == "high":
      high.append(r)
    if status == "low":
      low.append(r)
  abnormal_count = len(abnormal)
  if abnormal_count == 0:
    level, score = "Low", 15
  elif abnormal_count <= 2:
    level, score = "Moderate", 45
  elif abnormal_count <= 5:
    level, score = "High", 70
  else:
    level, score = "Needs Attention", 88
  if any(any(w in " ".join(map(safe_str, r.values())).lower() for w in critical_words) for r in abnormal):
    level, score = "Needs Attention", max(score, 92)
  return {
    "level": level,
    "score": score,
    "total_values": total,
    "abnormal_count": abnormal_count,
    "high_count": len(high),
    "low_count": len(low),
    "normal_count": max(0, total - abnormal_count),
    "note": "Rule-based educational risk estimate, not a diagnosis."
  }

def generate_health_suggestions(lab_records):
  """Safe, simple suggestions using 'may' language only."""
  suggestions = []
  for r in (lab_records or []):
    status = safe_str(r.get("Status", "")).strip().lower()
    if status not in ["high", "low", "abnormal"]:
      continue
    name = safe_str(r.get("Parameter", "value")).strip() or "value"
    lname = name.lower()
    if "hemoglobin" in lname or lname in ["hb"]:
      msg = "Low hemoglobin may be related to anemia, iron deficiency, B12/folate issues, or other causes. Discuss this with a doctor."
    elif "wbc" in lname or "white" in lname:
      msg = "Abnormal WBC may be related to infection, inflammation, stress, or other medical causes. Doctor review is recommended."
    elif "platelet" in lname:
      msg = "Abnormal platelets may need clinical correlation, especially with bleeding, bruising, fever, or infection symptoms."
    elif "cholesterol" in lname or "ldl" in lname or "triglycer" in lname:
      msg = "Abnormal lipid values may need diet, lifestyle, and medical review depending on age and risk factors."
    elif "glucose" in lname or "hba1c" in lname or "sugar" in lname:
      msg = "Abnormal sugar markers may need repeat testing and diabetes risk discussion with a healthcare professional."
    elif "creatinine" in lname or "urea" in lname or "egfr" in lname:
      msg = "Kidney-related abnormal values should be reviewed with hydration status, medicines, and medical history."
    elif "tsh" in lname or "thyroid" in lname or lname in ["t3", "t4"]:
      msg = "Thyroid-related abnormal values may need clinical review and sometimes repeat thyroid testing."
    elif "bilirubin" in lname or "alt" in lname or "ast" in lname or "sgpt" in lname or "sgot" in lname:
      msg = "Liver-related abnormal values may need review with symptoms, medicines, infections, and doctor advice."
    else:
      msg = f"{name} is marked {status}. Please discuss this result with a qualified healthcare professional."
    suggestions.append({"parameter": name, "status": status.title(), "suggestion": msg})
  if not suggestions:
    suggestions.append({"parameter": "General", "status": "Info", "suggestion": "No clear abnormal lab value was detected, but symptoms and medical history still matter. Review the report with a clinician if you feel unwell."})
  return suggestions[:8]

def build_polished_result_summary_html(report_category, report_subtype, ocr_quality, risk_score, classifier, suggestions):
  """A clean top summary card for the result page."""
  risk = html.escape(safe_str(risk_score.get("level", "Unknown")))
  risk_num = html.escape(safe_str(risk_score.get("score", "")))
  ocr_label = html.escape(safe_str(ocr_quality.get("label", "Unknown")))
  ocr_num = html.escape(safe_str(ocr_quality.get("score", "")))
  clf_type = html.escape(safe_str(classifier.get("report_type", report_subtype)))
  clf_conf = html.escape(safe_str(classifier.get("confidence", "")))
  sug_items = "".join([
    f"<li><b>{html.escape(safe_str(s.get('parameter','')))}</b>: {html.escape(safe_str(s.get('suggestion','')))}</li>"
    for s in (suggestions or [])[:4]
  ])
  return f"""
  <div class="summary-card" style="border:1px solid #dbeafe; background:linear-gradient(135deg,#ffffff,#f8fbff);">
    <div class="summary-head">
      <div class="summary-title">Professional Result Summary</div>
      <div class="complete-badge">Ready</div>
    </div>
    <div class="summary-meta">
      <span class="meta-pill">Detected: {html.escape(safe_str(report_category))}</span>
      <span class="meta-pill">Type: {clf_type}</span>
      <span class="meta-pill">Classifier confidence: {clf_conf}%</span>
      <span class="meta-pill">OCR: {ocr_label} ({ocr_num}%)</span>
      <span class="meta-pill">Risk: {risk} ({risk_num}/100)</span>
    </div>
    <div class="summary-item summary-alert">
      <div>
        <div class="summary-item-title">Overall risk level: {risk}</div>
        <div class="summary-item-text">{risk_score.get('abnormal_count',0)} abnormal values found from {risk_score.get('total_values',0)} extracted values. This is rule-based educational scoring only.</div>
      </div>
    </div>
    <div class="summary-item summary-normal">
      <div>
        <div class="summary-item-title">Simple health suggestions</div>
        <div class="summary-item-text"><ul style="margin:6px 0 0 18px; padding:0;">{sug_items}</ul></div>
      </div>
    </div>
    <div class="summary-footnote">{html.escape(MEDICAL_DISCLAIMER_TEXT)}</div>
  </div>
  """

def save_json_report(data):
  file_path = os.path.join(tempfile.gettempdir(), f"medical_report_{uuid.uuid4().hex}.json")
  with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
  return file_path

def split_text_lines(text, max_chars=95):
  words = safe_str(text).split()
  lines = []
  current = []
  current_len = 0
  for word in words:
    add_len = len(word) + (1 if current else 0)
    if current_len + add_len <= max_chars:
      current.append(word)
      current_len += add_len
    else:
      lines.append(" ".join(current))
      current = [word]
      current_len = len(word)
  if current:
    lines.append(" ".join(current))
  return lines if lines else [""]

def normalize_report_language_choice(language_choice):
  language_choice = safe_str(language_choice or "English").strip().lower()
  if "urdu" in language_choice and "english" in language_choice:
    return "Both"
  if "urdu" in language_choice:
    return "Urdu"
  return "English"

def _register_pdf_fonts():
  """Register Unicode fonts for English + Urdu PDF output in Colab."""
  regular = "Helvetica"
  bold = "Helvetica-Bold"
  candidates = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
  ]
  bold_candidates = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
  ]
  try:
    for path in candidates:
      if os.path.exists(path):
        pdfmetrics.registerFont(TTFont("MediBuddySans", path))
        regular = "MediBuddySans"
        break
    for path in bold_candidates:
      if os.path.exists(path):
        pdfmetrics.registerFont(TTFont("MediBuddySans-Bold", path))
        bold = "MediBuddySans-Bold"
        break
  except Exception:
    regular = "Helvetica"
    bold = "Helvetica-Bold"
  return regular, bold

def _rtl_text(text):
  text = safe_str(text)
  if arabic_reshaper is not None and get_display is not None:
    try:
      return get_display(arabic_reshaper.reshape(text))
    except Exception:
      return text
  return text

def _basic_urdu_translation(text, context="medical report"):
  """Local patient-friendly Urdu fallback when Groq is not configured."""
  text = _clean_pdf_text(text) if '_clean_pdf_text' in globals() else safe_str(text)
  lower = text.lower()
  # Common generated summary patterns
  if "educational" in lower and ("diagnosis" in lower or "medical advice" in lower):
    return "            "
  if "no patient information" in lower:
    return "       "
  if "x-ray" in lower or "x ray" in lower or "radiology" in lower:
    status_line = ""
    if "needs attention" in lower:
      status_line = "          "
    elif "no obvious acute abnormality" in lower:
      status_line = "         "
    elif "limited" in lower or "unclear" in lower:
      status_line = "        "
    return (
      "     " + status_line +
      "          "
      "          "
    )
  if "laboratory" in lower or "cbc" in lower or "hemoglobin" in lower or "monocytes" in lower or "low" in lower or "high" in lower:
    parts = []
    if "hemoglobin" in lower:
      parts.append("    ")
    if "low" in lower:
      parts.append("   ")
    if "high" in lower:
      parts.append("   ")
    if not parts:
      parts.append("    ")
    return " ".join(parts) + "              "
  # General fallback: not same English; gives useful Urdu note and keeps critical values if any.
  values = re.findall(r"[A-Za-z][A-Za-z\s/%()_-]{1,35}:?\s*\d+(?:\.\d+)?\s*[A-Za-z/%]*", text)[:8]
  value_note = "\n".join(values)
  if value_note:
    return "          :\n" + value_note
  return "          "

def _maybe_translate_to_urdu(text, context="medical report"):
  """Translate text to simple Urdu. Uses Groq when available; otherwise uses a local Urdu fallback."""
  text = safe_str(text).strip()
  if not text:
    return ""
  if client is None:
    return _basic_urdu_translation(text, context)
  try:
    prompt = (
      "Translate the following medical-report text into simple, patient-friendly Urdu. "
      "Keep numbers, units, lab names, and medical terms clear. Do not add new medical advice. "
      "Return Urdu only, not English.\n\n"
      f"Context: {context}\nText:\n{text[:3500]}"
    )
    response = client.chat.completions.create(
      model=TEXT_MODEL,
      messages=[
        {"role": "system", "content": "You translate medical education text into clear Urdu. Keep it concise and safe. Return Urdu only."},
        {"role": "user", "content": prompt},
      ],
      temperature=0.1,
      max_tokens=900,
    )
    translated = response.choices[0].message.content.strip()
    # If the model accidentally returns mostly English, use local Urdu fallback instead.
    urdu_chars = len(re.findall(r"[\u0600-\u06FF]", translated))
    if urdu_chars < 10:
      return _basic_urdu_translation(text, context)
    return translated
  except Exception:
    return _basic_urdu_translation(text, context)

def _clean_pdf_text(text):
  text = re.sub(r"#+", "", safe_str(text))
  text = re.sub(r"\s+", " ", text).strip()
  return text

def _draw_medibuddy_pdf_logo(canv, doc):
  """Draw the existing MediBuddy AI brand mark on every PDF page.

  The app logo is text/CSS based, not a separate image asset, so the PDF
  header recreates the same green medical brand mark directly with ReportLab.
  """
  try:
    page_width, page_height = doc.pagesize
    icon_size = 26
    icon_x = doc.leftMargin
    icon_y = page_height - 42
    text_x = icon_x + icon_size + 8
    text_y = icon_y + 8

    canv.saveState()
    canv.setFillColor(colors.HexColor("#EAF7F1"))
    canv.setStrokeColor(colors.HexColor("#B8DCD0"))
    canv.roundRect(icon_x, icon_y, icon_size, icon_size, 8, stroke=1, fill=1)

    # DNA-style mark matching the application's existing logo concept.
    canv.setStrokeColor(colors.HexColor("#08785D"))
    canv.setLineWidth(1.4)
    canv.bezier(icon_x + 8, icon_y + 6, icon_x + 18, icon_y + 9, icon_x + 8, icon_y + 17, icon_x + 18, icon_y + 20)
    canv.bezier(icon_x + 18, icon_y + 6, icon_x + 8, icon_y + 9, icon_x + 18, icon_y + 17, icon_x + 8, icon_y + 20)
    for y_offset in (8, 13, 18):
      canv.line(icon_x + 9, icon_y + y_offset, icon_x + 17, icon_y + y_offset + 1)

    canv.setFont("Helvetica-Bold", 10)
    canv.setFillColor(colors.HexColor("#17203A"))
    canv.drawString(text_x, text_y, "MediBuddy")
    ai_x = text_x + canv.stringWidth("MediBuddy ", "Helvetica-Bold", 10)
    canv.setFillColor(colors.HexColor("#08785D"))
    canv.drawString(ai_x, text_y, "AI")

    canv.setStrokeColor(colors.HexColor("#E2E8E5"))
    canv.setLineWidth(0.5)
    canv.line(doc.leftMargin, icon_y - 9, page_width - doc.rightMargin, icon_y - 9)
    canv.restoreState()
  except Exception:
    try:
      canv.restoreState()
    except Exception:
      pass

def save_pdf_report(payload, language_choice="English"):
  """Create a polished, readable English-only PDF summary."""
  # English-only export: ignore older Urdu / bilingual choices to prevent extra pages and font issues.
  language_choice = "English"
  original_name = safe_str(payload.get("original_filename", "")).strip()
  if original_name:
    base = os.path.splitext(os.path.basename(original_name))[0]
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or "medical_report"
  else:
    base = f"medical_report_{uuid.uuid4().hex[:8]}"
  suffix = "English"
  file_path = os.path.join(tempfile.gettempdir(), f"{base}_AI_summary_{suffix}.pdf")

  regular_font, bold_font = _register_pdf_fonts()
  doc = SimpleDocTemplate(
    file_path,
    pagesize=letter,
    rightMargin=42,
    leftMargin=42,
    topMargin=72,
    bottomMargin=42,
    title="AI Medical Report Summary",
  )

  styles = getSampleStyleSheet()
  title_style = ParagraphStyle(
    "MediBuddyTitle", parent=styles["Title"], fontName=bold_font, fontSize=18,
    leading=24, alignment=TA_CENTER, textColor=colors.HexColor("#17203a"), spaceAfter=10,
  )
  subtitle_style = ParagraphStyle(
    "MediBuddySubtitle", parent=styles["Normal"], fontName=regular_font, fontSize=10,
    leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#5f6b85"), spaceAfter=16,
  )
  h_style = ParagraphStyle(
    "MediBuddyHeading", parent=styles["Heading2"], fontName=bold_font, fontSize=13,
    leading=17, textColor=colors.HexColor("#312e81"), spaceBefore=12, spaceAfter=7,
  )
  body_style = ParagraphStyle(
    "MediBuddyBody", parent=styles["BodyText"], fontName=regular_font, fontSize=9.5,
    leading=14, textColor=colors.HexColor("#17203a"), spaceAfter=6,
  )
  small_style = ParagraphStyle(
    "MediBuddySmall", parent=styles["BodyText"], fontName=regular_font, fontSize=8.5,
    leading=12, textColor=colors.HexColor("#44506a"), spaceAfter=4,
  )
  urdu_style = ParagraphStyle(
    "MediBuddyUrdu", parent=styles["BodyText"], fontName=regular_font, fontSize=10.5,
    leading=17, alignment=TA_RIGHT, textColor=colors.HexColor("#17203a"), spaceAfter=8,
  )
  urdu_h_style = ParagraphStyle(
    "MediBuddyUrduHeading", parent=styles["Heading2"], fontName=bold_font, fontSize=13,
    leading=18, alignment=TA_RIGHT, textColor=colors.HexColor("#312e81"), spaceBefore=10, spaceAfter=7,
  )

  story = []

  def P(text, style=body_style):
    text = html.escape(_clean_pdf_text(text)).replace("\n", "<br/>")
    story.append(Paragraph(text, style))

  def H(text):
    story.append(Paragraph(html.escape(safe_str(text)), h_style))

  def U(text, style=urdu_style):
    shaped = _rtl_text(_clean_pdf_text(text)).replace("\n", "<br/>")
    story.append(Paragraph(html.escape(shaped), style))

  def UH(text):
    shaped = _rtl_text(safe_str(text))
    story.append(Paragraph(html.escape(shaped), urdu_h_style))

  def add_key_value_table(rows):
    if not rows:
      return
    data = [[Paragraph(f"<b>{html.escape(safe_str(k))}</b>", small_style), Paragraph(html.escape(safe_str(v)), small_style)] for k, v in rows if safe_str(v).strip()]
    if not data:
      return
    table = Table(data, colWidths=[150, 330])
    table.setStyle(TableStyle([
      ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f8fafc")),
      ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#dbe3f0")),
      ("INNERGRID", (0,0), (-1,-1), 0.35, colors.HexColor("#e6edf7")),
      ("VALIGN", (0,0), (-1,-1), "TOP"),
      ("LEFTPADDING", (0,0), (-1,-1), 8),
      ("RIGHTPADDING", (0,0), (-1,-1), 8),
      ("TOPPADDING", (0,0), (-1,-1), 6),
      ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))

  def add_lab_table(lab_records):
    if not lab_records:
      return
    header = ["Parameter", "Value", "Reference Range", "Status"]
    data = [[Paragraph(f"<b>{h}</b>", small_style) for h in header]]
    for row in lab_records[:40]:
      value = f"{row.get('Value','')} {row.get('Unit','')}".strip()
      data.append([
        Paragraph(html.escape(safe_str(row.get("Parameter", ""))), small_style),
        Paragraph(html.escape(safe_str(value)), small_style),
        Paragraph(html.escape(safe_str(row.get("Reference Range", ""))), small_style),
        Paragraph(html.escape(safe_str(row.get("Status", ""))), small_style),
      ])
    table = Table(data, colWidths=[155, 90, 160, 75], repeatRows=1)
    table.setStyle(TableStyle([
      ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8f4ef")),
      ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#312e81")),
      ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#dbe3f0")),
      ("INNERGRID", (0,0), (-1,-1), 0.35, colors.HexColor("#e6edf7")),
      ("VALIGN", (0,0), (-1,-1), "TOP"),
      ("LEFTPADDING", (0,0), (-1,-1), 6),
      ("RIGHTPADDING", (0,0), (-1,-1), 6),
      ("TOPPADDING", (0,0), (-1,-1), 5),
      ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

  report_category = safe_str(payload.get("report_category", "Medical Report"))
  report_subtype = safe_str(payload.get("report_subtype", ""))
  patient_info = payload.get("patient_info", {}) or {}
  lab_records = payload.get("lab_records", []) or []
  radiology_sections = payload.get("radiology_sections", {}) or {}
  summary = _clean_pdf_text(payload.get("summary", ""))
  explanation = _clean_pdf_text(payload.get("ai_explanation", ""))
  xray_image_path = safe_str(payload.get("image_path", "")).strip()
  ocr_quality = payload.get("ocr_quality", {}) or {}
  risk_score = payload.get("risk_score", {}) or {}
  classifier = payload.get("ml_classifier", {}) or {}
  health_suggestions = payload.get("health_suggestions", []) or []
  report_identity = f"{report_category} {report_subtype}".lower()
  is_xray_report = any(term in report_identity for term in ["x-ray", "x ray", "xray"])

  def add_xray_image_if_available():
    if not xray_image_path or not os.path.exists(xray_image_path):
      return
    try:
      story.append(Spacer(1, 8))
      H("Uploaded X-ray Image")
      with Image.open(xray_image_path) as img:
        w, h = img.size
      max_w, max_h = 420, 300
      scale = min(max_w / max(w, 1), max_h / max(h, 1))
      draw_w, draw_h = w * scale, h * scale
      img_obj = RLImage(xray_image_path, width=draw_w, height=draw_h)
      img_table = Table([[img_obj]], colWidths=[480])
      img_table.setStyle(TableStyle([
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#dbe3f0")),
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
      ]))
      story.append(img_table)
      story.append(Spacer(1, 10))
    except Exception:
      P("The uploaded X-ray image could not be embedded in the PDF.")

  def add_english_report():
    story.append(Paragraph("AI Medical Report Interpreter Summary", title_style))
    story.append(Paragraph("Educational summary only Not a confirmed diagnosis", subtitle_style))
    add_key_value_table([
      ("Report Category", report_category),
      ("Report Type", report_subtype),
      ("ML-style Classifier", classifier.get("report_type", report_subtype)),
      ("Classifier Confidence", f"{classifier.get('confidence', '')}%" if classifier else ""),
      ("OCR Quality", f"{ocr_quality.get('label', '')} ({ocr_quality.get('score', '')}%)" if ocr_quality else ""),
      ("Risk Level", f"{risk_score.get('level', '')} ({risk_score.get('score', '')}/100)" if risk_score else ""),
      ("Abnormal Values", risk_score.get("abnormal_count", "") if risk_score else ""),
      ("Generated Language", language_choice),
    ])
    if xray_image_path:
      add_xray_image_if_available()
    if patient_info:
      if not is_xray_report:
        H("Patient Information")
      add_key_value_table(list(patient_info.items()))
    elif not is_xray_report:
      H("Patient Information")
      P("No patient information was detected in the uploaded report.")
    H("Summary")
    P(summary or "No short summary was generated.")
    if health_suggestions:
      H("Simple Health Suggestions")
      for item in health_suggestions[:8]:
        P(f"{item.get('parameter','General')} ({item.get('status','Info')}): {item.get('suggestion','')}")
    if lab_records:
      H("Extracted Laboratory Results")
      add_lab_table(lab_records)
    if radiology_sections:
      H("Radiology / Detected Sections")
      add_key_value_table([(k, "; ".join(v) if isinstance(v, list) else v) for k, v in radiology_sections.items()])
    H("AI Explanation")
    P(explanation or "No explanation was generated.")
    H("Important Disclaimer")
    P("This report is for educational purposes only and should not be considered a confirmed medical diagnosis or medical advice. Please consult a qualified healthcare professional.")

  def add_urdu_report():
    urdu_summary = _maybe_translate_to_urdu(summary or "No short summary was generated.", "report summary")
    urdu_explanation = _maybe_translate_to_urdu(explanation or "No explanation was generated.", "AI explanation")
    section_text = "\n".join([f"{k}: {'; '.join(v) if isinstance(v, list) else v}" for k, v in radiology_sections.items()])
    lab_text = "\n".join([f"{r.get('Parameter','')}: {r.get('Value','')} {r.get('Unit','')} | {r.get('Status','')}" for r in lab_records[:25]])
    extra_urdu = _maybe_translate_to_urdu((section_text + "\n" + lab_text).strip(), "medical findings") if (section_text or lab_text) else ""
    UH("  ")
    U("    ")
    UH(" ")
    U(f" : {report_category}\n : {report_subtype}")
    if patient_info:
      UH(" ")
      U("\n".join([f"{k}: {v}" for k, v in patient_info.items()]))
    UH("")
    U(urdu_summary)
    if extra_urdu:
      UH(" ")
      U(extra_urdu)
    UH(" ")
    U(urdu_explanation)
    UH(" ")
    U("            ")

  # Always generate exactly one clean English report.
  add_english_report()

  doc.build(story, onFirstPage=_draw_medibuddy_pdf_logo, onLaterPages=_draw_medibuddy_pdf_logo)
  return file_path

# --------------------------
# Evaluation helpers
# --------------------------
def levenshtein_distance(seq1, seq2):
  len1, len2 = len(seq1), len(seq2)
  dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
  for i in range(len1 + 1):
    dp[i][0] = i
  for j in range(len2 + 1):
    dp[0][j] = j
  for i in range(1, len1 + 1):
    for j in range(1, len2 + 1):
      cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
      dp[i][j] = min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      )
  return dp[len1][len2]

def compute_ocr_text_metrics(reference_text, predicted_text):
  ref = normalize_text_for_compare(reference_text).split()
  pred = normalize_text_for_compare(predicted_text).split()

  if not ref and not pred:
    return {"WER": 0.0, "Token Precision": 1.0, "Token Recall": 1.0, "Token F1": 1.0}

  edit_distance = levenshtein_distance(ref, pred)
  wer = edit_distance / max(len(ref), 1)

  ref_counter = Counter(ref)
  pred_counter = Counter(pred)
  overlap = sum((ref_counter & pred_counter).values())

  precision = overlap / max(sum(pred_counter.values()), 1)
  recall = overlap / max(sum(ref_counter.values()), 1)
  f1 = 0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

  return {
    "WER": round(wer, 4),
    "Token Precision": round(precision, 4),
    "Token Recall": round(recall, 4),
    "Token F1": round(f1, 4)
  }

def compute_multiclass_metrics(y_true, y_pred):
  y_true = [safe_str(x) for x in y_true]
  y_pred = [safe_str(x) for x in y_pred]
  labels = sorted(set(y_true) | set(y_pred))
  if not y_true:
    return {"Accuracy": 0.0, "Macro Precision": 0.0, "Macro Recall": 0.0, "Macro F1": 0.0}

  accuracy = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)

  per_label = []
  for label in labels:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    per_label.append((precision, recall, f1))

  macro_precision = sum(p for p, _, _ in per_label) / max(len(per_label), 1)
  macro_recall = sum(r for _, r, _ in per_label) / max(len(per_label), 1)
  macro_f1 = sum(f for _, _, f in per_label) / max(len(per_label), 1)

  return {
    "Accuracy": round(accuracy, 4),
    "Macro Precision": round(macro_precision, 4),
    "Macro Recall": round(macro_recall, 4),
    "Macro F1": round(macro_f1, 4)
  }

def compute_readability_metrics(text):
  cleaned = safe_str(text).strip()
  if not cleaned:
    return {"Sentences": 0, "Words": 0, "Avg Words/Sentence": 0.0, "Avg Chars/Word": 0.0}

  sentences = max(len(re.findall(r"[.!?]+", cleaned)), 1)
  words = re.findall(r"\b[\w'-]+\b", cleaned)
  num_words = len(words)
  avg_words_sentence = num_words / max(sentences, 1)
  avg_chars_word = sum(len(w) for w in words) / max(num_words, 1)

  return {
    "Sentences": sentences,
    "Words": num_words,
    "Avg Words/Sentence": round(avg_words_sentence, 2),
    "Avg Chars/Word": round(avg_chars_word, 2)
  }

def evaluate_classification_file(csv_file):
  if not csv_file:
    empty = pd.DataFrame()
    return "Please upload a CSV with columns: y_true, y_pred", empty
  try:
    path = csv_file if isinstance(csv_file, str) else getattr(csv_file, "name", None)
    df = pd.read_csv(path)
    required = {"y_true", "y_pred"}
    if not required.issubset(df.columns):
      return f"CSV must contain columns: {sorted(required)}", pd.DataFrame()
    metrics = compute_multiclass_metrics(df["y_true"].tolist(), df["y_pred"].tolist())
    metrics_df = pd.DataFrame([metrics])
    md = "\n".join([f"- **{k}:** {v}" for k, v in metrics.items()])
    return f"### Classification Metrics\n{md}", metrics_df
  except Exception as e:
    return f"Error reading classification CSV: {e}", pd.DataFrame()

def evaluate_detection_file(csv_file):
  if not csv_file:
    empty = pd.DataFrame()
    return "Please upload a CSV with columns: y_true, y_pred", empty
  try:
    path = csv_file if isinstance(csv_file, str) else getattr(csv_file, "name", None)
    df = pd.read_csv(path)
    required = {"y_true", "y_pred"}
    if not required.issubset(df.columns):
      return f"CSV must contain columns: {sorted(required)}", pd.DataFrame()
    metrics = compute_multiclass_metrics(df["y_true"].tolist(), df["y_pred"].tolist())
    metrics_df = pd.DataFrame([metrics])
    md = "\n".join([f"- **{k}:** {v}" for k, v in metrics.items()])
    return f"### Abnormal Detection Metrics\n{md}", metrics_df
  except Exception as e:
    return f"Error reading detection CSV: {e}", pd.DataFrame()

def evaluate_ocr_pair(reference_text, predicted_text):
  metrics = compute_ocr_text_metrics(reference_text, predicted_text)
  readability = compute_readability_metrics(predicted_text)
  result = {**metrics, **readability}
  result_df = pd.DataFrame([result])
  md = "\n".join([f"- **{k}:** {v}" for k, v in result.items()])
  return f"### OCR / Text Quality Metrics\n{md}", result_df

print("Config and helper functions loaded successfully")

XRAY_ANALYSIS_PROMPT = """
You are an AI medical image assistant for educational purposes only.

Analyze the uploaded X-ray image carefully and return a STRICT JSON object only.
The uploaded X-ray may show ANY body part. Do not limit yourself to a few examples.
Supported examples include: chest/lungs, ribs, clavicle, shoulder, humerus/arm, elbow, forearm, wrist, hand/fingers, hip/pelvis, femur/thigh, knee, tibia/fibula/leg, ankle, foot/toes, cervical/thoracic/lumbar spine, skull/head, face/sinuses, jaw/dental, abdomen/KUB, whole limb, or an unknown X-ray area.

Use a 3-layer workflow:

Layer 1 safety classifier:
- Identify body part, view, image quality, and whether doctor/radiologist review is recommended.
- Do not diagnose. This is an educational triage-style review.
- Use "Needs attention" when ANY abnormal-looking or uncertain important finding is present.
- Use "Limited / unclear image" when body part, view, or image quality is not reliable.
- Use "No obvious acute abnormality" ONLY when body part and view are clear, the image is reviewable, and no body-part-specific abnormality is seen.

Layer 2 body-part-specific finding extractor:
First detect the body part and view. Then use the matching checklist. If the body part is not listed, use the universal X-ray checklist.

Universal X-ray checklist for ANY body part:
- visible anatomy/body part and view
- fracture-like line, bone break, bone fragment, or deformity
- dislocation, subluxation, joint malalignment, or abnormal spacing
- visible metal hardware: pins, screws, wires, plates, rods, nails, implants
- swelling, abnormal soft-tissue shadow, foreign body, or unclear abnormal area
- image quality limitations and uncertainty

Chest/lungs/ribs/clavicle checklist:
- lung opacity / white patch / focal shadow
- upper lung or apical shadow
- cavity-like dark area
- pleural effusion or fluid-like lower chest opacity
- pneumothorax or abnormal absent lung markings
- heart/mediastinum size or shift, if visible
- ribs/clavicles/shoulders for obvious fracture or deformity
- tubes, lines, devices, or surgical material
Important chest rule: if any focal opacity, patchy white area, asymmetric lung shadow, cavity-like area, effusion, pneumothorax-like sign, or uncertain lung abnormality is seen, status must be "Needs attention". Do not call it normal.

Bone/joint X-rays checklist:
- For knee: patella/kneecap, distal femur, tibia, fibula, joint alignment, fracture line, metal hardware, swelling.
- For hand/wrist/fingers: phalanges, metacarpals, carpal bones, radius/ulna, fracture line, dislocation, joint alignment.
- For foot/ankle/toes: tarsal bones, metatarsals, phalanges, tibia/fibula near ankle, talus/calcaneus, fracture line, dislocation, swelling.
- For shoulder/elbow/arm/forearm: humerus, radius, ulna, joint alignment, fracture line, dislocation, hardware.
- For hip/pelvis/femur/leg: pelvis, hip joints, femur, tibia/fibula, fracture line, joint alignment, hardware.
- For spine/neck/back: vertebral alignment, compression deformity, disc spacing, abnormal curve, hardware.
- For skull/face/jaw/dental: skull/facial/jaw/dental bones, sinus/jaw area, fracture-like line, alignment, dental hardware.
- For abdomen/KUB: visible abdomen field, bowel gas/stool pattern if visible, calcification/foreign body/device if visible; do not diagnose obstruction or stone.

Layer 3 safe report writer:
- Convert findings into simple patient-friendly bullet-style wording.
- Never say confirmed diagnosis, successful surgery, healed fracture, exact infection, TB, cancer, pneumonia, or exact hardware type unless a radiology report explicitly says it.
- Use cautious words: "appears", "possible", "may represent", "needs review", "should be confirmed".
- Keep every statement related to the actual uploaded X-ray body part. Do not mention lungs for non-chest images. Do not mention bones/joints only for chest unless ribs/clavicle/shoulder are visible and relevant.

Required JSON keys:
{
 "exam_type": "",
 "body_part": "",
 "view": "",
 "status": "",
 "layer_1_safety": {
  "body_part": "",
  "view": "",
  "image_quality": "",
  "status": "",
  "doctor_review_required": true,
  "urgency_reason": ""
 },
 "layer_2_findings": {
  "visible_anatomy": [],
  "hardware_present": null,
  "hardware_description": "",
  "possible_findings": [],
  "uncertainty": [],
  "confidence": {
   "body_part": 0.0,
   "view": 0.0,
   "hardware": 0.0,
   "fracture_or_abnormality": 0.0
  }
 },
 "layer_3_report": {
  "safe_impression": "",
  "patient_friendly_summary": "",
  "what_to_ask_doctor": []
 },
 "overall_impression": "",
 "key_findings": ["", "", ""],
 "simple_explanation": "",
 "caution": "",
 "raw_note": ""
}

Rules:
1. exam_type must match the image, for example Chest X-ray, Knee X-ray, Hand X-ray, Wrist X-ray, Arm X-ray, Elbow X-ray, Shoulder X-ray, Spine X-ray, Skull X-ray, Foot X-ray, Ankle X-ray, Hip/Pelvis X-ray, Abdomen/KUB X-ray, Dental/Jaw X-ray, or X-ray image.
2. body_part must be short and image-specific. Use Unknown only when the body part is truly unclear.
3. view should be ONE projection for the uploaded image: PA, AP, Lateral, Frontal, Oblique, or Unknown. Do not use combined labels such as AP / Lateral view unless multiple separate X-ray views are visible in the same image.
4. status must be one of: "No obvious acute abnormality", "Needs attention", or "Limited / unclear image".
5. Normal-output blocker: Never return "No obvious acute abnormality" if there is hardware, possible fracture, dislocation, abnormal alignment, focal opacity/shadow, cavity-like area, effusion, pneumothorax-like sign, swelling, deformity, abnormal soft-tissue shadow, poor image quality, low confidence, or important uncertainty.
6. key_findings must contain 3 to 6 useful, body-part-related findings in simple English. Do not repeat the same idea.
7. For chest X-rays, include chest/lung-specific observations if visible. If a white patch/opacity/shadow is visible, say it needs doctor/radiologist review, but do not diagnose TB/pneumonia/cancer.
8. For non-chest X-rays, do not mention lungs, air spaces, opacity, or heart unless the image actually shows chest anatomy.
9. Avoid weak generic findings such as "bright white areas are visible" unless you explain why that matters for that body part.
10. simple_explanation must be easy for a non-medical person and should not repeat the key_findings list.
11. caution must clearly say this is educational only and not a confirmed diagnosis.
12. raw_note can mention image quality or uncertainty in one short line.
13. Return JSON only. No markdown, no code fences, no extra text.

Example safe knee wording:
"This appears to be a lateral knee X-ray. Metal repair hardware is seen near the kneecap/knee region. This may represent possible prior patellar fracture fixation. A doctor or X-ray specialist should confirm the exact finding, hardware type, and bone-healing status."

Example safe chest wording:
"This appears to be a frontal chest X-ray. A visible patchy white shadow is seen in part of the lung area. This may be due to several causes and needs doctor or X-ray specialist review. This is not a diagnosis."
"""

XRAY_STUDY_METADATA_PROMPT = """
You are a careful medical X-ray triage assistant. Return strict JSON only.

Look at this uploaded image and classify the study at a high level.

Return a JSON object with exactly these keys:
- is_xray: true/false
- body_part: one short label such as Chest, Hand/Wrist, Forearm, Elbow, Shoulder, Knee, Leg, Foot/Ankle, Hip/Pelvis, Spine, Skull, Abdomen, Unknown body area
- body_part_confidence: number from 0 to 1
- view: one short single-view label such as Frontal, PA, AP, Lateral, Oblique, Axial, Mortise, Sunrise/Merchant, or Single view - projection uncertain
- image_quality: one of Acceptable, Blurred, Limited / unclear image
- blur_flag: true/false
- visible_anatomy: short list of visible anatomy names
- notes: one short sentence

Rules:
- Focus first on the body part and study type.
- Do not diagnose the full disease here.
- If the image is blurred or too unclear, set image_quality to Blurred or Limited / unclear image and blur_flag to true.
- Be conservative. If unsure, use Unknown body area or Single view - projection uncertain.
"""



def create_evaluation_pack():
  base_dir = os.path.join(tempfile.gettempdir(), f"fyp_eval_pack_{uuid.uuid4().hex}")
  os.makedirs(base_dir, exist_ok=True)

  classification_template = pd.DataFrame([
    {"y_true": "Laboratory Report", "y_pred": "Laboratory Report"},
    {"y_true": "Radiology Report", "y_pred": "Radiology Report"},
    {"y_true": "X-ray Image", "y_pred": "X-ray Image"},
    {"y_true": "Laboratory Report", "y_pred": "Radiology Report"},
    {"y_true": "Radiology Report", "y_pred": "Laboratory Report"},
  ])
  detection_template = pd.DataFrame([
    {"y_true": "High", "y_pred": "High"},
    {"y_true": "Normal", "y_pred": "Normal"},
    {"y_true": "Low", "y_pred": "Low"},
    {"y_true": "High", "y_pred": "Normal"},
    {"y_true": "Low", "y_pred": "Low"},
  ])
  manual_validation = pd.DataFrame([
    {
      "file_name": "",
      "actual_type": "",
      "predicted_type": "",
      "ocr_ok_yes_no": "",
      "lab_values_ok_yes_no": "",
      "abnormal_flags_ok_yes_no": "",
      "radiology_summary_ok_yes_no": "",
      "xray_review_ok_yes_no": "",
      "ai_explanation_clear_yes_no": "",
      "notes": "",
    }
    for _ in range(10)
  ])

  ocr_example_text = (
    "GROUND TRUTH TEXT\n"
    "Glucose 110 mg/dL 70-100\n"
    "Hemoglobin 10.2 g/dL 12.0-15.0\n"
    "Platelets 300 x10^9/L 150-450\n\n"
    "EXTRACTED / PREDICTED TEXT\n"
    "Glucose 110 mg/dL 70-100\n"
    "Hemoglobin 10.2 g/dL 12.0-15.0\n"
    "Platelets 300 x10^9/L 150-450\n"
  )

  results_template = """

# Final Evaluation Results Template

## OCR Evaluation
- Number of reports tested:
- OCR similarity score range:
- Main OCR issues observed:

## Report Classification
- Number of samples tested:
- Accuracy:
- Macro Precision:
- Macro Recall:
- Macro F1:

## Abnormal Detection
- Number of samples tested:
- Accuracy:
- Macro Precision:
- Macro Recall:
- Macro F1:

## Manual Validation Summary
- Explanation clarity:
- Report parsing quality:
- Main limitations observed:

## Short Conclusion
Write 4 to 6 lines summarizing how well the system performed and where it still needs improvement.
"""

  files = {
    "classification_template.csv": classification_template,
    "detection_template.csv": detection_template,
    "manual_validation_checklist.csv": manual_validation,
  }

  for filename, df in files.items():
    df.to_csv(os.path.join(base_dir, filename), index=False)

  with open(os.path.join(base_dir, "ocr_example.txt"), "w", encoding="utf-8") as f:
    f.write(ocr_example_text)

  with open(os.path.join(base_dir, "final_results_template.md"), "w", encoding="utf-8") as f:
    f.write(results_template)

  zip_path = os.path.join(tempfile.gettempdir(), f"fyp_evaluation_pack_{uuid.uuid4().hex}.zip")
  with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for filename in ["classification_template.csv", "detection_template.csv", "manual_validation_checklist.csv", "ocr_example.txt", "final_results_template.md"]:
      full_path = os.path.join(base_dir, filename)
      zf.write(full_path, arcname=filename)

  return zip_path

def encode_image_to_data_url(image_path):
  mime_type, _ = mimetypes.guess_type(image_path)
  if mime_type is None:
    mime_type = "image/jpeg"

  with open(image_path, "rb") as f:
    image_bytes = f.read()

  encoded = base64.b64encode(image_bytes).decode("utf-8")
  return f"data:{mime_type};base64,{encoded}"


def extract_json_object(text):
  text = safe_str(text).strip()
  if not text:
    return {}
  if text.startswith("{") and text.endswith("}"):
    try:
      return json.loads(text)
    except Exception:
      pass
  match = re.search(r"\{[\s\S]*\}", text)
  if match:
    candidate = match.group(0)
    try:
      return json.loads(candidate)
    except Exception:
      return {}
  return {}


# ============================================================
# X-RAY 3-LAYER NORMALIZATION HELPERS
# Layer 1: safety classifier
# Layer 2: detailed finding extractor
# Layer 3: safe report writer
# ============================================================

def _xray_as_list(value, max_items=12):
  """Coerce strings/lists into a clean list for stable JSON + UI rendering."""
  if value is None:
    return []
  if isinstance(value, list):
    raw_items = value
  elif isinstance(value, tuple):
    raw_items = list(value)
  elif isinstance(value, str):
    # Split only when the string looks like a semicolon/newline list.
    raw_items = re.split(r"\s*(?:\n|;|\u2022)\s*", value) if (";" in value or "\n" in value or "" in value) else [value]
  else:
    raw_items = [safe_str(value)]

  cleaned, seen = [], set()
  for item in raw_items:
    item = safe_str(item).strip(" -\n\t")
    if not item:
      continue
    key = re.sub(r"\s+", " ", item.lower())
    if key in seen:
      continue
    seen.add(key)
    cleaned.append(item)
    if len(cleaned) >= max_items:
      break
  return cleaned


def _xray_bool_or_none(value):
  if isinstance(value, bool):
    return value
  if value is None:
    return None
  value_l = safe_str(value).strip().lower()
  if value_l in {"true", "yes", "present", "seen", "visible", "1"}:
    return True
  if value_l in {"false", "no", "absent", "not seen", "0"}:
    return False
  return None


def _xray_confidence_value(value, default=0.0):
  try:
    v = float(value)
    if v > 1.0:
      v = v / 100.0
    return round(max(0.0, min(1.0, v)), 2)
  except Exception:
    return default


def _default_xray_visible_anatomy(body_part):
  bp = safe_str(body_part).lower()
  if "knee" in bp:
    return ["distal femur", "proximal tibia", "fibula", "patella"]
  if "hand" in bp or "wrist" in bp:
    return ["metacarpals", "phalanges", "carpal bones", "distal radius/ulna"]
  if "foot" in bp or "ankle" in bp:
    return ["tibia/fibula near ankle", "talus", "calcaneus", "metatarsals/phalanges"]
  if "elbow" in bp:
    return ["distal humerus", "proximal radius", "proximal ulna", "elbow joint"]
  if "shoulder" in bp:
    return ["humeral head", "shoulder joint region", "clavicle/scapula region"]
  if any(x in bp for x in ["arm", "forearm"]):
    return ["long bones of the arm/forearm", "nearby joint region"]
  if any(x in bp for x in ["hip", "pelvis"]):
    return ["pelvis/hip joint region", "proximal femur"]
  if "chest" in bp or "lung" in bp:
    return ["lungs", "ribs", "heart/mediastinum", "diaphragm"]
  if "spine" in bp:
    return ["vertebral bodies", "spinal alignment region"]
  if "skull" in bp or "head" in bp:
    return ["skull bones"]
  return []


def _xray_safe_phrase(text):
  """Reduce overconfident medical wording without hiding useful findings."""
  s = safe_str(text).strip()
  if not s:
    return ""

  replacements = [
    (r"\bsuccessfully repaired\b", "appears to show prior repair/fixation"),
    (r"\bsuccessful repair\b", "apparent prior repair/fixation"),
    (r"(?<!not a )\bconfirmed diagnosis\b", "possible finding"),
    (r"\bconfirmed\s+(fracture|dislocation|break|tear|infection|pneumonia|effusion|mass|lesion|abnormality)\b", r"possible \1"),
    (r"\bis confirmed as\b", "may represent"),
    (r"\bdefinitely\b", "possibly"),
    (r"\bdiagnostic of\b", "suggestive of"),
    (r"\bis diagnostic for\b", "is suggestive of"),
    (r"\bproves\b", "may suggest"),
    (r"\bhealed fracture\b", "possible healing/prior fracture change"),
    (r"\bORIF\b", "possible internal fixation"),
    (r"\btension band wiring\b", "possible tension-band type wiring/fixation"),
  ]
  for pattern, repl in replacements:
    s = re.sub(pattern, repl, s, flags=re.IGNORECASE)

  # Repair grammar from older outputs that over-softened clinician-confirmation wording.
  s = re.sub(r"should\s+be\s+suggestive\s+by\s+(a\s+)?(clinician|doctor|radiologist)", r"should be confirmed by \1\2", s, flags=re.IGNORECASE)
  s = re.sub(r"should\s+be\s+suggestive\s+clinically", "should be confirmed clinically", s, flags=re.IGNORECASE)
  s = re.sub(r"should\s+be\s+suggested\s+by\s+(a\s+)?(clinician|doctor|radiologist)", r"should be confirmed by \1\2", s, flags=re.IGNORECASE)

  # If a sentence starts with a hard diagnosis, soften it.
  s = re.sub(r"\bThis is a ([^.]{0,60}?fracture)", r"This may represent a \1", s, flags=re.IGNORECASE)
  s = re.sub(r"\bThe image shows a ([^.]{0,60}?fracture)", r"The image appears to show a \1", s, flags=re.IGNORECASE)
  return s




def _xray_dedupe_sentences(text, max_sentences=None):
  """Remove repeated sentence-level content from generated X-ray explanations."""
  raw = safe_str(text).strip()
  if not raw:
    return ""
  parts = re.split(r"(?<=[.!?])\s+", raw)
  cleaned = []
  seen = set()
  for part in parts:
    part = safe_str(part).strip()
    if not part:
      continue
    key = re.sub(r"[^a-z0-9]+", " ", part.lower()).strip()
    # Ignore very short keys so useful short safety sentences are preserved.
    if key and len(key) > 18 and key in seen:
      continue
    seen.add(key)
    cleaned.append(part)
    if max_sentences and len(cleaned) >= max_sentences:
      break
  return " ".join(cleaned).strip()


def _xray_alignment_attention_phrase(view):
  """Universal alignment wording that does not imply AP side-to-side comparison for a single lateral/oblique view."""
  v = safe_str(view).lower()
  if any(x in v for x in ["lateral", "oblique", "single", "unknown", "uncertain"]):
    return "Alignment check: bone/joint alignment is limited on this single view; a clinician should review the original X-ray if injury, pain, or positioning concern is present."
  return "Alignment check: side-to-side appearance or positioning looks uneven; a clinician should review the original X-ray if injury is suspected."


def _xray_clean_alignment_text(text, view):
  """Replace generic side-comparison alignment text when a single-view X-ray is being summarized."""
  s = safe_str(text).strip()
  if not s:
    return ""
  if re.search(r"alignment check:.*one side or area looks uneven compared with the other", s, flags=re.IGNORECASE):
    return _xray_alignment_attention_phrase(view)
  if re.search(r"one side or area looks uneven compared with the other", s, flags=re.IGNORECASE):
    return re.sub(
      r"one side or area looks uneven compared with the other,?\s*so a doctor should check for injury or positioning issues\.?",
      _xray_alignment_attention_phrase(view).replace("Alignment check: ", ""),
      s,
      flags=re.IGNORECASE,
    )
  return s


def _xray_refine_possible_finding_text(text, body_part):
  """Make finding labels more specific when the body part supports it, without overdiagnosing."""
  s = safe_str(text).strip()
  if not s:
    return ""
  bp = safe_str(body_part).lower()
  low = s.lower()
  if "knee" in bp and "fixation" in low and ("patella" in low or "kneecap" in low or "knee" in low):
    return "possible prior patellar fracture fixation"
  return s


def _xray_clean_generated_text(text, view="", body_part="", max_sentences=None, refine_possible=False):
  """Apply safe wording, grammar repair, alignment cleanup, and light de-duplication."""
  s = _xray_safe_phrase(text)
  s = _xray_clean_alignment_text(s, view)
  if refine_possible:
    s = _xray_refine_possible_finding_text(s, body_part)
  s = _xray_dedupe_sentences(s, max_sentences=max_sentences)
  return s

def _xray_unique_extend(base_items, new_items, max_items=8):
  items = _xray_as_list(base_items, max_items=20)
  seen = {re.sub(r"[^a-z0-9]+", " ", x.lower()).strip() for x in items}
  for item in _xray_as_list(new_items, max_items=20):
    key = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()
    if not key or key in seen:
      continue
    seen.add(key)
    items.append(item)
    if len(items) >= max_items:
      break
  return items


# ============================================================
# UNIVERSAL X-RAY CHECKLIST + SAFETY GATE HELPERS
# These helpers keep the app universal. Known body parts get related
# checklist wording; unknown body parts fall back to safe generic X-ray
# triage instead of pretending the image is normal.
# ============================================================

XRAY_BODY_PART_CHECKLISTS = {
  "chest": [
    "lungs and rib areas", "focal opacity/white patch", "upper-lung shadow",
    "cavity-like area", "fluid/effusion sign", "pneumothorax-like sign",
    "heart/mediastinum outline", "ribs/clavicles", "devices or metal"
  ],
  "ribs": ["ribs", "clavicle/shoulder area", "fracture-like line", "lung edge if visible", "alignment"],
  "clavicle": ["clavicle", "shoulder joint area", "fracture-like line", "alignment", "hardware"],
  "shoulder": ["humeral head", "shoulder joint", "clavicle/scapula", "dislocation", "fracture-like line", "hardware"],
  "elbow": ["distal humerus", "proximal radius/ulna", "elbow joint", "fracture-like line", "joint alignment", "swelling", "hardware"],
  "arm": ["humerus/forearm bones", "nearby joints", "fracture-like line", "alignment", "hardware"],
  "forearm": ["radius", "ulna", "wrist/elbow ends", "fracture-like line", "alignment", "hardware"],
  "wrist": ["distal radius/ulna", "carpal bones", "metacarpal bases", "fracture-like line", "dislocation", "alignment", "hardware"],
  "hand": ["phalanges", "metacarpals", "carpal bones if visible", "fracture-like line", "dislocation", "joint alignment", "hardware"],
  "finger": ["phalanges", "finger joints", "fracture-like line", "dislocation", "alignment"],
  "pelvis": ["pelvis", "hip joints", "proximal femur", "fracture-like line", "alignment", "hardware"],
  "hip": ["hip joint", "femoral head/neck", "pelvis if visible", "fracture-like line", "alignment", "hardware"],
  "femur": ["femur", "hip/knee ends if visible", "fracture-like line", "alignment", "hardware"],
  "knee": ["kneecap/patella", "distal femur", "tibia", "fibula", "joint alignment", "fracture-like line", "swelling", "hardware"],
  "leg": ["tibia", "fibula", "knee/ankle ends if visible", "fracture-like line", "alignment", "hardware"],
  "tibia": ["tibia", "fibula", "nearby joints", "fracture-like line", "alignment", "hardware"],
  "ankle": ["distal tibia/fibula", "talus", "ankle joint", "fracture-like line", "alignment", "swelling", "hardware"],
  "foot": ["tarsal bones", "metatarsals", "toes", "calcaneus if visible", "fracture-like line", "alignment", "hardware"],
  "toe": ["toe bones", "toe joints", "fracture-like line", "dislocation", "alignment"],
  "spine": ["vertebral alignment", "compression deformity", "disc spacing", "abnormal curve", "hardware"],
  "neck": ["cervical spine", "vertebral alignment", "disc spacing", "soft tissue shadow", "hardware"],
  "skull": ["skull bones", "sinus/face area if visible", "fracture-like line", "alignment"],
  "face": ["facial bones", "sinus region", "jaw/orbits if visible", "fracture-like line", "alignment"],
  "sinus": ["sinus spaces", "facial bones", "fluid/opacity if visible", "bone outline"],
  "jaw": ["mandible/maxilla", "teeth/dental hardware", "jaw alignment", "fracture-like line"],
  "dental": ["teeth", "jaw bone", "dental hardware", "bone around teeth"],
  "abdomen": ["abdomen/KUB field", "bowel gas pattern if visible", "calcification/stone-like shadow", "foreign body/device", "spine/pelvis if visible"],
  "kub": ["abdomen/KUB field", "bowel gas pattern if visible", "calcification/stone-like shadow", "foreign body/device", "spine/pelvis if visible"],
}


def _xray_body_family(body_part):
  bp = safe_str(body_part).lower()
  for key in sorted(XRAY_BODY_PART_CHECKLISTS.keys(), key=len, reverse=True):
    if key in bp:
      return key
  if "lung" in bp:
    return "chest"
  if "humerus" in bp:
    return "arm"
  if "radius" in bp or "ulna" in bp:
    return "forearm"
  if "fibula" in bp:
    return "leg"
  if "patella" in bp:
    return "knee"
  return "generic"


def _xray_checklist_for_body_part(body_part):
  family = _xray_body_family(body_part)
  return XRAY_BODY_PART_CHECKLISTS.get(family, [
    "body part and view", "fracture-like line", "bone/joint alignment",
    "metal hardware", "soft-tissue shadow/swelling", "foreign body/device",
    "image quality and uncertainty"
  ])


def _xray_related_checklist_sentence(body_part):
  items = _xray_checklist_for_body_part(body_part)[:5]
  return "Related checklist: " + ", ".join(items) + "."


def _xray_model_text_for_attention(result, layer2=None):
  """Collect only finding/impression text, excluding generic caution text."""
  layer2 = layer2 or {}
  parts = []
  for key in ["overall_impression", "simple_explanation", "raw_note"]:
    val = safe_str(result.get(key, "")) if isinstance(result, dict) else ""
    if val:
      parts.append(val)
  for item in _xray_as_list(result.get("key_findings", []) if isinstance(result, dict) else [], max_items=20):
    parts.append(item)
  for item in _xray_as_list(layer2.get("possible_findings", []), max_items=20):
    parts.append(item)
  hw_desc = safe_str(layer2.get("hardware_description", ""))
  if hw_desc:
    parts.append(hw_desc)
  return " ".join(parts).lower()


def _xray_text_has_attention_finding(text):
  """Text-based normal blocker. Avoids generic caution and tries not to fire on 'no obvious ...'."""
  t = safe_str(text).lower()
  if not t.strip():
    return False

  # Remove common normal/negative phrases so they do not trigger false alerts.
  negatives = [
    r"no obvious [^.]{0,60}", r"not seen", r"not detected", r"absent",
    r"without [^.]{0,60}", r"no clear [^.]{0,60}", r"no large [^.]{0,60}",
  ]
  cleaned = t
  for pat in negatives:
    cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)

  attention_patterns = [
    r"\b(possible|suggestive|suspected|visible|appears|seen|detected|flagged|present)\b.{0,70}\b(fracture|break|dislocation|subluxation|opacity|shadow|white patch|cavity|effusion|pneumothorax|consolidation|infiltrate|collapse|mass|lesion|hardware|deformity|swelling|foreign body)\b",
    r"\b(fracture|break|dislocation|subluxation|opacity|shadow|white patch|cavity|effusion|pneumothorax|consolidation|infiltrate|collapse|mass|lesion|hardware|deformity|swelling|foreign body)\b.{0,70}\b(possible|suggestive|suspected|visible|appears|seen|detected|flagged|present|cannot be ruled out)\b",
    r"\bcannot\s+be\s+ruled\s+out\b",
    r"\b(abnormal|uneven|asymmetric|marked|patchy|focal)\b.{0,60}\b(opacity|shadow|alignment|spacing|swelling|deformity)\b",
  ]
  return any(re.search(pat, cleaned, flags=re.IGNORECASE) for pat in attention_patterns)


def _xray_normal_allowed(body_part, view, status, layer1, layer2, original_result):
  """Strict conditions before allowing a green/normal-looking X-ray result."""
  if status != "No obvious acute abnormality":
    return False
  bp = safe_str(body_part).lower()
  vw = safe_str(view).lower()
  if not bp or "unknown" in bp or "unclear" in bp:
    return False
  if not vw or "unknown" in vw or "unclear" in vw or "uncertain" in vw:
    return False
  image_quality = safe_str(layer1.get("image_quality", "")).lower()
  if any(x in image_quality for x in ["poor", "limited", "unclear", "low"]):
    return False
  if layer2.get("hardware_present") is True:
    return False
  if _xray_as_list(layer2.get("possible_findings", []), max_items=20):
    return False
  if _xray_text_has_attention_finding(_xray_model_text_for_attention(original_result, layer2)):
    return False
  return True


def _xray_force_attention(result, reason, finding=None, possible_finding=None):
  """Safely upgrade a normalized result to Needs attention and keep fields consistent."""
  if not isinstance(result, dict):
    result = {}
  result["status"] = "Needs attention"
  layer1 = result.get("layer_1_safety", {}) if isinstance(result.get("layer_1_safety", {}), dict) else {}
  layer1["status"] = "Needs attention"
  layer1["doctor_review_required"] = True
  layer1["urgency_reason"] = _xray_clean_generated_text(reason, view=result.get("view", ""), body_part=result.get("body_part", ""), max_sentences=1)
  result["layer_1_safety"] = layer1

  if finding:
    findings = _xray_as_list(result.get("key_findings", []), max_items=20)
    findings = _xray_unique_extend(findings, [finding], max_items=8)
    result["key_findings"] = findings

  if possible_finding:
    layer2 = result.get("layer_2_findings", {}) if isinstance(result.get("layer_2_findings", {}), dict) else {}
    layer2["possible_findings"] = _xray_unique_extend(layer2.get("possible_findings", []), [possible_finding], max_items=10)
    conf = layer2.get("confidence", {}) if isinstance(layer2.get("confidence", {}), dict) else {}
    conf["fracture_or_abnormality"] = max(_xray_confidence_value(conf.get("fracture_or_abnormality", 0.0)), 0.55)
    layer2["confidence"] = conf
    result["layer_2_findings"] = layer2

  return result


def _normalize_xray_layer1(result, exam_type, body_part, view, status):
  layer = result.get("layer_1_safety", {}) if isinstance(result, dict) else {}
  if not isinstance(layer, dict):
    layer = {}

  doctor_review = layer.get("doctor_review_required", None)
  if doctor_review is None:
    doctor_review = status in {"Needs attention", "Limited / unclear image"}
  else:
    doctor_review = bool(_xray_bool_or_none(doctor_review))

  image_quality = safe_str(layer.get("image_quality", "")).strip()
  if not image_quality:
    image_quality = "limited/unclear" if status == "Limited / unclear image" else "reviewable"

  urgency_reason = safe_str(layer.get("urgency_reason", "")).strip()
  if not urgency_reason:
    if status == "Needs attention":
      urgency_reason = "One or more visual findings or uncertainties should be reviewed by a clinician."
    elif status == "Limited / unclear image":
      urgency_reason = "The image is limited, so a clinician should review the original image or repeat imaging if needed."
    else:
      urgency_reason = "No obvious urgent-looking issue was detected by this educational review, but symptoms still matter."

  return {
    "body_part": safe_str(layer.get("body_part", body_part)).strip() or body_part,
    "view": safe_str(layer.get("view", view)).strip() or view,
    "image_quality": image_quality,
    "status": safe_str(layer.get("status", status)).strip() or status,
    "doctor_review_required": doctor_review,
    "urgency_reason": _xray_safe_phrase(urgency_reason),
  }


def _normalize_xray_layer2(result, body_part):
  layer = result.get("layer_2_findings", {}) if isinstance(result, dict) else {}
  if not isinstance(layer, dict):
    layer = {}

  visible_anatomy = _xray_as_list(layer.get("visible_anatomy") or result.get("visible_anatomy"), max_items=10)
  if not visible_anatomy:
    visible_anatomy = _default_xray_visible_anatomy(body_part)

  hardware_present = _xray_bool_or_none(layer.get("hardware_present", result.get("hardware_present")))
  hardware_description = safe_str(layer.get("hardware_description", result.get("hardware_description", ""))).strip()
  if hardware_present is True and not hardware_description:
    hardware_description = "metallic orthopedic hardware appears visible; exact type should be confirmed clinically"
  elif hardware_present is False and not hardware_description:
    hardware_description = "no obvious metallic orthopedic hardware is described by this review"
  elif hardware_present is None and not hardware_description:
    hardware_description = "hardware presence is uncertain from this image"

  possible_findings = _xray_as_list(layer.get("possible_findings") or result.get("possible_findings"), max_items=10)
  uncertainty = _xray_as_list(layer.get("uncertainty") or result.get("uncertainty"), max_items=10)

  confidence = layer.get("confidence", result.get("confidence", {}))
  if not isinstance(confidence, dict):
    confidence = {}
  confidence = {
    "body_part": _xray_confidence_value(confidence.get("body_part", confidence.get("body", 0.0))),
    "view": _xray_confidence_value(confidence.get("view", 0.0)),
    "hardware": _xray_confidence_value(confidence.get("hardware", 0.0)),
    "fracture_or_abnormality": _xray_confidence_value(confidence.get("fracture_or_abnormality", confidence.get("abnormality", 0.0))),
  }

  # Add a conservative uncertainty by default so the output never sounds like a diagnosis.
  if not uncertainty:
    uncertainty = ["The exact diagnosis, fracture status, and clinical meaning cannot be confirmed from this educational image review alone."]

  return {
    "visible_anatomy": [_xray_clean_generated_text(x, body_part=body_part) for x in visible_anatomy],
    "hardware_present": hardware_present,
    "hardware_description": _xray_clean_generated_text(hardware_description, body_part=body_part),
    "possible_findings": [_xray_clean_generated_text(x, body_part=body_part, refine_possible=True) for x in possible_findings],
    "uncertainty": [_xray_clean_generated_text(x, body_part=body_part) for x in uncertainty],
    "confidence": confidence,
  }


def _build_xray_layer3_report(exam_type, body_part, view, status, layer1, layer2, overall_impression, simple_explanation):
  visible = layer2.get("visible_anatomy", []) or []
  possible = layer2.get("possible_findings", []) or []
  uncertainty = layer2.get("uncertainty", []) or []
  hardware_present = layer2.get("hardware_present")
  hardware_desc = safe_str(layer2.get("hardware_description", "")).strip()

  body_area = _plain_body_area(body_part) if "_plain_body_area" in globals() else (body_part or "the imaged area")
  view_text = view if view and "unknown" not in safe_str(view).lower() else "uploaded view"

  if hardware_present is True:
    # Build this sentence from the hardware description so the summary does
    # not repeat "metallic hardware" twice. This stays universal for all
    # bone/joint X-rays and still avoids a confirmed diagnosis.
    hardware_sentence = hardware_desc.rstrip(" .")
    safe_parts = [f"This appears to be a {view_text} {body_area} X-ray."]
    if hardware_sentence:
      safe_parts.append(f"{hardware_sentence}.")
    else:
      safe_parts.append("Metal hardware appears visible in the X-ray area.")
    if possible:
      safe_parts.append(f"Possible finding: {possible[0]}.")
    safe_parts.append("The exact type, position, and bone-healing status should be confirmed by a clinician.")
    safe_impression = " ".join(safe_parts)
  elif possible:
    safe_impression = (
      f"This appears to be a {view_text} {body_area} X-ray. "
      f"Possible finding: {possible[0]}. A qualified clinician should confirm the result."
    )
  else:
    safe_impression = overall_impression or f"This appears to be a {body_area} X-ray that should be interpreted by a qualified clinician."

  patient_summary_parts = []
  if visible:
    patient_summary_parts.append("Visible anatomy includes " + ", ".join(visible[:4]) + ".")
  if hardware_present is True:
    patient_summary_parts.append("Metal hardware appears visible, which can be seen after prior fracture or joint surgery.")
  elif hardware_present is False:
    patient_summary_parts.append("No obvious metal hardware was described by this educational review.")
  if possible:
    patient_summary_parts.append("The review flagged: " + "; ".join(possible[:3]) + ".")
  if uncertainty:
    patient_summary_parts.append("Uncertainty: " + uncertainty[0])
  patient_friendly_summary = " ".join(patient_summary_parts) or simple_explanation or safe_impression

  questions = []
  if hardware_present is True:
    questions.append("What type of hardware is present and is it in expected position?")
  if possible:
    questions.append("Is there a fracture, healing fracture, or post-operative change that needs treatment?")
  questions.append("Do my symptoms and physical exam match the X-ray findings?")

  return {
    "safe_impression": _xray_clean_generated_text(safe_impression, view=view, body_part=body_part, max_sentences=3),
    "patient_friendly_summary": _xray_clean_generated_text(patient_friendly_summary, view=view, body_part=body_part, max_sentences=5),
    "what_to_ask_doctor": _xray_as_list(questions, max_items=5),
  }


def _format_xray_layer2_for_text(layer2):
  visible = ", ".join(layer2.get("visible_anatomy", [])[:6]) or "not specified"
  hardware = layer2.get("hardware_description", "") or "not specified"
  possible = "; ".join(layer2.get("possible_findings", [])[:5]) or "no specific possible finding listed"
  uncertainty = "; ".join(layer2.get("uncertainty", [])[:4]) or "not specified"
  return (
    f"Visible anatomy: {visible}\n"
    f"Hardware check: {hardware}\n"
    f"Possible findings: {possible}\n"
    f"Uncertainty: {uncertainty}"
  )

def normalize_xray_result(result):
  """Normalize X-ray output while preserving the new 3-layer structure.

  Backward compatible with older model responses that only returned the original
  flat keys. The app UI/PDF can keep using exam_type/body_part/key_findings,
  while the richer layer_* fields add specificity and safer wording.
  """
  if not isinstance(result, dict):
    result = {}

  exam_type = safe_str(result.get("exam_type", "")).strip() or "X-ray image"
  body_part = safe_str(result.get("body_part", "")).strip() or "Unknown"
  view = _sanitize_xray_view_label(safe_str(result.get("view", "")).strip() or "Unknown", body_part)
  status = safe_str(result.get("status", "")).strip() or "Limited / unclear image"
  overall_impression = safe_str(result.get("overall_impression", "")).strip() or "No structured X-ray impression could be generated."
  simple_explanation = safe_str(result.get("simple_explanation", "")).strip() or overall_impression
  caution = safe_str(result.get("caution", "")).strip() or "This is educational only and not a confirmed diagnosis."
  raw_note = safe_str(result.get("raw_note", "")).strip()

  if status not in {"No obvious acute abnormality", "Needs attention", "Limited / unclear image"}:
    lowered = status.lower()
    if "normal" in lowered or "no obvious" in lowered:
      status = "No obvious acute abnormality"
    elif "unclear" in lowered or "limited" in lowered or "poor" in lowered:
      status = "Limited / unclear image"
    else:
      status = "Needs attention"

  layer1 = _normalize_xray_layer1(result, exam_type, body_part, view, status)
  # Keep top-level and layer-level identity aligned after any local correction.
  layer1["body_part"] = body_part
  layer1["view"] = view
  layer1["status"] = status

  layer2 = _normalize_xray_layer2(result, body_part)

  # Universal normal-output blocker. A green/normal result is allowed only when
  # the body part + view are clear and no body-part-specific attention finding
  # is present. This applies to ANY X-ray body area.
  attention_text_found = _xray_text_has_attention_finding(_xray_model_text_for_attention(result, layer2))
  if status == "No obvious acute abnormality" and (
    layer2.get("hardware_present") is True or layer2.get("possible_findings") or attention_text_found
  ):
    status = "Needs attention"
    layer1["status"] = status
    layer1["doctor_review_required"] = True
    layer1["urgency_reason"] = "Specific visible findings, abnormal-looking image patterns, or hardware should be reviewed by a qualified clinician."
  elif status == "No obvious acute abnormality" and not _xray_normal_allowed(body_part, view, status, layer1, layer2, result):
    status = "Limited / unclear image"
    layer1["status"] = status
    layer1["doctor_review_required"] = True
    layer1["urgency_reason"] = "The body part, view, or image quality is not clear enough for a confident normal-style educational summary."

  provided_layer3 = result.get("layer_3_report", {}) if isinstance(result.get("layer_3_report", {}), dict) else {}
  layer3 = _build_xray_layer3_report(
    exam_type=exam_type,
    body_part=body_part,
    view=view,
    status=status,
    layer1=layer1,
    layer2=layer2,
    overall_impression=overall_impression,
    simple_explanation=simple_explanation,
  )
  if provided_layer3:
    layer3["safe_impression"] = _xray_clean_generated_text(
      safe_str(provided_layer3.get("safe_impression", layer3["safe_impression"])).strip() or layer3["safe_impression"],
      view=view,
      body_part=body_part,
      max_sentences=3,
    )
    layer3["patient_friendly_summary"] = _xray_clean_generated_text(
      safe_str(provided_layer3.get("patient_friendly_summary", layer3["patient_friendly_summary"])).strip() or layer3["patient_friendly_summary"],
      view=view,
      body_part=body_part,
      max_sentences=5,
    )
    layer3["what_to_ask_doctor"] = _xray_as_list(provided_layer3.get("what_to_ask_doctor", layer3["what_to_ask_doctor"]), max_items=5)

  key_findings = result.get("key_findings", [])
  if not isinstance(key_findings, list):
    key_findings = [safe_str(key_findings)]
  key_findings = [_xray_safe_phrase(x).strip(" -\n\t") for x in key_findings if safe_str(x).strip()]
  if not key_findings:
    key_findings = ["No clear structured findings were returned by the model."]

  # Add a body-part-related checklist note so each X-ray report stays tied to
  # the uploaded anatomy, including less common X-ray types.
  checklist_sentence = _xray_related_checklist_sentence(body_part)
  if checklist_sentence and not any("related checklist" in x.lower() for x in key_findings):
    key_findings = _xray_unique_extend(key_findings, [checklist_sentence], max_items=8)

  # Add layer-2 detail to the main findings so the dashboard/PDF is more specific.
  anatomy = layer2.get("visible_anatomy", [])
  if anatomy and not any("visible anatomy" in x.lower() or "body part" in x.lower() for x in key_findings):
    key_findings = _xray_unique_extend(
      [f"Visible anatomy check: {', '.join(anatomy[:4])} are visible or expected in this view."],
      key_findings,
      max_items=8,
    )

  hardware_present = layer2.get("hardware_present")
  hardware_desc = safe_str(layer2.get("hardware_description", "")).strip()
  if hardware_present is True and hardware_desc and not any("hardware" in x.lower() or "metal" in x.lower() for x in key_findings):
    key_findings = _xray_unique_extend(
      key_findings,
      [f"Hardware check: {hardware_desc}."],
      max_items=8,
    )
  elif hardware_present is False and not any("hardware" in x.lower() or "metal" in x.lower() for x in key_findings):
    # Keep this lower priority; do not crowd out real abnormal findings.
    key_findings = _xray_unique_extend(key_findings, ["Hardware check: no obvious metallic fixation hardware was described by this review."], max_items=8)

  if layer2.get("possible_findings"):
    key_findings = _xray_unique_extend(
      key_findings,
      [f"Possible finding: {item}" for item in layer2.get("possible_findings", [])],
      max_items=8,
    )

  if layer2.get("uncertainty") and not any("uncertain" in x.lower() or "cannot" in x.lower() for x in key_findings):
    key_findings = _xray_unique_extend(
      key_findings,
      [f"Uncertainty note: {layer2['uncertainty'][0]}"],
      max_items=8,
    )

  overall_impression = _xray_clean_generated_text(layer3.get("safe_impression") or overall_impression, view=view, body_part=body_part, max_sentences=3)
  simple_explanation = _xray_clean_generated_text(layer3.get("patient_friendly_summary") or simple_explanation, view=view, body_part=body_part, max_sentences=5)
  caution = _xray_clean_generated_text(caution, view=view, body_part=body_part, max_sentences=3)
  if "not a confirmed diagnosis" not in caution.lower():
    caution = (caution.rstrip(" .") + ". This is not a confirmed diagnosis.").strip()
  if "consult" not in caution.lower() and "doctor" not in caution.lower() and "radiologist" not in caution.lower():
    caution += " Please consult a qualified doctor or radiologist for medical decisions."

  return {
    "exam_type": exam_type,
    "body_part": body_part,
    "view": view,
    "status": status,
    "layer_1_safety": layer1,
    "layer_2_findings": layer2,
    "layer_3_report": layer3,
    "overall_impression": overall_impression,
    "key_findings": key_findings[:8],
    "simple_explanation": simple_explanation,
    "caution": caution,
    "raw_note": raw_note,
  }


def fallback_xray_result():
  return normalize_xray_result({
    "exam_type": "X-ray image",
    "body_part": "Unknown",
    "view": "Unknown",
    "status": "Limited / unclear image",
    "overall_impression": "A structured X-ray summary could not be generated.",
    "key_findings": ["No structured findings available.", "Please review the uploaded image manually."],
    "simple_explanation": "The system could not produce a reliable structured review for this image.",
    "caution": "This is educational only and not a confirmed diagnosis."
  })

def xray_status_classes(status):
  status = safe_str(status).strip()
  if status == "No obvious acute abnormality":
    return "xray-green", ""
  if status == "Needs attention":
    return "xray-red", "!"
  return "xray-amber", "?"

def build_xray_visual_html(result):
  result = normalize_xray_result(result)
  status_cls, status_icon = xray_status_classes(result["status"])
  findings_html = "".join(
    f'<div class="xray-finding"> {html.escape(item)}</div>'
    for item in result["key_findings"]
  )
  raw_note_html = ""
  if result["raw_note"]:
    raw_note_html = f'<div class="xray-note">{html.escape(result["raw_note"])}</div>'

  return f"""
  <div class="xray-card">
    <div class="xray-head">
      <div>
        <div class="xray-title">X-ray visual summary</div>
        <div class="xray-subtitle">Clean visual explanation of the uploaded image</div>
      </div>
      <div class="xray-status-badge {status_cls}">{status_icon} {html.escape(result["status"])}</div>
    </div>

    <div class="xray-grid">
      <div class="xray-mini-card">
        <div class="xray-mini-label">Exam type</div>
        <div class="xray-mini-value">{html.escape(result["exam_type"])}</div>
      </div>
      <div class="xray-mini-card">
        <div class="xray-mini-label">Body part</div>
        <div class="xray-mini-value">{html.escape(result["body_part"])}</div>
      </div>
      <div class="xray-mini-card">
        <div class="xray-mini-label">View</div>
        <div class="xray-mini-value">{html.escape(result["view"])}</div>
      </div>
      <div class="xray-mini-card">
        <div class="xray-mini-label">Overall impression</div>
        <div class="xray-mini-value">{html.escape(result["overall_impression"])}</div>
      </div>
    </div>

    <div class="xray-section-title">Key findings</div>
    <div class="xray-findings-wrap">{findings_html}</div>

    <div class="xray-section-title">Simple explanation</div>
    <div class="xray-explain-box">{html.escape(result["simple_explanation"])}</div>

    {raw_note_html}

    <div class="xray-disclaimer-box"> {html.escape(result["caution"])}</div>
  </div>
  """

def build_xray_markdown(result):
  result = normalize_xray_result(result)
  layer1 = result.get("layer_1_safety", {}) or {}
  layer2 = result.get("layer_2_findings", {}) or {}
  layer3 = result.get("layer_3_report", {}) or {}

  findings = "\n".join([f"- {item}" for item in result["key_findings"]])
  visible = ", ".join(layer2.get("visible_anatomy", [])[:8]) or "Not specified"
  possible = "\n".join([f"- {item}" for item in layer2.get("possible_findings", [])]) or "- No specific possible finding listed."
  uncertainty = "\n".join([f"- {item}" for item in layer2.get("uncertainty", [])]) or "- Not specified."
  doctor_questions = "\n".join([f"- {item}" for item in layer3.get("what_to_ask_doctor", [])]) or "- Ask whether the X-ray findings match your symptoms and exam."
  hardware_text = layer2.get("hardware_description", "Not specified")
  confidence = layer2.get("confidence", {}) or {}
  confidence_text = ", ".join([f"{k}: {v}" for k, v in confidence.items()]) or "Not available"
  easy_summary = "\n".join([f"- {item}" for item in _xray_easy_summary_bullets(result)])

  return f"""### X-ray quick summary

{easy_summary}

**Exam type:** {result['exam_type']} 
**Body part:** {result['body_part']} 
**View:** {result['view']} 
**Status:** {result['status']} 
**Doctor review required:** {'Yes' if layer1.get('doctor_review_required') else 'Not clearly required by this educational review'} 

### Layer 1 Safety classifier
- Image quality: {layer1.get('image_quality', 'Not specified')}
- Safety status: {layer1.get('status', result['status'])}
- Reason: {layer1.get('urgency_reason', 'Not specified')}

### Layer 2 Detailed finding extractor
- Visible anatomy: {visible}
- Hardware check: {hardware_text}
- Confidence: {confidence_text}

### Possible findings
{possible}

### Key findings
{findings}

### Layer 3 Safe report writer
{layer3.get('patient_friendly_summary', result['simple_explanation'])}

### Questions to ask your doctor
{doctor_questions}

### Uncertainty notes
{uncertainty}

### Important note
{result['caution']}
"""


def _crop_to_xray_content(gray):
  """Crop black borders/background so body-part detection focuses on the X-ray area."""
  try:
    if gray is None or gray.size == 0:
      return gray
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    nonzero = blur[blur > 5]
    if nonzero.size == 0:
      return gray

    # Use a low threshold to keep soft-tissue and bone, while removing black borders.
    threshold = max(18, min(75, int(np.percentile(nonzero, 35))))
    mask = (blur > threshold).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
      return gray
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 0.03 * h * w:
      return gray

    x, y, bw, bh = cv2.boundingRect(largest)
    margin_x = max(6, int(bw * 0.05))
    margin_y = max(6, int(bh * 0.05))
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w, x + bw + margin_x)
    y2 = min(h, y + bh + margin_y)
    cropped = gray[y1:y2, x1:x2]
    if cropped.size == 0 or cropped.shape[0] < 80 or cropped.shape[1] < 80:
      return gray
    return cropped
  except Exception:
    return gray






def assess_xray_image_quality(image_path):
  """Screen for unreadable or heavily blurred X-ray uploads before analysis."""
  try:
    img = cv2.imread(image_path)
    if img is None:
      return {"ok": False, "blurry": True, "image_quality": "Limited / unclear image", "message": "The uploaded image could not be read."}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    focus = _focus_xray_region_from_image(img)
    if focus is not None and getattr(focus, 'size', 0):
      gray = focus
    gray = _crop_to_xray_content(gray)
    if gray is None or gray.size == 0:
      return {"ok": False, "blurry": True, "image_quality": "Limited / unclear image", "message": BLURRY_XRAY_IMAGE_ERROR}

    h, w = gray.shape[:2]
    if min(h, w) < 120:
      return {"ok": False, "blurry": True, "image_quality": "Limited / unclear image", "message": "This X-ray image is too small for a reliable review. Please upload a clearer image."}

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 45, 135)
    edge_density = float(edges.mean()) / 255.0
    contrast = float(gray.std())

    severe_blur = lap_var < 18
    likely_blur = lap_var < 28 and edge_density < 0.028
    low_detail = contrast < 14 and edge_density < 0.03
    blurry = severe_blur or likely_blur or low_detail

    quality = "Acceptable" if not blurry else "Blurred"
    message = "OK" if not blurry else BLURRY_XRAY_IMAGE_ERROR
    return {
      "ok": not blurry,
      "blurry": blurry,
      "image_quality": quality if blurry else "Acceptable",
      "message": message,
      "laplacian_variance": round(lap_var, 2),
      "edge_density": round(edge_density, 4),
      "contrast": round(contrast, 2),
    }
  except Exception as e:
    return {"ok": False, "blurry": True, "image_quality": "Limited / unclear image", "message": f"Image quality screen failed: {e}"}




def detect_annotated_or_composite_xray_image(image_path):
  """Detect red circles/labels or collage-style images that can mislead X-ray body-part analysis."""
  try:
    img = cv2.imread(image_path)
    if img is None:
      return {"is_annotated": False, "reason": ""}

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Strong red marks/circles/arrows are common in teaching images and confuse medical-image review.
    red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([168, 70, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red1, red2)
    red_ratio = float((red_mask > 0).mean())

    # Strong non-grayscale color content is suspicious for annotations.
    saturation_mean = float(hsv[:, :, 1].mean())
    high_sat_ratio = float((hsv[:, :, 1] > 80).mean())

    # Multiple panels: look for a strong vertical or horizontal gap separating two radiographs.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col_dark = (gray < 25).mean(axis=0)
    row_dark = (gray < 25).mean(axis=1)
    vertical_gap = False
    horizontal_gap = False
    if w > 250:
      middle_cols = col_dark[int(w * 0.30): int(w * 0.70)]
      if middle_cols.size and float(middle_cols.max()) > 0.82:
        vertical_gap = True
    if h > 250:
      middle_rows = row_dark[int(h * 0.30): int(h * 0.70)]
      if middle_rows.size and float(middle_rows.max()) > 0.82:
        horizontal_gap = True

    is_annotated = (
      red_ratio > 0.0012
      or (high_sat_ratio > 0.018 and saturation_mean > 18)
      or (vertical_gap and high_sat_ratio > 0.006)
      or (horizontal_gap and high_sat_ratio > 0.006)
    )

    reasons = []
    if red_ratio > 0.0012:
      reasons.append("red circles/marks detected")
    if high_sat_ratio > 0.018 and saturation_mean > 18:
      reasons.append("colored labels/annotations detected")
    if vertical_gap or horizontal_gap:
      reasons.append("possible multiple X-ray panels detected")

    return {
      "is_annotated": bool(is_annotated),
      "reason": "; ".join(reasons),
      "red_ratio": round(red_ratio, 5),
      "high_saturation_ratio": round(high_sat_ratio, 5),
    }
  except Exception as e:
    return {"is_annotated": False, "reason": f"annotation check failed: {e}"}


def build_annotated_xray_result(image_path=None):
  """Safe educational result for annotated/collage X-ray images."""
  return normalize_xray_result({
    "exam_type": "Annotated or composite X-ray image",
    "body_part": "Body part not safely confirmed from annotated image",
    "view": "Limited / annotated image",
    "status": "Limited / unclear image",
    "overall_impression": ANNOTATED_XRAY_IMAGE_ERROR,
    "key_findings": [
      "Image suitability check: this appears to be an annotated, labeled, circled, or multi-panel X-ray image.",
      "AI review check: red circles, labels, or multiple panels can make the system identify the wrong body part.",
      "Recommendation: upload the original single, clear, unmarked X-ray image for a more relevant educational explanation.",
      "Safety note: a doctor or radiologist should review the original X-ray for accurate findings."
    ],
    "simple_explanation": (
      "This looks like a teaching, annotated, labeled, or combined X-ray image rather than one original clinical X-ray. "
      "Because marks, circles, labels, or multiple panels can confuse the AI, this app should not give a detailed finding from this upload. "
      "Please upload the original single unmarked X-ray. This is educational only and not a diagnosis."
    ),
    "caution": "Educational use only. This is not a confirmed diagnosis and does not replace a doctor or radiologist.",
    "layer_1_safety": {
      "body_part": "Uncertain from annotated/composite image",
      "view": "Limited / annotated image",
      "image_quality": "Limited / annotated image",
      "status": "Limited / unclear image",
      "doctor_review_required": True,
      "urgency_reason": "Annotated or multi-panel images can mislead AI interpretation."
    },
    "layer_2_findings": {
      "visible_anatomy": [],
      "hardware_present": None,
      "hardware_description": "Hardware cannot be safely confirmed from an annotated or composite image.",
      "possible_findings": [],
      "uncertainty": [
        "The original unmarked X-ray is needed for a body-part-specific educational review.",
        "Labels, red circles, and multi-panel layouts can cause incorrect body-part detection."
      ],
      "confidence": {"body_part": 0.0, "view": 0.0, "hardware": 0.0, "fracture_or_abnormality": 0.0}
    },
    "layer_3_report": {
      "safe_impression": ANNOTATED_XRAY_IMAGE_ERROR,
      "patient_friendly_summary": "Please upload the original single, clear, unmarked X-ray image.",
      "what_to_ask_doctor": [
        "Can you review the original X-ray image, not the marked teaching image?",
        "Is there a fracture, alignment problem, or other abnormality?"
      ]
    }
  })



def detect_xray_film_photo_or_composite_upload(image_path):
  """Detect phone photos of X-ray films, multi-panel sheets, or strong background/glare.

  These uploads are valid medical-looking images, but they are not reliable for
  single-image body-part classification. They can make the same image return
  Chest once and Spine another time, so Auto-detect should fall back to a
  stable Needs review result.
  """
  try:
    img = cv2.imread(image_path)
    if img is None:
      return {"is_limited_upload": False, "reason": ""}

    h, w = img.shape[:2]
    if h < 120 or w < 120:
      return {"is_limited_upload": False, "reason": ""}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    high_sat_ratio = float((sat > 70).mean())
    sat_mean = float(sat.mean())
    gray_like_ratio = float((sat < 28).mean())

    border = max(8, int(min(h, w) * 0.07))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    border_sat = sat[border_mask]
    border_high_sat_ratio = float((border_sat > 70).mean()) if border_sat.size else 0.0
    border_sat_mean = float(border_sat.mean()) if border_sat.size else 0.0

    phone_photo_background = (
      (high_sat_ratio > 0.025 and sat_mean > 18)
      or (border_high_sat_ratio > 0.030 and border_sat_mean > 22)
      or (gray_like_ratio < 0.90 and high_sat_ratio > 0.012)
    )

    dark_ratio = float((gray < 35).mean())
    edges = cv2.Canny(gray, 55, 150)
    edge_density = float((edges > 0).mean())
    row_dark = (gray < 35).mean(axis=1)
    col_dark = (gray < 35).mean(axis=0)
    horizontal_separator_count = int(np.sum(row_dark > 0.82))
    vertical_separator_count = int(np.sum(col_dark > 0.82))
    many_separators = (
      horizontal_separator_count > max(18, int(h * 0.045))
      and vertical_separator_count > max(12, int(w * 0.025))
      and dark_ratio > 0.22
    )
    complex_film_sheet = phone_photo_background and (edge_density > 0.08 or many_separators)

    limited = bool(phone_photo_background or complex_film_sheet or many_separators)
    reasons = []
    if phone_photo_background:
      reasons.append("photo/background around X-ray film detected")
    if complex_film_sheet or many_separators:
      reasons.append("possible multi-panel X-ray film sheet detected")
    if high_sat_ratio > 0.025 or border_high_sat_ratio > 0.030:
      reasons.append("non-grayscale photo content detected")

    return {
      "is_limited_upload": limited,
      "reason": "; ".join(reasons),
      "high_saturation_ratio": round(high_sat_ratio, 5),
      "border_high_saturation_ratio": round(border_high_sat_ratio, 5),
      "gray_like_ratio": round(gray_like_ratio, 5),
      "edge_density": round(edge_density, 5),
      "dark_ratio": round(dark_ratio, 5),
    }
  except Exception as e:
    return {"is_limited_upload": False, "reason": f"film/composite check failed: {e}"}


def build_xray_film_photo_needs_review_result(image_path=None, screen=None):
  """Stable safe output for photos of physical X-ray films or multi-panel sheets."""
  screen = screen if isinstance(screen, dict) else {}
  reason = safe_str(screen.get("reason", "photo of an X-ray film or multi-panel sheet detected")).strip()
  safe_summary = (
    "This appears to be a photo of an X-ray film or a multi-panel X-ray sheet, not one clear single exported X-ray image. "
    "Because the image includes extra background, glare, or multiple small X-ray panels, the exact body region and view cannot be confirmed reliably. "
    "Please upload one clear single X-ray image or use the official radiology report."
  )
  return normalize_xray_result({
    "exam_type": "X-ray image",
    "body_part": "Needs review",
    "view": "Not confirmed",
    "status": "Limited / unclear image",
    "overall_impression": safe_summary,
    "key_findings": [
      "Image suitability check: this looks like a photo of an X-ray film or a multi-panel X-ray sheet.",
      "Body-region check: Auto-detect is disabled for this upload because it may confuse Chest, Spine, Knee, or other body areas.",
      "Recommendation: upload one clear single X-ray image, or rely on the official radiology report for the exact body region and findings.",
    ],
    "simple_explanation": safe_summary,
    "caution": "Educational use only. This is not a confirmed diagnosis and does not replace a doctor or radiologist.",
    "raw_note": reason,
    "layer_1_safety": {
      "body_part": "Needs review",
      "view": "Not confirmed",
      "image_quality": "Limited / unclear image",
      "status": "Limited / unclear image",
      "doctor_review_required": True,
      "urgency_reason": reason or "Film-photo/multi-panel upload is not reliable for body-part auto-detection.",
    },
    "layer_2_findings": {
      "visible_anatomy": [],
      "hardware_present": None,
      "hardware_description": "Not assessed because the upload is a photo of a film or a multi-panel sheet.",
      "possible_findings": [],
      "uncertainty": [
        "The exact body region cannot be confirmed reliably from this kind of upload.",
        "Multiple panels, glare, or background can cause inconsistent AI labels on repeated runs.",
      ],
      "confidence": {"body_part": 0.0, "view": 0.0, "hardware": 0.0, "fracture_or_abnormality": 0.0},
    },
    "layer_3_report": {
      "safe_impression": safe_summary,
      "patient_friendly_summary": safe_summary,
      "what_to_ask_doctor": [
        "Which body region and projection does the official X-ray report mention?",
        "Can you review the original clear X-ray image instead of the phone photo of the film?",
        "Are there any findings in the official radiology report that need treatment?",
      ],
    },
  })


def _xray_result_cache_key(image_path, selected_xray_region="Auto-detect"):
  try:
    if not image_path or not os.path.exists(image_path):
      return ""
    file_hash = calculate_file_hash(image_path)
    region_key = re.sub(r"[^a-z0-9]+", "_", safe_str(selected_xray_region or "Auto-detect").strip().lower()).strip("_")
    return f"{XRAY_STABLE_RESULT_VERSION}:{file_hash}:{region_key}"
  except Exception:
    return ""


def _xray_get_cached_result(cache_key):
  try:
    if cache_key and cache_key in XRAY_RESULT_CACHE:
      return json.loads(json.dumps(XRAY_RESULT_CACHE[cache_key], ensure_ascii=False))
  except Exception:
    pass
  return None


def _xray_store_cached_result(cache_key, result):
  result = normalize_xray_result(result if isinstance(result, dict) else {})
  try:
    if cache_key:
      XRAY_RESULT_CACHE[cache_key] = json.loads(json.dumps(result, ensure_ascii=False))
  except Exception:
    pass
  return result


def _sanitize_low_confidence_hardware_claims(result):
  """Remove false metal/hardware claims unless hardware evidence is strong.

  Keeps the findings/summary structure unchanged but prevents bone edges from being called hardware.
  """
  result = normalize_xray_result(result)
  layer2 = result.get("layer_2_findings", {}) if isinstance(result.get("layer_2_findings", {}), dict) else {}
  confidence = layer2.get("confidence", {}) if isinstance(layer2.get("confidence", {}), dict) else {}
  try:
    hw_conf = float(confidence.get("hardware", 0.0) or 0.0)
  except Exception:
    hw_conf = 0.0

  hw_present = layer2.get("hardware_present", None)
  body_family = _body_part_family_label(result.get("body_part", ""))

  # Be stricter for hand/wrist/forearm/foot/ankle because cortical bone edges often look like metal.
  required = 0.92 if body_family in {"hand_wrist", "arm_forearm", "foot_ankle"} else 0.82
  allow_hardware = (hw_present is True and hw_conf >= required)

  if allow_hardware:
    return result

  hardware_terms = re.compile(r"\b(hardware|metal|metallic|fixation|screw|wire|plate|pin|rod|implant|surgical repair|orthopedic hardware)\b", re.I)

  cleaned = []
  for item in result.get("key_findings", []) or []:
    s = safe_str(item).strip()
    if not s:
      continue
    if hardware_terms.search(s):
      continue
    cleaned.append(s)

  # Remove hardware-related possible findings.
  if isinstance(layer2, dict):
    layer2["hardware_present"] = False
    layer2["hardware_description"] = "No metal repair hardware is confirmed by this educational review."
    layer2["possible_findings"] = [
      x for x in (layer2.get("possible_findings", []) or [])
      if not hardware_terms.search(safe_str(x))
    ]
    result["layer_2_findings"] = layer2

  if not any("hardware" in x.lower() for x in cleaned):
    cleaned.insert(1 if cleaned else 0, "Hardware check: no metal repair hardware is confirmed by this educational review.")

  result["key_findings"] = cleaned[:6]

  for key in ["overall_impression", "simple_explanation"]:
    txt = safe_str(result.get(key, ""))
    # Drop only sentences containing low-confidence hardware claims.
    parts = re.split(r"(?<=[.!?])\s+", txt)
    parts = [p for p in parts if not hardware_terms.search(p)]
    if parts:
      result[key] = " ".join(parts).strip()

  return normalize_xray_result(result)


# ============================================================
# X-RAY BODY-REGION SAFETY GATE
# Prevents confident wrong labels like Chest/Knee/Ankle when the
# AI metadata pass and local image-layout checks do not agree.
# ============================================================
XRAY_BODY_REGION_CHOICES = [
  "Auto-detect", "Chest", "Knee", "Foot/Ankle", "Hand/Wrist",
  "Shoulder", "Elbow", "Arm/Forearm", "Hip/Pelvis", "Spine",
  "Skull/Face", "Abdomen/KUB", "Other / Not sure"
]


def _xray_selected_region_to_body_part(selected_region):
  """Normalize the optional user-selected X-ray body region."""
  choice = safe_str(selected_region).strip()
  if not choice or choice.lower() in {"auto", "auto-detect", "autodetect", "automatic"}:
    return ""
  mapping = {
    "chest": "Chest",
    "knee": "Knee",
    "foot/ankle": "Foot/Ankle",
    "hand/wrist": "Hand/Wrist",
    "shoulder": "Shoulder",
    "elbow": "Elbow",
    "arm/forearm": "Arm/Forearm",
    "hip/pelvis": "Hip/Pelvis",
    "spine": "Spine",
    "skull/face": "Skull/Face",
    "abdomen/kub": "Abdomen/KUB",
  }
  return mapping.get(choice.lower(), "")


def _xray_set_body_region_needs_review(result, reason=""):
  """Return a safe generic X-ray result instead of a confident wrong body label."""
  result = normalize_xray_result(result if isinstance(result, dict) else {})
  safe_reason = safe_str(reason).strip() or "The exact X-ray body region could not be confirmed confidently."
  safe_summary = (
    "This appears to be a medical X-ray image, but the system could not confidently confirm "
    "the exact body region. To avoid showing a wrong Chest, Knee, Foot/Ankle, or other label, "
    "please review the original image with a doctor or radiologist."
  )
  result["exam_type"] = "X-ray image"
  result["body_part"] = "Needs review"
  result["status"] = "Limited / unclear image"
  result["overall_impression"] = safe_summary
  result["simple_explanation"] = safe_summary
  result["key_findings"] = [
    "Body-region confidence check: the system did not confidently agree on the exact X-ray body part.",
    "The uploaded image should be reviewed manually before using a body-specific summary.",
    "No chest-, knee-, ankle-, or other body-specific conclusion is shown because the body region is uncertain.",
  ]
  result["caution"] = "This is educational only and not a confirmed diagnosis. Please consult a qualified doctor or radiologist."
  result["raw_note"] = (safe_str(result.get("raw_note", "")).strip() + " " + safe_reason).strip()
  result["layer_1_safety"] = {
    "body_part": "Needs review",
    "view": safe_str(result.get("view", "Unknown")) or "Unknown",
    "image_quality": "Limited / unclear image",
    "status": "Limited / unclear image",
    "doctor_review_required": True,
    "urgency_reason": safe_reason,
  }
  result["layer_2_findings"] = {
    "visible_anatomy": [],
    "hardware_present": None,
    "hardware_description": "Not assessed because the body region was not confirmed confidently.",
    "possible_findings": [],
    "uncertainty": [safe_reason],
    "confidence": {"body_part": 0.0, "view": 0.0, "hardware": 0.0, "fracture_or_abnormality": 0.0},
  }
  result["layer_3_report"] = {
    "safe_impression": safe_summary,
    "patient_friendly_summary": safe_summary,
    "what_to_ask_doctor": [
      "Which body region and projection does this X-ray show?",
      "Does the official radiology report mention any finding that needs treatment?",
      "Do my symptoms match anything visible on this X-ray?",
    ],
  }
  return normalize_xray_result(result)


def _xray_apply_user_selected_region(result, selected_region="Auto-detect"):
  """Use user-provided metadata when the user explicitly selects an X-ray body region."""
  selected_body = _xray_selected_region_to_body_part(selected_region)
  if not selected_body:
    return normalize_xray_result(result)
  result = normalize_xray_result(result if isinstance(result, dict) else {})
  old_body = safe_str(result.get("body_part", "Unknown")) or "Unknown"
  result["body_part"] = selected_body
  result["exam_type"] = f"{selected_body} X-ray"
  result["raw_note"] = (safe_str(result.get("raw_note", "")).strip() + f" User-selected body region used: {selected_body}. Previous auto label: {old_body}.").strip()
  layer1 = result.get("layer_1_safety", {}) if isinstance(result.get("layer_1_safety", {}), dict) else {}
  layer1["body_part"] = selected_body
  result["layer_1_safety"] = layer1
  layer2 = result.get("layer_2_findings", {}) if isinstance(result.get("layer_2_findings", {}), dict) else {}
  if not layer2.get("visible_anatomy"):
    layer2["visible_anatomy"] = _default_xray_visible_anatomy(selected_body)
  conf = layer2.get("confidence", {}) if isinstance(layer2.get("confidence", {}), dict) else {}
  conf["body_part"] = max(_xray_confidence_value(conf.get("body_part", 0.0)), 0.95)
  layer2["confidence"] = conf
  result["layer_2_findings"] = layer2
  findings = _xray_as_list(result.get("key_findings", []), max_items=10)
  user_note = f"Body region selected by user: {selected_body}. The AI summary is written for this selected region."
  if not any("body region selected by user" in f.lower() for f in findings):
    findings.insert(0, user_note)
  result["key_findings"] = findings[:8]
  return normalize_xray_result(result)


def _xray_body_region_confidence_gate(result, image_path=None, study_meta=None, selected_region="Auto-detect"):
  """Use Auto-detect only when body-region evidence is consistent enough."""
  selected_body = _xray_selected_region_to_body_part(selected_region)
  if selected_body:
    return _xray_apply_user_selected_region(result, selected_region)

  result = normalize_xray_result(result if isinstance(result, dict) else {})
  study_meta = study_meta if isinstance(study_meta, dict) else {}
  local = _local_body_part_check_for_result(image_path) if image_path and os.path.exists(image_path) else {}

  final_body = safe_str(result.get("body_part", ""))
  meta_body = safe_str(study_meta.get("body_part", ""))
  local_body = safe_str(local.get("body_part", ""))
  final_family = _body_part_family_label(final_body)
  meta_family = _body_part_family_label(meta_body)
  local_family = _body_part_family_label(local_body)

  try:
    meta_conf = float(study_meta.get("body_part_confidence", 0.0) or 0.0)
  except Exception:
    meta_conf = 0.0
  try:
    local_score = int(local.get("local_score", 0) or 0)
  except Exception:
    local_score = 0

  layer2 = result.get("layer_2_findings", {}) if isinstance(result.get("layer_2_findings", {}), dict) else {}
  conf_obj = layer2.get("confidence", {}) if isinstance(layer2.get("confidence", {}), dict) else {}
  result_conf = _xray_confidence_value(conf_obj.get("body_part", 0.0))

  known = [x for x in [final_family, meta_family, local_family] if x != "unknown"]
  has_disagreement = len(set(known)) >= 2
  chest_nonchest = (
    (final_family == "chest" and local_family not in {"unknown", "chest"})
    or (local_family == "chest" and final_family not in {"unknown", "chest"})
    or (meta_family == "chest" and local_family not in {"unknown", "chest"})
  )

  # Strong local extremity evidence can fix obvious Chest/Knee/Ankle mistakes.
  if local_family != "unknown" and final_family != "unknown" and final_family != local_family and local_score >= 8:
    result["body_part"] = local_body
    result["exam_type"] = f"{local_body} X-ray"
    result["raw_note"] = (safe_str(result.get("raw_note", "")).strip() + f" Strong local body-region evidence used: {local_body}.").strip()
    return normalize_xray_result(result)

  # Otherwise, do not guess. This is safer than showing a confident wrong label.
  if has_disagreement and (chest_nonchest or meta_conf < 0.80 or result_conf < 0.80 or local_score < 8):
    reason = (
      f"Auto-detect disagreement: AI/final={final_body or 'Unknown'}, "
      f"metadata={meta_body or 'Unknown'} (confidence {meta_conf:.2f}), "
      f"local image check={local_body or 'Unknown'} (score {local_score})."
    )
    return _xray_set_body_region_needs_review(result, reason)

  if final_family == "unknown" and meta_family == "unknown" and local_family == "unknown":
    return _xray_set_body_region_needs_review(result, "No reliable body-region evidence was found by the auto-detect checks.")

  if result_conf < 0.50 and meta_conf < 0.65 and local_score < 6:
    return _xray_set_body_region_needs_review(result, "Body-region confidence was too low for a body-specific X-ray label.")

  return result

def detect_xray_study_metadata(image_path):
  """Run a dedicated body-part/view/quality metadata pass before full analysis."""
  local = _local_body_part_check_for_result(image_path) if os.path.exists(image_path) else {}
  quality = assess_xray_image_quality(image_path) if os.path.exists(image_path) else {}

  fallback_body = safe_str(local.get("body_part", "Unknown body area")) or "Unknown body area"
  fallback_view = _sanitize_xray_view_label(local.get("view", "Unknown"), fallback_body)
  fallback_quality = "Blurred" if quality.get("blurry") else "Acceptable"
  fallback = {
    "is_xray": True,
    "body_part": fallback_body,
    "body_part_confidence": 0.55 if fallback_body and "unknown" not in fallback_body.lower() else 0.25,
    "view": fallback_view,
    "image_quality": fallback_quality,
    "blur_flag": bool(quality.get("blurry")),
    "visible_anatomy": _default_xray_visible_anatomy(fallback_body)[:6],
    "notes": safe_str(local.get("note", "local metadata pass")),
  }

  if client is None or not os.path.exists(image_path):
    return fallback

  try:
    image_data_url = encode_image_to_data_url(image_path)
    response = client.chat.completions.create(
      model=VISION_MODEL,
      messages=[
        {"role": "system", "content": "You are a careful medical image assistant for educational purposes only. Return strict JSON only."},
        {"role": "user", "content": [
          {"type": "text", "text": XRAY_STUDY_METADATA_PROMPT},
          {"type": "image_url", "image_url": {"url": image_data_url}}
        ]}
      ],
      temperature=0.0,
    )
    raw = safe_str(response.choices[0].message.content).strip()
    parsed = extract_json_object(raw) or {}
    if not isinstance(parsed, dict) or not parsed:
      return fallback

    body_part = safe_str(parsed.get("body_part", fallback["body_part"])) or fallback["body_part"]
    view = _sanitize_xray_view_label(parsed.get("view", fallback["view"]), body_part)
    conf = parsed.get("body_part_confidence", fallback["body_part_confidence"])
    try:
      conf = float(conf)
    except Exception:
      conf = fallback["body_part_confidence"]
    meta = {
      "is_xray": bool(parsed.get("is_xray", True)),
      "body_part": body_part,
      "body_part_confidence": max(0.0, min(1.0, conf)),
      "view": view,
      "image_quality": safe_str(parsed.get("image_quality", fallback_quality)) or fallback_quality,
      "blur_flag": bool(parsed.get("blur_flag", quality.get("blurry", False))),
      "visible_anatomy": [safe_str(x) for x in parsed.get("visible_anatomy", []) if safe_str(x).strip()][:8] or fallback["visible_anatomy"],
      "notes": safe_str(parsed.get("notes", "")).strip(),
    }

    # If the model is vague but the local detector is strong, keep the stronger known local label.
    if local and meta["body_part_confidence"] < 0.65 and "unknown" not in fallback_body.lower():
      meta["body_part"] = fallback_body
      meta["view"] = fallback_view if fallback_view not in {"Unknown", "Single view - projection uncertain"} else meta["view"]
      meta["visible_anatomy"] = meta["visible_anatomy"] or fallback["visible_anatomy"]
      meta["notes"] = (meta["notes"] + " Local body-part evidence also supports this study type.").strip()

    if quality.get("blurry"):
      meta["blur_flag"] = True
      meta["image_quality"] = "Blurred"

    return meta
  except Exception:
    return fallback


def _apply_study_metadata_hint(result, study_meta):
  """Use the dedicated metadata pass to correct obvious body-part/view mismatches."""
  result = normalize_xray_result(result)
  if not isinstance(study_meta, dict) or not study_meta:
    return result

  meta_body = safe_str(study_meta.get("body_part", "")).strip()
  meta_view = _sanitize_xray_view_label(study_meta.get("view", "Unknown"), meta_body)
  meta_conf = study_meta.get("body_part_confidence", 0.0)
  try:
    meta_conf = float(meta_conf)
  except Exception:
    meta_conf = 0.0

  body_part = safe_str(result.get("body_part", "Unknown"))
  body_lower = body_part.lower()
  meta_lower = meta_body.lower()

  meta_family = _body_part_family_label(meta_body)
  body_family = _body_part_family_label(body_part)
  chest_nonchest_conflict = bool(meta_body) and (_is_chest_body_part(meta_body) != _is_chest_body_part(body_part))
  family_conflict = meta_family != "unknown" and body_family != "unknown" and meta_family != body_family

  # Safer rule: metadata may correct an Unknown label, but known-vs-known conflicts
  # need high confidence. Otherwise the final confidence gate will show Needs review
  # instead of forcing a wrong Chest/Knee/Ankle label.
  if meta_body and "unknown" not in meta_lower:
    obvious_mismatch = False
    if "unknown" in body_lower or "unclear" in body_lower:
      obvious_mismatch = meta_conf >= 0.72
    elif chest_nonchest_conflict or family_conflict:
      obvious_mismatch = meta_conf >= 0.82

    if obvious_mismatch:
      old = body_part or "Unknown"
      result["body_part"] = meta_body
      result["exam_type"] = f"{meta_body} X-ray"
      layer1 = result.get("layer_1_safety", {}) if isinstance(result.get("layer_1_safety", {}), dict) else {}
      layer1["body_part"] = meta_body
      result["layer_1_safety"] = layer1
      result["raw_note"] = (safe_str(result.get("raw_note", "")).strip() + f" Body-part label adjusted from {old} using the dedicated study-metadata pass.").strip()

  if meta_view not in {"", "Unknown", "Single view - projection uncertain"}:
    model_view = safe_str(result.get("view", "Unknown"))
    if _combined_or_ambiguous_xray_view(model_view) or "unknown" in model_view.lower() or "single view" in model_view.lower():
      result["view"] = meta_view
      layer1 = result.get("layer_1_safety", {}) if isinstance(result.get("layer_1_safety", {}), dict) else {}
      layer1["view"] = meta_view
      result["layer_1_safety"] = layer1

  if study_meta.get("blur_flag"):
    result["status"] = "Limited / unclear image"
    result["overall_impression"] = BLURRY_XRAY_IMAGE_ERROR
    result["simple_explanation"] = BLURRY_XRAY_IMAGE_ERROR
    result["key_findings"] = ["Image quality check: the uploaded X-ray appears blurred or too unclear for a reliable review."]

  return normalize_xray_result(result)

def _focus_xray_region_from_image(img):
  """Return the most likely X-ray region from an uploaded image.

  Normal uploads are already just the X-ray, but users sometimes upload a
  screenshot that contains the X-ray plus app UI. This conservative crop keeps
  body-part detection focused on the radiograph-like area instead of text/cards.
  """
  try:
    if img is None or img.size == 0:
      return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 120 or w < 120:
      return gray

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    candidates = []

    # Thresholds look for low-saturation, higher-contrast radiograph regions.
    # UI cards are usually very bright and low contrast, so they score lower.
    for gray_threshold, sat_limit in [(205, 110), (220, 90)]:
      mask = ((gray < gray_threshold) & (sat < sat_limit)).astype(np.uint8) * 255
      kernel = np.ones((7, 7), np.uint8)
      mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
      mask = cv2.dilate(mask, kernel, iterations=1)
      contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area_ratio = (bw * bh) / max(h * w, 1)
        if area_ratio < 0.02 or area_ratio > 0.95 or bw < 100 or bh < 90:
          continue

        roi = gray[y:y + bh, x:x + bw]
        if roi.size == 0:
          continue
        contrast = float(roi.std())
        dark_ratio = float((roi < 85).mean())
        bright_ratio = float((roi > 170).mean())
        mid_ratio = float(((roi >= 85) & (roi <= 220)).mean())

        if contrast < 22 or (dark_ratio < 0.025 and mid_ratio < 0.25):
          continue

        score = (contrast * 1.5) + (dark_ratio * 90) + (bright_ratio * 22) + (area_ratio * 20)
        candidates.append((score, x, y, bw, bh))

    if not candidates:
      return gray

    candidates.sort(reverse=True)
    score, x, y, bw, bh = candidates[0]
    if score < 45:
      return gray

    margin_x = max(4, int(bw * 0.01))
    margin_y = max(4, int(bh * 0.01))
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w, x + bw + margin_x)
    y2 = min(h, y + bh + margin_y)
    cropped = gray[y1:y2, x1:x2]
    return cropped if cropped.size else gray
  except Exception:
    try:
      return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
      return None



def _xray_is_isolated_extremity_shape(gray):
  """Detect an isolated limb/foot/hand-like X-ray shape that can falsely look like chest.

  Chest X-rays usually fill most of the cropped radiograph field and have a
  roughly central, bilateral thorax layout. A lateral foot/ankle or other
  extremity often appears as one elongated bright object on a black background.
  """
  try:
    if gray is None or gray.size == 0:
      return False, "isolated extremity shape unavailable"
    work = _crop_to_xray_content(gray)
    if work is None or work.size == 0:
      work = gray
    h, w = work.shape[:2]
    if h < 90 or w < 90:
      return False, "image too small for isolated extremity check"

    aspect = w / max(h, 1)
    dark_ratio = float((work < 45).mean())
    active_mask = (work > 35).astype(np.uint8)
    active_ratio = float(active_mask.mean())

    edges = cv2.Canny(work, 45, 135)
    edge_density = float(edges.mean()) / 255.0

    non_dark = work[work > 10]
    thresh = max(115, int(np.percentile(non_dark, 68))) if non_dark.size else 135
    bright_mask = (work > thresh).astype(np.uint8) * 255
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = max(h * w, 1)
    bright_areas = [cv2.contourArea(c) / img_area for c in contours if cv2.contourArea(c) / img_area > 0.001]
    largest_bright = max(bright_areas) if bright_areas else 0.0

    ys, xs = np.where(active_mask > 0)
    if len(xs) < 250:
      return False, "not enough active X-ray pixels for extremity check"
    pts = np.column_stack([xs, ys]).astype(np.float32)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    elongation = float(vals[1] / max(vals[0], 1e-6))
    angle = abs(math.degrees(math.atan2(vecs[1, 1], vecs[0, 1])))
    angle = min(angle, 180 - angle)

    # The strongest false-chest pattern: a single elongated foot/hand/limb
    # object on a black background. This is common in lateral foot/ankle films.
    isolated = (
      0.70 <= aspect <= 2.35
      and 0.18 <= active_ratio <= 0.62
      and dark_ratio >= 0.35
      and edge_density >= 0.030
      and elongation >= 2.15
      and largest_bright >= 0.055
    )

    note = (
      f"isolated extremity shape {isolated}; aspect {aspect:.2f}; active area {active_ratio:.0%}; "
      f"dark area {dark_ratio:.0%}; elongation {elongation:.2f}; edge density {edge_density:.1%}; "
      f"largest bright component {largest_bright:.0%}; main angle {angle:.0f}"
    )
    return bool(isolated), note
  except Exception as e:
    return False, f"isolated extremity check failed: {e}"

def _chest_projection_score(gray):
  """Score whether an image has a chest/rib/lung projection layout.

  Chest X-rays are not always wide. Many uploaded AP/PA chest images are
  portrait or nearly square, so this score uses lung-field layout rather than
  aspect ratio alone. The goal is to stop obvious chest X-rays being reported
  as knee/orthopedic images when ribs and clavicles create strong bone edges.
  """
  try:
    if gray is None or gray.size == 0:
      return 0, "chest score unavailable"
    work = gray
    try:
      cropped = _crop_to_xray_content(gray)
      if cropped is not None and cropped.size > 0:
        work = cropped
    except Exception:
      work = gray

    h, w = work.shape[:2]
    if h < 100 or w < 100:
      return 0, "image too small for chest layout check"

    aspect = w / max(h, 1)
    left_lung = work[int(0.18 * h):int(0.90 * h), int(0.06 * w):int(0.43 * w)]
    right_lung = work[int(0.18 * h):int(0.90 * h), int(0.57 * w):int(0.94 * w)]
    center = work[int(0.08 * h):int(0.95 * h), int(0.40 * w):int(0.60 * w)]
    upper = work[:max(1, int(0.30 * h)), int(0.08 * w):int(0.92 * w)]

    if left_lung.size == 0 or right_lung.size == 0 or center.size == 0:
      return 0, "chest layout regions unavailable"

    side_mean = (float(left_lung.mean()) + float(right_lung.mean())) / 2.0
    center_mean = float(center.mean())
    center_side_diff = center_mean - side_mean
    side_dark_ratio = (float((left_lung < 110).mean()) + float((right_lung < 110).mean())) / 2.0
    side_mid_ratio = (float(((left_lung >= 75) & (left_lung <= 185)).mean()) + float(((right_lung >= 75) & (right_lung <= 185)).mean())) / 2.0
    center_bright_ratio = float((center > 155).mean())
    upper_bright_ratio = float((upper > 155).mean()) if upper.size else 0.0
    lung_lr_diff = abs(float(left_lung.mean()) - float(right_lung.mean()))

    try:
      left_edges = cv2.Canny(left_lung, 45, 135)
      right_edges = cv2.Canny(right_lung, 45, 135)
      side_edge_density = (float(left_edges.mean()) + float(right_edges.mean())) / (2.0 * 255.0)
      total_edge_density = float(cv2.Canny(work, 45, 135).mean()) / 255.0
    except Exception:
      side_edge_density = 0.0
      total_edge_density = 0.0

    score = 0

    # Aspect is a weak clue only. Chest AP/PA uploads are commonly portrait,
    # square, or wide depending on crop, so do not penalize portrait chest.
    if 0.58 <= aspect <= 2.10:
      score += 2
    if 0.70 <= aspect <= 1.65:
      score += 1
    if 1.15 <= aspect <= 1.95:
      score += 1

    # Strong chest clues: two darker lung fields with a brighter central
    # spine/mediastinum column and rib/vascular texture inside the lung fields.
    if center_side_diff > 16:
      score += 2
    if center_side_diff > 35:
      score += 2
    if side_dark_ratio > 0.07:
      score += 1
    if side_dark_ratio > 0.16:
      score += 1
    if side_mid_ratio > 0.18:
      score += 1
    if side_edge_density > 0.018:
      score += 2
    if side_edge_density > 0.035:
      score += 1
    if center_bright_ratio > 0.18:
      score += 1
    if upper_bright_ratio > 0.20:
      score += 1
    if lung_lr_diff < 70:
      score += 1
    if total_edge_density > 0.035 and side_dark_ratio > 0.06:
      score += 1

    # Guard against knee/limb images: blank black margins around a bright
    # central bone can look like "dark lung fields" but has little texture.
    if side_edge_density < 0.012 and side_dark_ratio > 0.18:
      score -= 3
    if side_dark_ratio < 0.035 and center_side_diff < 14:
      score -= 4
    if aspect < 0.48 or aspect > 2.45:
      score -= 4

    isolated_extremity, isolated_note = _xray_is_isolated_extremity_shape(work)
    if isolated_extremity:
      # Strong blocker for lateral foot/ankle/hand/limb X-rays that have
      # a single elongated bone cluster on black background.
      score -= 10

    note = (
      f"chest projection score: {score}; aspect {aspect:.2f}; "
      f"centre-side brightness difference {center_side_diff:.0f}; side dark area {side_dark_ratio:.0%}; "
      f"side edge density {side_edge_density:.1%}; {isolated_note}"
    )
    return int(score), note
  except Exception:
    return 0, "chest layout check failed"


def _local_chest_attention_screen(gray):
  """High-sensitivity local screen for chest X-rays.

  It does not diagnose. It only blocks a false-normal result when the lung
  fields show clear asymmetry, focal white/patchy shadow, or limited quality.
  """
  try:
    if gray is None or gray.size == 0:
      return {"needs_attention": False, "finding": "", "possible_finding": "", "note": "chest screen unavailable", "confidence": 0.0}
    work = _crop_to_xray_content(gray)
    if work is None or work.size == 0:
      work = gray
    h, w = work.shape[:2]
    if h < 140 or w < 140:
      return {"needs_attention": False, "finding": "", "possible_finding": "", "note": "image too small for chest screen", "confidence": 0.0}

    chest_score, chest_note = _chest_projection_score(work)
    if chest_score < 6:
      return {"needs_attention": False, "finding": "", "possible_finding": "", "note": chest_note, "confidence": 0.0}

    y1, y2 = int(0.12 * h), int(0.88 * h)
    left = work[y1:y2, int(0.04 * w):int(0.44 * w)]
    right = work[y1:y2, int(0.56 * w):int(0.96 * w)]
    if left.size == 0 or right.size == 0:
      return {"needs_attention": False, "finding": "", "possible_finding": "", "note": "lung regions unavailable", "confidence": 0.0}

    left_mean = float(left.mean())
    right_mean = float(right.mean())
    mean_diff = abs(left_mean - right_mean)
    left_bright = float((left > 170).mean())
    right_bright = float((right > 170).mean())
    bright_diff = abs(left_bright - right_bright)
    max_bright = max(left_bright, right_bright)
    left_very_bright = float((left > 190).mean())
    right_very_bright = float((right > 190).mean())
    very_bright_diff = abs(left_very_bright - right_very_bright)

    # Upper-zone check catches apical/upper-lung opacities in cropped chest images.
    uy1, uy2 = int(0.14 * h), int(0.62 * h)
    uleft = work[uy1:uy2, int(0.05 * w):int(0.44 * w)]
    uright = work[uy1:uy2, int(0.56 * w):int(0.95 * w)]
    upper_mean_diff = abs(float(uleft.mean()) - float(uright.mean())) if uleft.size and uright.size else 0.0
    upper_bright_diff = abs(float((uleft > 170).mean()) - float((uright > 170).mean())) if uleft.size and uright.size else 0.0

    score = 0
    if mean_diff >= 24:
      score += 2
    elif mean_diff >= 18:
      score += 1
    if bright_diff >= 0.14:
      score += 2
    elif bright_diff >= 0.09:
      score += 1
    if very_bright_diff >= 0.10:
      score += 1
    if max_bright >= 0.46 and mean_diff >= 16:
      score += 1
    if upper_mean_diff >= 24:
      score += 1
    if upper_bright_diff >= 0.12:
      score += 1

    # Prefer sensitivity for patient safety: when a chest image has obvious
    # one-sided patchy brightness, do not let the report say normal.
    needs_attention = score >= 3
    confidence = _xray_confidence_value(min(0.85, score / 7.0))
    note = (
      f"{chest_note}; lung side mean difference {mean_diff:.0f}; "
      f"bright-area difference {bright_diff:.0%}; upper-zone difference {upper_mean_diff:.0f}"
    )
    if not needs_attention:
      return {"needs_attention": False, "finding": "", "possible_finding": "", "note": note, "confidence": confidence}

    finding = "Chest/lung check: a visible uneven white shadow or opacity is present in a lung area, so a doctor or X-ray specialist should review it."
    possible_finding = "possible abnormal lung opacity or patchy shadow that needs radiologist review"
    return {"needs_attention": True, "finding": finding, "possible_finding": possible_finding, "note": note, "confidence": confidence}
  except Exception as e:
    return {"needs_attention": False, "finding": "", "possible_finding": "", "note": f"chest screen failed: {e}", "confidence": 0.0}


def _apply_universal_xray_image_safety_gate(result, image_path=None):
  """Final high-sensitivity safety gate after the vision model/local detector.

  This keeps the app universal: any X-ray can be uploaded. Known body parts use
  related checks; unknown or unclear studies become limited/review recommended,
  not confidently normal. For chest, a local opacity/asymmetry screen prevents
  obvious abnormal lung shadows from being reported as normal.
  """
  result = normalize_xray_result(result)
  body_part = safe_str(result.get("body_part", ""))
  view = safe_str(result.get("view", ""))

  # Do not allow a normal-style result for unknown body part/view.
  if result.get("status") == "No obvious acute abnormality":
    if "unknown" in body_part.lower() or "unclear" in body_part.lower() or "unknown" in view.lower() or "uncertain" in view.lower():
      result["status"] = "Limited / unclear image"
      layer1 = result.get("layer_1_safety", {}) if isinstance(result.get("layer_1_safety", {}), dict) else {}
      layer1["status"] = result["status"]
      layer1["doctor_review_required"] = True
      layer1["urgency_reason"] = "The body part or view is not clear enough for a confident normal-style educational summary."
      result["layer_1_safety"] = layer1

  # Body-specific local safety screen for chest/lung images.
  try:
    if image_path and os.path.exists(image_path):
      img = cv2.imread(image_path)
      gray = _focus_xray_region_from_image(img) if img is not None else None
      if gray is not None and gray.size:
        cropped_for_gate = _crop_to_xray_content(gray)
        limb_override_for_gate = _local_long_limb_or_extremity_override(cropped_for_gate)
        chest_score, _ = _chest_projection_score(cropped_for_gate)
        is_chest_like = (_is_chest_body_part(body_part) or chest_score >= 8) and not limb_override_for_gate
        if is_chest_like:
          screen = _local_chest_attention_screen(gray)
          if screen.get("needs_attention"):
            result = _xray_force_attention(
              result,
              "Visible lung-area shadowing/opacity or asymmetry should be reviewed by a doctor or X-ray specialist.",
              finding=screen.get("finding"),
              possible_finding=screen.get("possible_finding"),
            )
            if not _is_chest_body_part(result.get("body_part", "")) and chest_score >= 8:
              result["body_part"] = "Chest"
              result["exam_type"] = "Chest X-ray"
              result["view"] = _sanitize_xray_view_label(result.get("view", "Frontal"), "Chest")
            result["overall_impression"] = "This appears to be a chest X-ray with a visible lung-area shadow/opacity that needs doctor or X-ray specialist review."
            result["simple_explanation"] = (
              "This chest X-ray has a visible white or patchy shadow in the lung area. "
              "This can happen for different reasons, such as infection, inflammation, old scarring, or another lung condition. "
              "This is not a diagnosis, but it should be reviewed by a doctor or X-ray specialist."
            )
            result["raw_note"] = (safe_str(result.get("raw_note", "")).strip() + " " + safe_str(screen.get("note", ""))).strip()
  except Exception:
    pass

  # Final body-part relevance guard: never leave a chest/lung report on a clear limb/extremity image.
  try:
    if image_path and os.path.exists(image_path):
      final_evidence = _local_body_part_check_for_result(image_path)
      local_bp = safe_str(final_evidence.get("body_part", ""))
      local_score = int(final_evidence.get("local_score", 0) or 0)
      if local_bp and _body_part_family_label(local_bp) != "unknown":
        if _body_part_family_label(local_bp) != _body_part_family_label(result.get("body_part", "")) and local_score >= 7:
          result["body_part"] = local_bp
          result["exam_type"] = f"{local_bp} X-ray"
          result["view"] = _sanitize_xray_view_label(final_evidence.get("view", result.get("view", "Unknown")), local_bp)
          if safe_str(result.get("status", "")).lower() == "no obvious acute abnormality":
            result["status"] = "Needs attention"
          result["overall_impression"] = (
            f"This appears to be a {result['view']} {local_bp} X-ray. "
            "The image should be reviewed by a doctor or radiologist if there is pain, injury, swelling, or limited movement."
          )
          result["key_findings"] = [
            f"Body part check: this image appears more consistent with a {local_bp} X-ray than the earlier AI label.",
            "Image relevance check: the report type has been corrected to match the uploaded X-ray area.",
            "Review check: a doctor or radiologist should review the original X-ray for the final interpretation.",
          ]
          result["simple_explanation"] = (
            f"This looks like an X-ray of the {local_bp.lower()} area. "
            "This educational tool cannot confirm a diagnosis. "
            "Please have a doctor or radiologist review the original X-ray, especially if there is pain, swelling, injury, deformity, numbness, or trouble using the limb."
          )
  except Exception:
    pass

  result = _sanitize_low_confidence_hardware_claims(result)
  return normalize_xray_result(result)


def _combined_or_ambiguous_xray_view(view):
  """Return True when a view label describes multiple possible projections.

  A single uploaded radiograph should not be reported as "AP / Lateral".
  Combined labels are only appropriate when the image actually contains more
  than one separate view. This helper is intentionally conservative and keeps
  real single-projection labels unchanged.
  """
  v = safe_str(view).strip().lower()
  if not v:
    return True
  if "unknown" in v or "unclear" in v or "projection uncertain" in v:
    return True
  combined_tokens = ["/", " or ", " and ", ","]
  return any(tok in v for tok in combined_tokens)


def _sanitize_xray_view_label(view, body_part=""):
  """Normalize X-ray view wording for a universal single-image workflow."""
  raw = safe_str(view).strip()
  if not raw:
    return "Unknown"
  v = re.sub(r"\s+", " ", raw).strip()
  vl = v.lower()

  # Preserve strong single-projection labels.
  single_map = {
    "pa": "PA",
    "ap": "AP",
    "lateral": "Lateral",
    "frontal": "Frontal",
    "oblique": "Oblique",
    "axial": "Axial",
    "sunrise": "Sunrise/Merchant",
    "merchant": "Sunrise/Merchant",
    "mortise": "Mortise",
  }
  for key, label in single_map.items():
    if re.fullmatch(rf"{key}(?: view| projection)?", vl):
      return label

  # Do not show AP/Lateral, PA/AP, etc. for one uploaded image.
  if _combined_or_ambiguous_xray_view(v):
    bp = safe_str(body_part).lower()
    if "chest" in bp or "lung" in bp:
      return "Frontal chest view"
    return "Single view - projection uncertain"

  # Light cleanup for model wording like "lateral knee view".
  if "lateral" in vl:
    return "Lateral"
  if "oblique" in vl:
    return "Oblique"
  if re.search(r"\bpa\b", vl):
    return "PA"
  if re.search(r"\bap\b", vl):
    return "AP"
  if "frontal" in vl:
    return "Frontal"
  return v


def _bright_component_stats(gray, threshold=None):
  """Small reusable component summary used by local body/view heuristics."""
  if gray is None or gray.size == 0:
    return []
  h, w = gray.shape[:2]
  if threshold is None:
    non_dark = gray[gray > 10]
    threshold = max(115, int(np.percentile(non_dark, 72))) if non_dark.size else 150
  mask = (gray >= threshold).astype(np.uint8) * 255
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  comps = []
  image_area = max(h * w, 1)
  for c in contours:
    x, y, bw, bh = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if area <= 0:
      continue
    comps.append({
      "x": x,
      "y": y,
      "w": bw,
      "h": bh,
      "area_ratio": float(area) / image_area,
      "cx": (x + bw / 2) / max(w, 1),
      "cy": (y + bh / 2) / max(h, 1),
      "aspect": bw / max(bh, 1),
    })
  return comps


def _knee_lateral_projection_score(gray):
  """Detect a lateral-knee-like layout without making a diagnosis.

  The main signal is a small patella-like bright component near the anterior
  side of the knee, plus a tall/single-joint layout. This is deliberately used
  only to label the projection, not to diagnose an injury.
  """
  try:
    work = _crop_to_xray_content(gray)
    if work is None or work.size == 0:
      work = gray
    h, w = work.shape[:2]
    if h < 100 or w < 80:
      return 0, "knee lateral check unavailable"

    aspect = w / max(h, 1)
    score = 0
    reasons = []
    if aspect < 0.95:
      score += 2
      reasons.append(f"tall single-knee layout aspect {aspect:.2f}")
    elif aspect < 1.15:
      score += 1
      reasons.append(f"near-square single-knee layout aspect {aspect:.2f}")

    # Search both anterior side bands for a compact patella-like component.
    y1, y2 = int(0.18 * h), int(0.72 * h)
    bands = [("left", 0, int(0.42 * w)), ("right", int(0.58 * w), w)]
    patella_like = 0
    for side, x1, x2 in bands:
      roi = work[y1:y2, x1:x2]
      if roi.size == 0:
        continue
      # Use a moderate threshold so the patella-like side component is not
      # lost when it is less bright than the femur/tibia cortex or hardware.
      comps = _bright_component_stats(roi, threshold=max(90, min(145, int(np.percentile(roi[roi > 10], 45)))) if np.any(roi > 10) else 115)
      for comp in comps:
        ar = comp["area_ratio"]
        # A patella on a lateral knee is compact, side-positioned, and not full-height.
        if 0.008 <= ar <= 0.16 and 0.18 <= comp["cy"] <= 0.90 and comp["h"] <= 0.55 * roi.shape[0]:
          patella_like += 1
          reasons.append(f"compact side component on {side}")
          break
    if patella_like:
      score += 4

    edges = cv2.Canny(work, 45, 135)
    edge_density = float(edges.mean()) / 255.0
    if edge_density > 0.055:
      score += 1
      reasons.append("joint-edge detail present")

    return score, "; ".join(reasons) if reasons else "no lateral-knee pattern found"
  except Exception:
    return 0, "knee lateral check failed"


def _infer_single_xray_projection_locally(gray, body_part):
  """Infer a single projection label from the image layout.

  This replaces the old hardcoded view_map that returned combined labels such
  as AP / Lateral view. It is universal: when the projection is uncertain it
  says so instead of pretending both AP and lateral views were uploaded.
  """
  try:
    bp = safe_str(body_part).lower()
    if gray is None or gray.size == 0 or "unknown" in bp:
      return "Unknown"

    work = _crop_to_xray_content(gray)
    if work is None or work.size == 0:
      work = gray
    h, w = work.shape[:2]
    aspect = w / max(h, 1)
    chest_score, _ = _chest_projection_score(work)

    if "chest" in bp or "lung" in bp:
      # AP vs PA usually cannot be proven from a screenshot alone.
      return "Frontal chest view" if chest_score >= 5 or aspect > 1.15 else "Unknown"

    if "knee" in bp:
      lat_score, _ = _knee_lateral_projection_score(work)
      if lat_score >= 4:
        return "Lateral"
      # A wider, symmetric knee is more likely frontal/AP, but keep uncertain if weak.
      if aspect >= 0.95:
        return "AP" if lat_score <= 2 else "Single view - projection uncertain"
      return "Single view - projection uncertain"

    # Body-part-specific conservative rules for common projections.
    if any(x in bp for x in ["spine", "skull"]):
      return "Lateral" if aspect < 0.85 else "AP" if aspect >= 0.85 else "Unknown"
    if any(x in bp for x in ["hand", "wrist"]):
      return "PA" if aspect >= 0.70 else "Oblique"
    if any(x in bp for x in ["foot", "ankle"]):
      if aspect > 1.25:
        return "Oblique"
      return "AP" if aspect >= 0.80 else "Lateral"
    if any(x in bp for x in ["elbow", "shoulder", "hip", "pelvis", "arm", "forearm", "leg"]):
      # Long narrow limb images are often lateral/profile; wider joint images are often frontal.
      if aspect < 0.70:
        return "Lateral"
      if aspect > 1.20:
        return "AP"
      return "Single view - projection uncertain"

    return "Single view - projection uncertain"
  except Exception:
    return "Unknown"

def _is_chest_body_part(body_part):
  return "chest" in safe_str(body_part).lower() or "lung" in safe_str(body_part).lower()


def _is_orthopedic_body_part(body_part):
  bp = safe_str(body_part).lower()
  return any(x in bp for x in [
    "knee", "hand", "wrist", "arm", "forearm", "elbow", "shoulder",
    "hip", "pelvis", "leg", "ankle", "foot", "spine", "skull", "head"
  ])


def _article_for_body_area(body_area):
  area = safe_str(body_area).strip() or "imaged area"
  return "the imaged area" if area.lower().startswith("the ") else f"the {area}"
def _plain_body_area(body_part):
  body_part = safe_str(body_part).strip()
  if not body_part or "unknown" in body_part.lower() or "unclear" in body_part.lower():
    return "the imaged area"
  return body_part


def _body_part_advice_phrase(body_part):
  """Return simple symptom language matched to the likely body part."""
  bp = safe_str(body_part).lower()
  if "chest" in bp:
    return "chest pain, breathing trouble, fever, a bad cough, or worsening shortness of breath"
  if any(x in bp for x in ["knee", "leg", "hip", "pelvis", "foot", "ankle"]):
    return "strong pain, swelling, deformity, numbness, or trouble standing or walking"
  if any(x in bp for x in ["hand", "wrist", "arm", "forearm", "elbow", "shoulder"]):
    return "strong pain, swelling, deformity, numbness, or trouble moving or using the arm/hand"
  if any(x in bp for x in ["spine", "back", "neck"]):
    return "severe pain, weakness, numbness, trouble walking, or loss of bladder/bowel control"
  if "skull" in bp or "head" in bp:
    return "severe headache, vomiting, confusion, fainting, weakness, or vision changes"
  return "severe pain, swelling, numbness, deformity, fever, or worsening symptoms"




def _detect_orthopedic_hardware_locally(gray, body_part):
  """Conservative local detector for obvious metal fixation hardware.

  This is not a diagnostic detector. It only looks for clusters of very bright,
  straight, thin structures that are more metal-like than normal bone texture.
  It is used as a fallback when no vision API is available.
  """
  try:
    bp = safe_str(body_part).lower()
    if not _is_orthopedic_body_part(body_part) or gray is None or gray.size == 0:
      return {
        "hardware_present": None,
        "hardware_description": "hardware check not applicable or not available for this image",
        "confidence": 0.0,
        "line_count": 0,
      }

    work = _crop_to_xray_content(gray)
    if work is None or work.size == 0:
      work = gray
    h, w = work.shape[:2]
    if h < 120 or w < 120:
      return {
        "hardware_present": None,
        "hardware_description": "image is too small for a reliable hardware screen",
        "confidence": 0.0,
        "line_count": 0,
      }

    # Metal usually occupies a tiny area and is among the brightest pixels.
    non_dark = work[work > 10]
    if non_dark.size == 0:
      return {
        "hardware_present": None,
        "hardware_description": "hardware check was limited by image brightness",
        "confidence": 0.0,
        "line_count": 0,
      }
    high_thresh = int(max(210, min(252, np.percentile(non_dark, 98.8))))
    bright_mask = (work >= high_thresh).astype(np.uint8) * 255
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    bright_ratio = float((bright_mask > 0).mean())

    edges = cv2.Canny(bright_mask, 40, 120)
    min_len = max(18, int(min(h, w) * 0.055))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=12, minLineLength=min_len, maxLineGap=5)

    line_count = 0
    angle_bins = []
    lengths = []
    if lines is not None:
      for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in line]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_len:
          continue
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        angle = min(angle, 180 - angle)
        # Hardware wires/screws/pins are often near-straight; bones create many curved fragments.
        if 0 <= angle <= 25 or 65 <= angle <= 115 or 25 < angle < 65:
          line_count += 1
          angle_bins.append(round(angle / 15) * 15)
          lengths.append(length)

    # Connected very-bright components add evidence, but too much bright area is likely bone/exposure.
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    slender_components = 0
    for c in contours:
      x, y, cw, ch = cv2.boundingRect(c)
      area = cv2.contourArea(c)
      if area < 6:
        continue
      long_side = max(cw, ch)
      short_side = max(1, min(cw, ch))
      slenderness = long_side / short_side
      if long_side >= min_len and slenderness >= 3.0:
        slender_components += 1

    repeated_angle_bonus = 0
    if angle_bins:
      common = Counter(angle_bins).most_common(1)[0][1]
      if common >= 2:
        repeated_angle_bonus = 1

    score = 0
    if 0.0004 <= bright_ratio <= 0.035:
      score += 1
    if line_count >= 2:
      score += 2
    if line_count >= 4:
      score += 1
    if slender_components >= 1:
      score += 1
    if slender_components >= 2:
      score += 1
    score += repeated_angle_bonus
    if bright_ratio > 0.08:
      score -= 2

    confidence = _xray_confidence_value(min(0.9, max(0.0, score / 7.0)))
    hardware_present = True if score >= 4 else (False if score <= 1 else None)

    region = "the imaged orthopedic region"
    if "knee" in bp:
      region = "the patella/knee joint region"
    elif "wrist" in bp or "hand" in bp:
      region = "the hand/wrist region"
    elif "ankle" in bp or "foot" in bp:
      region = "the foot/ankle region"
    elif "hip" in bp or "pelvis" in bp:
      region = "the hip/pelvis region"
    elif "elbow" in bp:
      region = "the elbow region"
    elif any(x in bp for x in ["arm", "forearm", "shoulder", "leg"]):
      region = "the imaged bone/joint region"

    if hardware_present is True:
      desc = f"metallic fixation hardware appears projected near {region}. The exact type, position, and fracture-healing status should be confirmed by a clinician"
    elif hardware_present is False:
      desc = "no obvious metallic fixation hardware was detected by the local screen"
    else:
      desc = f"hardware presence is uncertain; a clinician should review {region} on the original X-ray"

    return {
      "hardware_present": hardware_present,
      "hardware_description": desc,
      "confidence": confidence,
      "line_count": int(line_count),
      "bright_ratio": round(bright_ratio, 4),
    }
  except Exception as e:
    return {
      "hardware_present": None,
      "hardware_description": f"local hardware screen could not be completed: {str(e)}",
      "confidence": 0.0,
      "line_count": 0,
    }


def _build_local_xray_finding_layer(gray, body_part):
  visible_anatomy = _default_xray_visible_anatomy(body_part)
  hardware = _detect_orthopedic_hardware_locally(gray, body_part)
  hardware_present = hardware.get("hardware_present")

  possible_findings = []
  uncertainty = ["Local image processing cannot confirm a diagnosis; radiologist review of the original X-ray is needed."]

  bp = safe_str(body_part).lower()
  if hardware_present is True:
    if "knee" in bp:
      possible_findings.append("possible prior patellar fracture fixation")
    else:
      possible_findings.append("possible prior surgical fixation or orthopedic hardware in the imaged region")
    uncertainty.append("The exact hardware type, fracture healing status, and whether the hardware position is expected cannot be confirmed by this tool.")
  elif hardware_present is None:
    uncertainty.append("Hardware detection is uncertain from the local pixel-based review.")

  conf = {
    "body_part": 0.65 if visible_anatomy else 0.35,
    "view": 0.45,
    "hardware": hardware.get("confidence", 0.0),
    "fracture_or_abnormality": 0.55 if possible_findings else 0.2,
  }

  return {
    "visible_anatomy": visible_anatomy,
    "hardware_present": hardware_present,
    "hardware_description": hardware.get("hardware_description", ""),
    "possible_findings": possible_findings,
    "uncertainty": uncertainty,
    "confidence": conf,
  }



def _local_long_limb_or_extremity_override(gray):
  """High-priority deterministic body-region override for obvious non-chest extremity X-rays.

  This prevents long forearm/wrist/ankle/foot images from being mislabeled as Chest.
  It is intentionally conservative and only fires for strong geometric patterns.
  """
  try:
    work = _crop_to_xray_content(gray)
    if work is None or work.size == 0:
      work = gray
    h, w = work.shape[:2]
    if h < 80 or w < 80:
      return None

    aspect = w / max(h, 1)
    inv_aspect = h / max(w, 1)
    edges = cv2.Canny(work, 45, 135)
    edge_density = float(edges.mean()) / 255.0

    non_dark = work[work > 10]
    thresh = max(110, int(np.percentile(non_dark, 72))) if non_dark.size else 140
    comps = _bright_component_stats(work, threshold=thresh)
    component_count = len([c for c in comps if c.get("area_ratio", 0) > 0.001])

    lines = cv2.HoughLinesP(
      edges, 1, np.pi / 180, threshold=35,
      minLineLength=max(40, int(min(h, w) * 0.35)),
      maxLineGap=max(6, int(min(h, w) * 0.04))
    )
    horizontal_long = 0
    vertical_long = 0
    if lines is not None:
      for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in line]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < max(40, int(min(h, w) * 0.30)):
          continue
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        angle = min(angle, 180 - angle)
        if angle <= 18:
          horizontal_long += 1
        if angle >= 72:
          vertical_long += 1

    def _band_complexity(region):
      if region is None or region.size == 0:
        return 0, 0.0
      e = cv2.Canny(region, 45, 135)
      ed = float(e.mean()) / 255.0
      nd = region[region > 10]
      th = max(105, int(np.percentile(nd, 70))) if nd.size else 135
      cc = len([c for c in _bright_component_stats(region, threshold=th) if c.get("area_ratio", 0) > 0.001])
      return cc, ed

    left_band = work[:, :max(1, int(0.26 * w))]
    right_band = work[:, int(0.74 * w):]
    top_band = work[:max(1, int(0.26 * h)), :]
    bottom_band = work[int(0.74 * h):, :]
    lcc, led = _band_complexity(left_band)
    rcc, red = _band_complexity(right_band)
    tcc, ted = _band_complexity(top_band)
    bcc, bed = _band_complexity(bottom_band)
    end_complexity = max(lcc, rcc, tcc, bcc)
    end_edge = max(led, red, ted, bed)

    isolated_extremity, isolated_note = _xray_is_isolated_extremity_shape(work)
    if isolated_extremity and edge_density >= 0.030:
      return {
        "body_part": "Foot/Ankle",
        "view": "Lateral" if aspect >= 0.95 else "Single view - projection uncertain",
        "score": 11,
        "note": f"extremity override: isolated lateral foot/ankle-like X-ray shape; {isolated_note}",
      }

    if aspect >= 2.15 and edge_density >= 0.025:
      chest_score, _ = _chest_projection_score(work)
      if chest_score < 11:
        if end_complexity >= 5 or end_edge > 0.10:
          return {
            "body_part": "Forearm/Wrist",
            "view": "Lateral" if aspect >= 2.5 else "Single view - projection uncertain",
            "score": 10,
            "note": f"extremity override: long horizontal limb layout, aspect {aspect:.2f}, long shaft lines {horizontal_long}, end small-bone complexity {end_complexity}",
          }
        if horizontal_long >= 2:
          return {
            "body_part": "Arm/Forearm",
            "view": "Lateral" if aspect >= 2.5 else "Single view - projection uncertain",
            "score": 9,
            "note": f"extremity override: long horizontal paired-bone/long-bone layout, aspect {aspect:.2f}, long shaft lines {horizontal_long}",
          }

    if inv_aspect >= 2.15 and edge_density >= 0.025:
      chest_score, _ = _chest_projection_score(work)
      if chest_score < 10:
        if end_complexity >= 5 or vertical_long >= 2:
          return {
            "body_part": "Leg/Forearm",
            "view": "Lateral" if inv_aspect >= 2.5 else "Single view - projection uncertain",
            "score": 9,
            "note": f"extremity override: long vertical limb layout, aspect {aspect:.2f}, long shaft lines {vertical_long}, end complexity {end_complexity}",
          }

    if aspect >= 1.45 and component_count >= 7 and edge_density >= 0.055:
      chest_score, _ = _chest_projection_score(work)
      if chest_score < 10 and (end_complexity >= 6 or max(lcc, rcc) >= 6):
        return {
          "body_part": "Foot/Ankle",
          "view": "Lateral" if aspect >= 1.35 else "Single view - projection uncertain",
          "score": 8,
          "note": f"extremity override: foot/ankle-like small-bone cluster, components {component_count}, aspect {aspect:.2f}",
        }

    return None
  except Exception:
    return None


def _body_part_family_label(body_part):
  """Normalize body labels into broad families for mismatch checks."""
  bp = safe_str(body_part).lower()
  if "chest" in bp or "lung" in bp or "rib" in bp or "clavicle" in bp:
    return "chest"
  if "hand" in bp or "wrist" in bp or "finger" in bp:
    return "hand_wrist"
  if "forearm" in bp or "arm" in bp or "humerus" in bp:
    return "arm_forearm"
  if "elbow" in bp:
    return "elbow"
  if "shoulder" in bp:
    return "shoulder"
  if "foot" in bp or "ankle" in bp or "toe" in bp:
    return "foot_ankle"
  if "knee" in bp or "patella" in bp:
    return "knee"
  if "leg" in bp or "tibia" in bp or "fibula" in bp or "femur" in bp:
    return "leg"
  if "hip" in bp or "pelvis" in bp:
    return "hip_pelvis"
  if "spine" in bp or "neck" in bp or "back" in bp:
    return "spine"
  if "skull" in bp or "head" in bp or "jaw" in bp or "dental" in bp or "face" in bp:
    return "skull_face"
  if "abdomen" in bp or "kub" in bp:
    return "abdomen"
  return "unknown"


def _detect_body_part_locally(gray, h, w):
  """Detect the likely imaged body part without assuming every dark image is chest."""
  try:
    cropped = _crop_to_xray_content(gray)
    if cropped is not None and cropped.size > 0:
      gray = cropped
      h, w = gray.shape[:2]
  except Exception:
    pass

  aspect = w / max(h, 1)

  # High-priority non-chest extremity override. This fixes long forearm/wrist,
  # arm, leg, ankle, and foot X-rays being incorrectly labeled as Chest.
  extremity_override = _local_long_limb_or_extremity_override(gray)
  if extremity_override:
    bp = extremity_override.get("body_part", "Unknown body area")
    view = _sanitize_xray_view_label(extremity_override.get("view", "Single view - projection uncertain"), bp)
    note = f"local detector score: {int(extremity_override.get('score', 9))}; {extremity_override.get('note', '')}"
    return bp, view, note

  mean_intensity = float(gray.mean())
  contrast = float(gray.std())

  bone_bright = float((gray > 205).mean())
  bone_mid = float(((gray > 120) & (gray <= 205)).mean())
  bright_ratio = float((gray > 165).mean())
  dark_ratio = float((gray < 45).mean())
  active_ratio = float((gray > 35).mean())

  edges = cv2.Canny(gray, 45, 135)
  edge_density = float(edges.mean()) / 255.0

  t = gray[:h // 3, :]
  m = gray[h // 3: 2 * h // 3, :]
  b = gray[2 * h // 3:, :]
  top_mean = float(t.mean()) if t.size else mean_intensity
  mid_mean = float(m.mean()) if m.size else mean_intensity
  bot_mean = float(b.mean()) if b.size else mean_intensity

  left_mean = float(gray[:, :w // 2].mean()) if w > 1 else mean_intensity
  right_mean = float(gray[:, w // 2:].mean()) if w > 1 else mean_intensity
  lr_diff = abs(left_mean - right_mean)

  cy, cx = h // 2, w // 2
  radius = max(8, min(h, w) // 5)
  mask = np.zeros_like(gray, dtype=np.uint8)
  cv2.circle(mask, (cx, cy), radius, 255, -1)
  centre_mean = float(gray[mask > 0].mean()) if mask.any() else mean_intensity

  # Count bright bone-like components. Many small components often mean hand/foot.
  try:
    thresh_val = max(120, int(mean_intensity + 0.55 * max(contrast, 1)))
    bone_mask = (gray > thresh_val).astype(np.uint8) * 255
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(bone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = max(h * w, 1)
    component_areas = [cv2.contourArea(c) / img_area for c in contours if cv2.contourArea(c) / img_area > 0.0015]
    component_count = len(component_areas)
    largest_component = max(component_areas) if component_areas else 0.0
  except Exception:
    component_count = 0
    largest_component = 0.0

  chest_projection_score, chest_projection_note = _chest_projection_score(gray)

  scores = {
    "Knee": 0,
    "Hand/Wrist": 0,
    "Foot/Ankle": 0,
    "Arm/Forearm": 0,
    "Elbow": 0,
    "Shoulder": 0,
    "Hip/Pelvis": 0,
    "Spine": 0,
    "Skull": 0,
    "Chest": 0,
  }

  # Chest must be clearly chest-like. Black background alone should not count as lungs.
  # The chest projection score looks for a wide radiograph with two darker side
  # lung fields and a brighter central spine/mediastinum column.
  scores["Chest"] += chest_projection_score
  if aspect > 2.05 and chest_projection_score < 11:
    scores["Chest"] -= 8
  if aspect < 0.55 and chest_projection_score < 10:
    scores["Chest"] -= 6
  if aspect > 1.35:
    scores["Chest"] += 1
  if 0.08 < dark_ratio < 0.65 and active_ratio > 0.35:
    scores["Chest"] += 1
  if lr_diff < 28 and aspect > 1.35:
    scores["Chest"] += 1
  if active_ratio < 0.25:
    scores["Chest"] -= 3
  if aspect < 1.10 and chest_projection_score < 7:
    scores["Chest"] -= 4

  # Knee: a joint in the middle with strong bone edges; may be AP or lateral.
  if 0.55 <= aspect <= 1.65:
    scores["Knee"] += 2
  if centre_mean > mean_intensity + 8:
    scores["Knee"] += 2
  if edge_density > 0.055:
    scores["Knee"] += 2
  if bright_ratio > 0.08 or bone_mid > 0.18:
    scores["Knee"] += 2
  if 1 <= component_count <= 6:
    scores["Knee"] += 1
  if mid_mean >= top_mean - 8 and mid_mean >= bot_mean - 8:
    scores["Knee"] += 1

  # Hand/wrist and foot/ankle: many small bones and many edges.
  if component_count >= 8:
    scores["Hand/Wrist"] += 4
    scores["Foot/Ankle"] += 3
  if edge_density > 0.11:
    scores["Hand/Wrist"] += 2
    scores["Foot/Ankle"] += 2
  if aspect < 0.95 and component_count >= 5:
    scores["Hand/Wrist"] += 2
  if aspect >= 0.85 and component_count >= 5:
    scores["Foot/Ankle"] += 2
  if bot_mean > top_mean + 8 and component_count >= 4:
    scores["Foot/Ankle"] += 1

  # Long bones / arms / forearms often look long and narrow or long and wide.
  if (aspect < 0.55 or aspect > 1.65) and 1 <= component_count <= 5:
    scores["Arm/Forearm"] += 3
  if largest_component > 0.04 and edge_density > 0.04:
    scores["Arm/Forearm"] += 1

  # Elbow: joint-like, similar to knee but usually smaller/narrower.
  if 0.50 <= aspect <= 1.45 and 1 <= component_count <= 5:
    scores["Elbow"] += 2
  if centre_mean > mean_intensity + 10 and edge_density > 0.07:
    scores["Elbow"] += 2

  # Shoulder: one-sided round bone/head, usually asymmetric.
  if 0.70 <= aspect <= 1.55 and bright_ratio > 0.08:
    scores["Shoulder"] += 2
  if lr_diff > 18 and centre_mean > mean_intensity:
    scores["Shoulder"] += 2

  # Hip/pelvis: wide, lower-half bone structures, often fairly symmetric.
  if aspect > 1.05 and bright_ratio > 0.10 and lr_diff < 24:
    scores["Hip/Pelvis"] += 3
  if bot_mean > top_mean + 8:
    scores["Hip/Pelvis"] += 1

  # Spine: tall central column.
  if aspect < 0.62:
    scores["Spine"] += 3
  if abs(top_mean - bot_mean) < 22 and centre_mean > mean_intensity:
    scores["Spine"] += 2

  # Skull: round/oval image area with a bright rim and moderate symmetry.
  if 0.75 <= aspect <= 1.30 and edge_density > 0.04 and bone_bright > 0.06:
    scores["Skull"] += 2
  if lr_diff < 18 and abs(top_mean - bot_mean) < 35 and component_count <= 4:
    scores["Skull"] += 1

  ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
  best, best_score = ranked[0]
  second, second_score = ranked[1]

  # Use a conservative chest override when the layout strongly looks like chest.
  # This fixes chest X-rays being mislabeled as knee because ribs/shoulders can
  # produce strong bone-edge signals. Do not require a wide aspect ratio because
  # many uploaded chest X-rays are portrait or nearly square crops.
  if chest_projection_score >= 8:
    best = "Chest"
    best_score = max(best_score, scores["Chest"], chest_projection_score)
  elif chest_projection_score >= 7 and scores["Chest"] >= max(scores.get("Knee", 0), scores.get("Shoulder", 0)) - 1:
    best = "Chest"
    best_score = max(best_score, scores["Chest"], chest_projection_score)
  elif best == "Chest" and (best_score < 7 or best_score - second_score < 2):
    best, best_score = second, second_score

  if best_score < 3:
    best = "Unknown body area"

  # Infer one projection for the single uploaded image instead of hardcoding
  # combined labels like "AP / Lateral view" for every knee/elbow/spine image.
  view = _infer_single_xray_projection_locally(gray, best)
  if best == "Knee":
    knee_view_score, knee_view_note = _knee_lateral_projection_score(gray)
    confidence_note = (
      f"local detector score: {best_score}; {chest_projection_note}; "
      f"knee lateral score: {knee_view_score}; {knee_view_note}"
    )
  else:
    confidence_note = f"local detector score: {best_score}; {chest_projection_note}; view inferred as {view}"
  return best, view, confidence_note


def _analyze_xray_locally(image_path):
  """Local OpenCV-based X-ray analysis runs when no API key is configured.

  It provides a simple educational summary for any body-part X-ray without
  hardcoding chest/lung language. The local method cannot diagnose subtle
  fractures, so it uses cautious wording and asks for radiologist review.
  """
  try:
    img = cv2.imread(image_path)
    if img is None:
      return fallback_xray_result()

    film_photo_screen = detect_xray_film_photo_or_composite_upload(image_path) if os.path.exists(image_path) else {"is_limited_upload": False}
    if film_photo_screen.get("is_limited_upload"):
      return build_xray_film_photo_needs_review_result(image_path, film_photo_screen)

    quality = assess_xray_image_quality(image_path)
    annotation_screen = detect_annotated_or_composite_xray_image(image_path) if os.path.exists(image_path) else {"is_annotated": False}
    if annotation_screen.get("is_annotated"):
      return build_annotated_xray_result(image_path)

    if quality.get("blurry"):
      return normalize_xray_result({
        "exam_type": "X-ray image",
        "body_part": "Unknown",
        "view": "Unknown",
        "status": "Limited / unclear image",
        "overall_impression": BLURRY_XRAY_IMAGE_ERROR,
        "key_findings": ["Image quality check: the uploaded X-ray appears blurred or too unclear for a reliable review."],
        "simple_explanation": BLURRY_XRAY_IMAGE_ERROR,
        "caution": "Please upload a sharper X-ray image. This educational tool should not be used on blurred images.",
      })

    gray_candidate = _focus_xray_region_from_image(img)
    if gray_candidate is None or gray_candidate.size == 0:
      return fallback_xray_result()

    gray = _crop_to_xray_content(gray_candidate)
    h, w = gray.shape[:2]

    mean_intensity = float(gray.mean())
    contrast = float(gray.std())
    dark_ratio = float((gray < 50).mean())
    bright_ratio = float((gray > 200).mean())
    mid_grey_ratio = float(((gray >= 50) & (gray <= 200)).mean())

    # Use the focused region for colour/grayscale check when possible.
    try:
      if gray_candidate.shape[:2] != img.shape[:2]:
        sat_mean = 0.0
      else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat_mean = float(hsv[:, :, 1].mean())
    except Exception:
      sat_mean = 0.0
    is_grayscale_like = sat_mean < 35

    body_part, view, conf_note = _detect_body_part_locally(gray, h, w)
    body_area = _plain_body_area(body_part)
    symptom_phrase = _body_part_advice_phrase(body_part)
    article_area = _article_for_body_area(body_area)

    edges = cv2.Canny(gray, 45, 135)
    edge_density = float(edges.mean()) / 255.0
    left_mean = float(gray[:, :w // 2].mean()) if w > 1 else mean_intensity
    right_mean = float(gray[:, w // 2:].mean()) if w > 1 else mean_intensity
    asymmetry = abs(left_mean - right_mean)
    top_mean = float(gray[:h // 2, :].mean()) if h > 1 else mean_intensity
    bot_mean = float(gray[h // 2:, :].mean()) if h > 1 else mean_intensity
    tb_diff = abs(top_mean - bot_mean)

    findings = []
    status = "No obvious acute abnormality"
    local_layer_2 = _build_local_xray_finding_layer(gray, body_part)
    if local_layer_2.get("hardware_present") is True:
      status = "Needs attention"

    if contrast < 18:
      status = "Limited / unclear image"
      findings.append("Image quality check: the X-ray is not clear enough for a reliable educational review.")
    elif not is_grayscale_like:
      status = "Limited / unclear image"
      findings.append("Image quality check: the upload looks like a colour photo, so the X-ray review may be limited.")
    else:
      if _is_chest_body_part(body_part):
        findings.append("Body part check: this appears to be a Chest X-ray, with ribs and lung areas visible.")
        if asymmetry > 32:
          status = "Needs attention"
          findings.append("Chest side-to-side check: one side looks noticeably different in brightness, so a radiologist should review it.")
        else:
          findings.append("Chest side-to-side check: no large one-sided brightness difference was detected by this basic review.")

        if dark_ratio > 0.08:
          findings.append("Lung area check: darker lung areas are visible, which is expected on many chest X-rays, but the official report is needed for meaning.")
        else:
          findings.append("Lung area check: the image does not show strong dark lung-area detail, so review quality may be limited.")

        if bright_ratio > 0.18 or edge_density > 0.08:
          findings.append("Chest detail check: rib and central chest details are visible; this basic tool cannot confirm infection, fluid, or other chest disease.")

      elif _is_orthopedic_body_part(body_part):
        findings.append(f"Body part check: this appears to be a {body_part} X-ray.")

        if local_layer_2.get("hardware_present") is True:
          findings.append(f"Hardware check: {local_layer_2.get('hardware_description', 'possible metallic hardware is visible')}.")
        elif local_layer_2.get("hardware_present") is None:
          findings.append("Hardware check: metal hardware is not confirmed by the local review, so the original image should be checked by a clinician.")

        if asymmetry > 28:
          status = "Needs attention"
          findings.append(_xray_alignment_attention_phrase(view))
        else:
          findings.append("Alignment check: no obvious major bone displacement is seen in this basic review.")

        if contrast >= 25 and edge_density >= 0.035:
          if status == "No obvious acute abnormality":
            findings.append("Broken bone check: no obvious large break line is seen by this basic review, but small cracks can be missed.")
          else:
            findings.append("Broken bone check: a small crack or break cannot be ruled out from this basic review, so medical review is important.")
        else:
          status = "Limited / unclear image" if status != "Needs attention" else status
          findings.append("Broken bone check: image detail is limited, so a fracture cannot be ruled out by this tool.")

        bp = body_part.lower()
        if "knee" in bp:
          findings.append("Joint check: the knee joint area is visible; a clinician should check joint spacing, swelling, and small cracks if pain is present.")
        elif any(x in bp for x in ["hand", "wrist", "foot", "ankle"]):
          findings.append("Small-bone check: this area has many small bones, so a radiologist should check carefully for tiny cracks or joint injury.")
        elif any(x in bp for x in ["arm", "forearm", "elbow", "shoulder", "leg", "hip", "pelvis"]):
          findings.append("Bone-position check: the main bone outlines are visible; a doctor should confirm there is no subtle fracture or joint injury.")
        elif any(x in bp for x in ["spine", "skull", "head"]):
          findings.append("Special-area check: this body area needs careful medical reading, especially after injury or severe symptoms.")

        if tb_diff > 32:
          findings.append("Soft-tissue/position check: brightness changes around the area may be from positioning, swelling, or normal X-ray exposure differences.")

      else:
        findings.append("Body part check: the system could not clearly identify which body part is shown.")
        findings.append("Abnormality check: a doctor or radiologist should review the original image because the body area is unclear.")
        if bright_ratio > 0.08 or mid_grey_ratio > 0.45:
          findings.append("X-ray detail check: bone-like and soft-tissue-like areas are visible, but the meaning cannot be confirmed by this tool.")

      if len(findings) < 3:
        if status == "No obvious acute abnormality":
          findings.append(f"Overall check: no obvious urgent-looking problem was detected in this basic review of {article_area}.")
        else:
          findings.append(f"Review advice: show this X-ray to a qualified doctor or radiologist, especially if you have {symptom_phrase}.")

    while len(findings) < 3:
      findings.append("A qualified radiologist should review the original image for a proper medical reading.")

    if status == "No obvious acute abnormality":
      if _is_orthopedic_body_part(body_part):
        impression = f"This looks like a {body_area} X-ray, and no obvious large broken bone or major dislocation was detected by this basic review."
        simple = (
          f"This is a simple educational review of your uploaded {body_area} X-ray. "
          "The system checked the body part, bone alignment, obvious break lines, and the joint or nearby soft-tissue area. "
          "It did not see an obvious urgent-looking bone problem, but only a doctor or radiologist can confirm the result."
        )
      elif _is_chest_body_part(body_part):
        impression = "This looks like a Chest X-ray, and this basic review did not detect a large urgent-looking chest difference."
        simple = (
          "This is a simple educational review of your uploaded Chest X-ray. "
          "The system checked whether the image looks like a chest view and compared the two lung sides in a basic way. "
          "It cannot confirm infection, fluid, or other chest disease; a radiologist or doctor must read the X-ray."
        )
      else:
        impression = f"This looks like an X-ray of {article_area}, and this basic review did not find an obvious urgent-looking issue."
        simple = (
          "This is a simple educational review of your uploaded X-ray. "
          "The system looked at image clarity, contrast, and visible body structures. "
          "Only a doctor or radiologist can confirm what the X-ray means."
        )
    elif status == "Needs attention":
      impression = f"This looks like a {body_area} X-ray, and one or more visible checks should be reviewed by a doctor."
      simple = (
        f"This is a simple educational review of your uploaded {body_area} X-ray. "
        "The system noticed a visible difference in alignment, brightness, or image pattern. "
        f"This does not prove a diagnosis, but you should show it to a doctor or radiologist, especially if you have {symptom_phrase}."
      )
    else:
      impression = "The uploaded X-ray image is limited or unclear, so a reliable simple summary cannot be confirmed."
      simple = (
        "This image could not be reviewed clearly by the local system. "
        "Please upload a clearer X-ray image or use the official radiology report. "
        "A doctor or radiologist should review the original image."
      )

    raw_note = (
      f"Local image check only. Mean brightness {mean_intensity:.0f}/255, contrast {contrast:.1f}, "
      f"dark area {dark_ratio:.0%}, bright area {bright_ratio:.0%}, edge density {edge_density:.0%}, {conf_note}."
    )

    return normalize_xray_result({
      "exam_type": f"{body_part} X-ray" if "unknown" not in body_part.lower() else "X-ray image (local pixel analysis)",
      "body_part": body_part,
      "view": view,
      "status": status,
      "layer_1_safety": {
        "body_part": body_part,
        "view": view,
        "image_quality": "reviewable" if status != "Limited / unclear image" else "limited/unclear",
        "status": status,
        "doctor_review_required": status in {"Needs attention", "Limited / unclear image"},
        "urgency_reason": "Specific local image checks need clinician review." if status == "Needs attention" else "Educational local review only.",
      },
      "layer_2_findings": local_layer_2,
      "overall_impression": impression,
      "key_findings": findings[:6],
      "simple_explanation": simple,
      "caution": (
        "Educational use only. This is not a confirmed diagnosis. "
        "Please consult a qualified doctor or radiologist for medical decisions."
      ),
      "raw_note": raw_note,
    })
  except Exception as e:
    return normalize_xray_result({
      "exam_type": "X-ray image",
      "body_part": "Unknown",
      "view": "Unknown",
      "status": "Limited / unclear image",
      "overall_impression": f"Local image analysis could not be completed: {str(e)}",
      "key_findings": ["Image processing error during local analysis."],
      "simple_explanation": "Please try uploading the image again.",
      "caution": "This is educational only and not a confirmed diagnosis."
    })



def _local_body_part_check_for_result(image_path):
  """Run only the local body-part detector and return a small evidence dict."""
  try:
    img = cv2.imread(image_path)
    if img is None:
      return {}
    gray_candidate = _focus_xray_region_from_image(img)
    if gray_candidate is None or gray_candidate.size == 0:
      return {}
    gray = _crop_to_xray_content(gray_candidate)
    h, w = gray.shape[:2]
    body_part, view, note = _detect_body_part_locally(gray, h, w)
    local_score_match = re.search(r"local detector score:\s*(-?\d+)", safe_str(note))
    chest_score_match = re.search(r"chest projection score:\s*(-?\d+)", safe_str(note))
    local_score = int(local_score_match.group(1)) if local_score_match else 0
    chest_score = int(chest_score_match.group(1)) if chest_score_match else 0
    return {
      "body_part": body_part,
      "view": view,
      "note": note,
      "local_score": local_score,
      "chest_score": chest_score,
    }
  except Exception:
    return {}


def _correct_xray_body_part_with_local_detector(result, image_path):
  """Conservatively fix obvious body-part mismatches from the vision response."""
  result = normalize_xray_result(result)
  evidence = _local_body_part_check_for_result(image_path)
  if not evidence:
    return result

  model_body = safe_str(result.get("body_part", ""))
  model_body_lower = model_body.lower()
  local_body = safe_str(evidence.get("body_part", ""))
  local_lower = local_body.lower()
  local_score = int(evidence.get("local_score", 0) or 0)
  chest_score = int(evidence.get("chest_score", 0) or 0)

  if not local_body or "unknown" in local_lower:
    return result

  should_override = False
  if "unknown" in model_body_lower or "unclear" in model_body_lower:
    should_override = local_score >= 5
  elif _is_chest_body_part(local_body) and not _is_chest_body_part(model_body):
    should_override = chest_score >= 8
  elif _is_chest_body_part(model_body) and not _is_chest_body_part(local_body):
    # Helps knee/hand/arm uploads that a model accidentally labels as chest.
    should_override = local_score >= 7 and chest_score < 7
  elif _is_orthopedic_body_part(local_body) and _is_orthopedic_body_part(model_body):
    # If the main model says one bone/joint study but the local image layout clearly
    # looks like another orthopedic region, prefer the stronger local body-part label.
    if local_lower != model_body_lower:
      if "knee" in model_body_lower and any(tok in local_lower for tok in ["hand", "wrist", "foot", "ankle", "arm", "forearm", "elbow", "shoulder"]):
        should_override = local_score >= 4
      else:
        should_override = local_score >= 6

  local_view = _sanitize_xray_view_label(evidence.get("view") or "Unknown", local_body)
  model_view = safe_str(result.get("view", "Unknown"))
  local_family = _body_part_family_label(local_body)
  model_family = _body_part_family_label(model_body)

  # Strong mismatch rule: use local only when it is very strong. For moderate
  # disagreement, do not force another body part; the final confidence gate will
  # show Needs review instead of a confident wrong label.
  if local_family != "unknown" and model_family != "unknown" and local_family != model_family:
    if local_score >= 8:
      should_override = True
    else:
      return _xray_set_body_region_needs_review(
        result,
        f"Auto-detect body-region mismatch between model label ({model_body}) and local image check ({local_body}, score {local_score})."
      )

  same_body_family = (
    local_family == model_family and local_family != "unknown"
  )

  # Even when the body-part label is already correct, fix a combined/ambiguous
  # projection label such as "AP / Lateral view" if the local image layout gives
  # a more specific single-view label like "Lateral".
  should_update_view = (
    local_view not in {"", "Unknown", "Single view - projection uncertain"}
    and (_combined_or_ambiguous_xray_view(model_view) or "single view" in model_view.lower())
    and (should_override or same_body_family)
  )

  if not should_override and not should_update_view:
    return result

  findings = [safe_str(x) for x in result.get("key_findings", []) if safe_str(x).strip()]

  if should_override:
    old_body = result.get("body_part", "Unknown")
    result["body_part"] = local_body
    result["exam_type"] = f"{local_body} X-ray"
    result["raw_note"] = (safe_str(result.get("raw_note", "")).strip() + " " +
               f"Body-part label adjusted from {old_body} using local image layout check.").strip()
    correction_finding = f"Body part check: this image layout appears more consistent with a {local_body} X-ray."
    if not any("body part check" in x.lower() for x in findings):
      findings.insert(0, correction_finding)

  if should_update_view:
    old_view = result.get("view", "Unknown")
    result["view"] = local_view
    layer1 = result.get("layer_1_safety", {}) if isinstance(result.get("layer_1_safety", {}), dict) else {}
    layer1["view"] = local_view
    result["layer_1_safety"] = layer1
    result["raw_note"] = (safe_str(result.get("raw_note", "")).strip() + " " +
               f"View label adjusted from {old_view} to {local_view} using local single-image projection check.").strip()
    view_finding = f"View check: this appears to be a single {local_view} projection, not multiple AP/lateral views."
    if not any("view check" in x.lower() for x in findings):
      findings.insert(0, view_finding)

  result["key_findings"] = findings[:6]
  return normalize_xray_result(result)

def analyze_xray_image(image_path, selected_xray_region="Auto-detect"):
  try:
    if not image_path:
      return fallback_xray_result()

    cache_key = _xray_result_cache_key(image_path, selected_xray_region)
    cached_result = _xray_get_cached_result(cache_key)
    if cached_result is not None:
      return cached_result

    if os.path.exists(image_path):
      film_photo_screen = detect_xray_film_photo_or_composite_upload(image_path)
      if film_photo_screen.get("is_limited_upload"):
        return _xray_store_cached_result(
          cache_key,
          build_xray_film_photo_needs_review_result(image_path, film_photo_screen)
        )

    quality = assess_xray_image_quality(image_path) if os.path.exists(image_path) else {"ok": False, "blurry": True, "message": BLURRY_XRAY_IMAGE_ERROR}
    if quality.get("blurry"):
      return _xray_store_cached_result(cache_key, {
        "exam_type": "X-ray image",
        "body_part": "Unknown",
        "view": "Unknown",
        "status": "Limited / unclear image",
        "overall_impression": quality.get("message", BLURRY_XRAY_IMAGE_ERROR),
        "key_findings": ["Image quality check: the uploaded X-ray appears blurred or too unclear for a reliable review."],
        "simple_explanation": quality.get("message", BLURRY_XRAY_IMAGE_ERROR),
        "caution": "Please upload a sharper X-ray image. This educational tool should not be used on blurred images.",
        "layer_1_safety": {
          "image_quality": "Blurred",
          "status": "Limited / unclear image",
          "doctor_review_required": True,
          "urgency_reason": "The uploaded image is too blurred or unclear for a reliable review.",
        },
      })

    if client is None:
      # No GROQ API key use local OpenCV-based pixel analysis, then run
      # the same universal safety gate used for vision-model output.
      study_meta = detect_xray_study_metadata(image_path)
      result = _analyze_xray_locally(image_path)
      result = _apply_study_metadata_hint(result, study_meta)
      result = _correct_xray_body_part_with_local_detector(result, image_path)
      result = _sanitize_low_confidence_hardware_claims(result)
      result = _apply_universal_xray_image_safety_gate(result, image_path)
      result = _xray_body_region_confidence_gate(result, image_path, study_meta, selected_xray_region)
      return _xray_store_cached_result(cache_key, result)

    if not os.path.exists(image_path):
      return normalize_xray_result({
        "exam_type": "X-ray image",
        "body_part": "Unknown",
        "view": "Unknown",
        "status": "Limited / unclear image",
        "overall_impression": "Uploaded X-ray file was not found.",
        "key_findings": ["The uploaded file path appears invalid or expired."],
        "simple_explanation": "Please upload the image again and rerun the analysis.",
        "caution": "This is educational only and not a confirmed diagnosis."
      })

    study_meta = detect_xray_study_metadata(image_path)
    if study_meta.get("blur_flag") or safe_str(study_meta.get("image_quality", "")).lower().startswith("blur"):
      return _xray_store_cached_result(cache_key, {
        "exam_type": "X-ray image",
        "body_part": safe_str(study_meta.get("body_part", "Unknown")) or "Unknown",
        "view": safe_str(study_meta.get("view", "Unknown")) or "Unknown",
        "status": "Limited / unclear image",
        "overall_impression": BLURRY_XRAY_IMAGE_ERROR,
        "key_findings": ["Image quality check: the uploaded X-ray appears blurred or too unclear for a reliable review."],
        "simple_explanation": BLURRY_XRAY_IMAGE_ERROR,
        "caution": "Please upload a sharper X-ray image. This educational tool should not be used on blurred images.",
      })

    image_data_url = encode_image_to_data_url(image_path)
    body_hint = safe_str(study_meta.get("body_part", "Unknown body area")) or "Unknown body area"
    selected_body_hint = _xray_selected_region_to_body_part(selected_xray_region)
    if selected_body_hint:
      body_hint = selected_body_hint
    view_hint = safe_str(study_meta.get("view", "Unknown")) or "Unknown"
    anatomy_hint = ", ".join([safe_str(x) for x in study_meta.get("visible_anatomy", []) if safe_str(x).strip()][:8])
    quality_hint = safe_str(study_meta.get("image_quality", "Acceptable")) or "Acceptable"
    user_region_note = f" User-selected body region = {selected_body_hint}. Treat the image as this body region unless the image is clearly not an X-ray." if selected_body_hint else " Auto-detect mode is enabled; do not guess the body part if confidence is low."
    prompt_text = XRAY_ANALYSIS_PROMPT + f"\n\nPre-analysis study metadata (verify against the image and correct it if needed): likely body part = {body_hint}; likely single-view label = {view_hint}; likely visible anatomy = {anatomy_hint or 'not specified'}; image quality = {quality_hint}.{user_region_note}"

    response = client.chat.completions.create(
      model=VISION_MODEL,
      messages=[
        {
          "role": "system",
          "content": "You are a careful medical image assistant for educational purposes only. Return strict JSON only."
        },
        {
          "role": "user",
          "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": image_data_url}}
          ]
        }
      ],
      temperature=0.1
    )
    raw_text = safe_str(response.choices[0].message.content).strip()
    parsed = extract_json_object(raw_text)
    if not parsed:
      parsed = {
        "exam_type": "X-ray image",
        "body_part": body_hint,
        "view": view_hint,
        "status": "Needs attention",
        "overall_impression": raw_text[:240] if raw_text else "The model returned an unstructured answer.",
        "key_findings": [raw_text[:180] if raw_text else "No structured findings returned."],
        "simple_explanation": raw_text[:400] if raw_text else "No explanation returned.",
        "caution": "This is educational only and not a confirmed diagnosis."
      }
    result = normalize_xray_result(parsed)
    result = _apply_study_metadata_hint(result, study_meta)
    result = _correct_xray_body_part_with_local_detector(result, image_path)
    result = _sanitize_low_confidence_hardware_claims(result)
    result = _apply_universal_xray_image_safety_gate(result, image_path)
    result = _xray_body_region_confidence_gate(result, image_path, study_meta, selected_xray_region)
    return _xray_store_cached_result(cache_key, result)
  except Exception as e:
    return normalize_xray_result({
      "exam_type": "X-ray image",
      "body_part": "Unknown",
      "view": "Unknown",
      "status": "Limited / unclear image",
      "overall_impression": f"Error during X-ray analysis: {str(e)}",
      "key_findings": ["The AI image analysis step failed during processing."],
      "simple_explanation": "Please try the image again. If the problem continues, review the API key or model access.",
      "caution": "This is educational only and not a confirmed diagnosis."
    })


def analyze_xray_for_gradio(uploaded_xray_file, selected_xray_region="Auto-detect"):
  try:
    if uploaded_xray_file is None:
      empty = fallback_xray_result()
      return "", build_xray_visual_html(empty), build_xray_markdown(empty)

    if isinstance(uploaded_xray_file, str):
      image_path = uploaded_xray_file
    elif hasattr(uploaded_xray_file, "name"):
      image_path = uploaded_xray_file.name
    else:
      empty = normalize_xray_result({
        "overall_impression": "Invalid uploaded file.",
        "key_findings": ["The uploaded object could not be read as an image."],
        "simple_explanation": "Please upload a PNG or JPG X-ray image."
      })
      return None, build_xray_visual_html(empty), build_xray_markdown(empty)

    # X-ray validation: when Groq vision is available, use it as the source of truth.
    # If API is unavailable, fall back to the strict local radiograph-style check.
    vision_validation = validate_xray_with_vision_model(image_path)
    local_validation = looks_like_xray_image_file(image_path)

    if vision_validation is False or (vision_validation is None and not local_validation):
      error_result = normalize_xray_result({
        "exam_type": "Invalid image",
        "body_part": "Unknown",
        "view": "Unknown",
        "status": "Limited / unclear image",
        "overall_impression": NON_XRAY_IMAGE_ERROR,
        "key_findings": [NON_XRAY_IMAGE_ERROR],
        "simple_explanation": NON_XRAY_IMAGE_ERROR,
        "caution": "Please upload a real medical X-ray image."
      })
      return None, build_validation_error_html(NON_XRAY_IMAGE_ERROR), f"### Upload Error\n\n{NON_XRAY_IMAGE_ERROR}"

    result = analyze_xray_image(image_path, selected_xray_region)
    log_analysis_event({
      "analysis_type": "X-ray",
      "file_name": os.path.basename(image_path),
      "report_category": "Radiology / X-ray",
      "report_subtype": result.get("exam_type", "X-ray") if isinstance(result, dict) else "X-ray",
      "risk_level": result.get("status", "Unknown") if isinstance(result, dict) else "Unknown",
      "risk_score": 0,
      "ocr_quality": "Image analysis",
      "ocr_score": 0,
      "total_tests": 0,
      "abnormal_count": 1 if isinstance(result, dict) and "attention" in safe_str(result.get("status", "")).lower() else 0,
      "status": "success"
    })
    return image_path, build_xray_visual_html(result), build_xray_markdown(result)
  except Exception as e:
    empty = normalize_xray_result({
      "overall_impression": f"X-ray UI error: {str(e)}",
      "key_findings": ["The system could not render the X-ray summary."],
      "simple_explanation": "Please try uploading the image again."
    })
    return None, build_xray_visual_html(empty), build_xray_markdown(empty)





# 
# CELL 4 OF 5 MAIN FUNCTIONS
# 

def analyze_medical_report_ui(uploaded_file, pdf_language="English"):
  pdf_language = "English" # English-only PDF export
  empty_df = pd.DataFrame(columns=["Parameter", "Value", "Unit", "Reference Range", "Status", "% in Range", "Severity"])
  empty_visual_html = build_lab_visual_html(empty_df, report_category="Unknown")
  empty_summary_html = build_summary_card_html("Unknown", "Unknown", {}, empty_df, {})
  empty_sections_md = "**Detected Sections / Findings**\n\n- No report analyzed yet."
  empty_json = "{}"

  def fail(message, raw_text="", formatted_text="", patient_info_md="", report_type_display=""):
    return (
      message,
      report_type_display,
      patient_info_md,
      raw_text,
      formatted_text,
      empty_visual_html,
      empty_sections_md,
      "",
      empty_summary_html,
      empty_json,
      None,
      None,
    )

  if not uploaded_file:
    return fail("Please upload a PDF or image file.")

  try:
    analysis_start_time = time.time()
    file_path = uploaded_file if isinstance(uploaded_file, str) else getattr(uploaded_file, "name", None)
    valid_file, validation_message = validate_uploaded_medical_file(file_path)
    if not valid_file:
      return fail(validation_message)

    file_hash = calculate_file_hash(file_path)
    cache_key = f"{PARSER_VERSION}:{file_hash}" if file_hash else ""
    if cache_key and cache_key in REPORT_ANALYSIS_CACHE:
      cached_outputs = REPORT_ANALYSIS_CACHE[cache_key]
      cached_outputs = list(cached_outputs)
      cached_outputs[0] = "Report loaded instantly from cache."
      return tuple(cached_outputs)

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
      raw_text = extract_text_from_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
      raw_text = extract_text_from_image(file_path)
    else:
      return fail("Unsupported file format. Please upload PDF, PNG, JPG, or JPEG.")

    if not raw_text or raw_text.startswith("OCR Error") or raw_text.startswith("PDF Extraction Error"):
      return fail("We could not read this file clearly. Please upload a clearer image or PDF.", raw_text=raw_text)

    formatted_text = clean_text(raw_text)
    ocr_quality = compute_ocr_quality_score(raw_text, file_path=file_path)
    ml_classifier = classify_report_type_ml_style(formatted_text)
    report_meta = detect_report_type(formatted_text, file_path=file_path)
    report_category = report_meta["category"]
    report_subtype = report_meta["subtype"]
    report_type_display = f"{report_category} {report_subtype}"

    patient_info = extract_patient_info(formatted_text)
    patient_info_md = format_patient_info_md(patient_info)

    parsed_records = parse_lab_records(formatted_text)
    lab_df = build_lab_dataframe(parsed_records)

    if not looks_like_medical_report_text(formatted_text, parsed_records, report_category):
      reset_report_context()
      return fail(NON_MEDICAL_REPORT_ERROR, raw_text=raw_text, formatted_text=formatted_text)

    risk_score = build_risk_score(lab_df.to_dict(orient="records") if lab_df is not None else [])
    health_suggestions = generate_health_suggestions(lab_df.to_dict(orient="records") if lab_df is not None else [])

    # Re-check classification after dynamic extraction so lab-style reports are not mis-labeled.
    if (report_category != "Laboratory Report") and (lab_df is not None and len(lab_df) >= 2):
      report_category = "Laboratory Report"
      if any(k in formatted_text.lower() for k in ["wbc", "rbc", "hemoglobin", "haemoglobin", "platelet", "cbc"]):
        report_subtype = "CBC / Hematology Report"
      else:
        report_subtype = "General Laboratory Report"
      report_type_display = f"{report_category} {report_subtype}"

    detected_sections = {}
    if report_category == "Radiology Report":
      detected_sections = parse_radiology_sections(formatted_text)
    elif report_category == "General Medical Report":
      detected_sections = parse_general_medical_sections(formatted_text)
    elif report_category == "Laboratory Report" and lab_df.empty:
      detected_sections = parse_general_medical_sections(formatted_text)

    sections_md = format_detected_sections_md(detected_sections, report_category)
    visual_html = build_lab_visual_html(lab_df, report_category=report_category)

    ai_explanation = generate_ai_explanation(
      formatted_text,
      report_category,
      report_subtype,
      lab_df,
      detected_sections
    )

    final_summary = generate_final_report_summary(
      report_category,
      report_subtype,
      patient_info,
      lab_df,
      detected_sections
    )
    summary_html = build_summary_card_html(
      report_category,
      report_subtype,
      patient_info,
      lab_df,
      detected_sections
    )
    polished_summary_html = build_polished_result_summary_html(
      report_category,
      report_subtype,
      ocr_quality,
      risk_score,
      ml_classifier,
      health_suggestions
    )
    summary_html = polished_summary_html + summary_html

    json_payload = {
      "original_filename": os.path.basename(file_path),
      "report_category": report_category,
      "report_subtype": report_subtype,
      "classification_reason": report_meta.get("reason", ""),
      "ml_classifier": ml_classifier,
      "ocr_quality": ocr_quality,
      "risk_score": risk_score,
      "health_suggestions": health_suggestions,
      "medical_disclaimer": MEDICAL_DISCLAIMER_TEXT,
      "patient_info": patient_info,
      "raw_text": raw_text,
      "formatted_text": formatted_text,
      "lab_records": lab_df.to_dict(orient="records"),
      "radiology_sections": detected_sections,
      "ai_explanation": ai_explanation,
      "summary": final_summary,
    }

    json_file = save_json_report(json_payload)
    pdf_file = save_pdf_report(json_payload, pdf_language)
    # Save extracted lab values locally for the user-side comparison dashboard.
    save_lab_report_snapshot(json_payload)

    CURRENT_REPORT_CONTEXT["raw_text"] = raw_text
    CURRENT_REPORT_CONTEXT["formatted_text"] = formatted_text
    CURRENT_REPORT_CONTEXT["report_category"] = report_category
    CURRENT_REPORT_CONTEXT["report_subtype"] = report_subtype
    CURRENT_REPORT_CONTEXT["patient_info"] = patient_info
    CURRENT_REPORT_CONTEXT["lab_records"] = lab_df.to_dict(orient="records")
    CURRENT_REPORT_CONTEXT["radiology_sections"] = detected_sections
    CURRENT_REPORT_CONTEXT["ai_explanation"] = ai_explanation
    CURRENT_REPORT_CONTEXT["summary"] = final_summary
    CURRENT_REPORT_CONTEXT["ocr_quality"] = ocr_quality
    CURRENT_REPORT_CONTEXT["risk_score"] = risk_score
    CURRENT_REPORT_CONTEXT["health_suggestions"] = health_suggestions
    CURRENT_REPORT_CONTEXT["ml_classifier"] = ml_classifier

    log_analysis_event({
      "analysis_type": "Lab Report",
      "file_name": os.path.basename(file_path),
      "report_category": report_category,
      "report_subtype": report_subtype,
      "risk_level": risk_score.get("level", "Unknown") if isinstance(risk_score, dict) else "Unknown",
      "risk_score": risk_score.get("score", 0) if isinstance(risk_score, dict) else 0,
      "ocr_quality": ocr_quality.get("label", "Unknown") if isinstance(ocr_quality, dict) else "Unknown",
      "ocr_score": ocr_quality.get("score", 0) if isinstance(ocr_quality, dict) else 0,
      "total_tests": int(len(lab_df)) if hasattr(lab_df, "__len__") else 0,
      "abnormal_count": int(len(lab_df[lab_df["Status"].astype(str).str.lower().isin(["high", "low", "abnormal", "critical"])])) if hasattr(lab_df, "columns") and "Status" in lab_df.columns else 0,
      "status": "success"
    })

    elapsed_seconds = max(0.0, time.time() - analysis_start_time)
    success_outputs = (
      f"Report analyzed successfully in {elapsed_seconds:.1f} seconds.",
      report_type_display,
      patient_info_md,
      raw_text,
      formatted_text,
      visual_html,
      sections_md,
      ai_explanation,
      summary_html,
      json.dumps(json_payload, indent=2, ensure_ascii=False),
      pdf_file,
      json_file
    )
    if cache_key:
      REPORT_ANALYSIS_CACHE[cache_key] = success_outputs
    return success_outputs

  except Exception as e:
    return fail("Something went wrong while analyzing this report. Please try a clearer file or a different PDF/image.")

def ask_question_about_report(question, chat_history):
  """Report-aware chatbot for lab and X-ray results."""
  if chat_history is None:
    chat_history = []

  if not question or not str(question).strip():
    return chat_history, ""

  question = str(question).strip()

  if not CURRENT_REPORT_CONTEXT.get("formatted_text") and not CURRENT_REPORT_CONTEXT.get("ai_explanation"):
    chat_history = chat_history + [
      {"role": "user", "content": question},
      {"role": "assistant", "content": " Please analyze a lab report or X-ray first. Then I can answer your questions using your actual report data."}
    ]
    return chat_history, ""

  # Collect all report data 
  lab_records    = CURRENT_REPORT_CONTEXT.get("lab_records", []) or []
  patient_info    = CURRENT_REPORT_CONTEXT.get("patient_info", {}) or {}
  formatted_text   = CURRENT_REPORT_CONTEXT.get("formatted_text", "") or ""
  ai_explanation   = CURRENT_REPORT_CONTEXT.get("ai_explanation", "") or ""
  summary      = CURRENT_REPORT_CONTEXT.get("summary", "") or ""
  radiology_sections = CURRENT_REPORT_CONTEXT.get("radiology_sections", {}) or {}
  report_category  = CURRENT_REPORT_CONTEXT.get("report_category", "") or ""
  report_subtype   = CURRENT_REPORT_CONTEXT.get("report_subtype", "") or ""

  abnormal_records, normal_records, unknown_records = [], [], []
  for row in lab_records:
    status = safe_str(row.get("Status", "")).strip().lower()
    compact = {
      "Parameter":    safe_str(row.get("Parameter", "")),
      "Value":      safe_str(row.get("Value", "")),
      "Unit":      safe_str(row.get("Unit", "")),
      "Reference Range": safe_str(row.get("Reference Range", "")),
      "Status":     safe_str(row.get("Status", "")),
      "Severity":    safe_str(row.get("Severity", "")),
    }
    if status in ["high", "low", "abnormal", "critical", "needs attention"]:
      abnormal_records.append(compact)
    elif status == "normal":
      normal_records.append(compact)
    else:
      unknown_records.append(compact)

  # Build chat context for multi-turn memory 
  chat_context = []
  for msg in chat_history[-6:]:
    role  = msg.get("role", "")  if isinstance(msg, dict) else ""
    content = msg.get("content", "") if isinstance(msg, dict) else safe_str(msg)
    if content:
      chat_context.append({"role": role, "content": safe_str(content)[:600]})

  # Markdown table helper 
  def markdown_table(records, max_rows=20):
    if not records:
      return "_No matching values found._"
    rows = ["| Parameter | Value | Unit | Reference Range | Status |",
        "|-----------|-------|------|-----------------|--------|"]
    for r in records[:max_rows]:
      rows.append(
        f"| {r.get('Parameter','')} | {r.get('Value','')} | "
        f"{r.get('Unit','')} | {r.get('Reference Range','')} | {r.get('Status','')} |"
      )
    return "\n".join(rows)

  # Smart local fallback: tries to answer directly from data 
  q_lower = question.lower()

  def _pi(keys):
    """Search patient_info dict for any of the given keys."""
    for k in keys:
      for pk, pv in patient_info.items():
        if k in str(pk).lower() and safe_str(pv).strip() not in ("", "not found", "n/a"):
          return safe_str(pv).strip()
    return ""

  def _search_text(patterns):
    for pat in patterns:
      m = re.search(pat, formatted_text, re.IGNORECASE)
      if m:
        val = safe_str(m.group(1)).strip(" :,-\n")
        if val:
          return val
    return ""

  # Patient name quick answer
  if any(w in q_lower for w in ["patient name", "patient ka naam", "naam kya", "name kya", "patient naam", "name?"]):
    name = _pi(["patient name", "name"]) or _search_text([
      r"patient\s*name\s*[:\-]?\s*([A-Za-z .]+)",
      r"name\s*[:\-]?\s*([A-Za-z .]+)"
    ])
    if name:
      fallback_answer = f"The patient name is **{name}**.\n\n_Please consult a qualified doctor for medical advice._"
    else:
      fallback_answer = "Patient name was not clearly found in the uploaded report.\n\n_Please consult a qualified doctor for medical advice._"

  # Age quick answer
  elif any(w in q_lower for w in ["age", "umar", "umra"]):
    age = _pi(["age"]) or _search_text([r"age\s*[:\-]?\s*(\d{1,3})"])
    if age:
      fallback_answer = f"The patient's age is **{age}**.\n\n_Please consult a qualified doctor for medical advice._"
    else:
      fallback_answer = "Patient age was not clearly found in the uploaded report."

  # Gender quick answer
  elif any(w in q_lower for w in ["gender", "sex", "male", "female"]):
    gender = _pi(["gender", "sex"]) or _search_text([r"(?:gender|sex)\s*[:\-]?\s*(male|female|m\b|f\b)"])
    if gender:
      fallback_answer = f"The patient's gender is **{gender}**.\n\n_Please consult a qualified doctor for medical advice._"
    else:
      fallback_answer = "Patient gender was not clearly found in the uploaded report."

  # Abnormal / table / what's wrong
  elif any(w in q_lower for w in ["abnormal", "table", "wrong", "problem", "issue", "serious", "attention", "high", "low"]):
    fallback_answer = (
      f"### Values Needing Attention ({len(abnormal_records)} found)\n\n"
      f"{markdown_table(abnormal_records)}\n\n"
      f"_Please consult a qualified doctor for medical advice._"
    )

  # General fallback
  else:
    fallback_answer = (
      f"Here is a summary of your report:\n\n"
      f"- **Report type:** {report_subtype or report_category or 'Unknown'}\n"
      f"- **Total lab tests:** {len(lab_records)}\n"
      f"- **Abnormal values:** {len(abnormal_records)}\n\n"
      f"### Abnormal Values\n\n{markdown_table(abnormal_records)}\n\n"
      f"_Please consult a qualified doctor for medical advice._"
    )

  answer = fallback_answer

  if client is not None:
    try:
      prompt = f"""You are MediBuddy AI a smart, direct, friendly educational medical-report assistant.

CRITICAL RULES READ CAREFULLY:
1. Answer ONLY the specific question asked. Do NOT give a full report summary unless the user asks for it.
2. If the user asks ONE thing (e.g. "patient name") answer ONLY that one thing in 1-2 lines, then add the safety note.
3. If asked patient name just say "The patient name is [name]." nothing else except the safety note.
4. If asked age just say "The patient's age is [age]."
5. If asked about a specific test (hemoglobin etc.) give value, unit, reference range, status in 2-3 lines.
6. If asked for a table / abnormal values / what's wrong give a clean Markdown table.
7. If asked for advice / diet / lifestyle give specific practical advice based on the abnormal values.
8. Answer in the SAME language the user wrote in (Urdu / English / mix). Be warm and concise.
9. Use bold for key values. Use tables ONLY when listing multiple items.
10. ALWAYS end with this exact line: _Please consult a qualified doctor for medical advice._
11. DO NOT start with "Report Summary" or show all report fields unless specifically asked.
12. DO NOT say "I cannot access" you have all the data below.

 REPORT DATA 

Patient Information:
{json.dumps(patient_info, ensure_ascii=False, indent=2)}

Raw Extracted Report Text (search here for name, date, doctor, lab):
{formatted_text[:6000]}

All Lab Records ({len(lab_records)} tests):
{json.dumps(lab_records[:60], ensure_ascii=False, indent=2)}

Abnormal Values ({len(abnormal_records)}):
{json.dumps(abnormal_records[:40], ensure_ascii=False, indent=2)}

Normal Values ({len(normal_records)}):
{json.dumps(normal_records[:40], ensure_ascii=False, indent=2)}

Report Category: {report_category}
Report Type: {report_subtype}
AI Explanation: {ai_explanation[:2000]}
Summary: {summary[:1500]}
Radiology: {json.dumps(radiology_sections, ensure_ascii=False, indent=2)}

Recent Chat:
{json.dumps(chat_context, ensure_ascii=False, indent=2)}

 USER QUESTION 
{question}

Answer the question DIRECTLY and SPECIFICALLY. Do NOT dump the full report. Just answer what was asked."""

      response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=900,
      )
      answer = response.choices[0].message.content.strip()
    except Exception as e:
      answer = fallback_answer + f"\n\n_AI error: {e}_"

  chat_history = chat_history + [
    {"role": "user",   "content": question},
    {"role": "assistant", "content": answer},
  ]
  return chat_history, ""

print("Main functions loaded successfully")





# 
# CELL 5 OF 5 MEDIBUDDY AI GRADIO UI
# Redesigned patient-friendly interface:
# Page 1 Welcome Page 2 Choose option Page 3 Lab result Page 4 X-ray result Page 5 AI assistant
# 

def _ui_escape(value):
  return html.escape(safe_str(value))

def reset_report_context():
  """Clear the current report context so old results do not appear after a failed upload."""
  CURRENT_REPORT_CONTEXT["raw_text"] = ""
  CURRENT_REPORT_CONTEXT["formatted_text"] = ""
  CURRENT_REPORT_CONTEXT["report_category"] = ""
  CURRENT_REPORT_CONTEXT["report_subtype"] = ""
  CURRENT_REPORT_CONTEXT["patient_info"] = {}
  CURRENT_REPORT_CONTEXT["lab_records"] = []
  CURRENT_REPORT_CONTEXT["radiology_sections"] = {}
  CURRENT_REPORT_CONTEXT["ai_explanation"] = ""
  CURRENT_REPORT_CONTEXT["summary"] = ""

def status_tone(status):
  status = safe_str(status).strip().lower()
  if status == "high":
    return "danger"
  if status == "low":
    return "warning"
  if status == "normal":
    return "success"
  return "muted"

def get_uploaded_path(uploaded_file):
  if uploaded_file is None:
    return None
  if isinstance(uploaded_file, str):
    return uploaded_file
  return getattr(uploaded_file, "name", None)

def build_uploaded_file_info(uploaded_file):
  """Small inline file name display below the upload area."""
  path = get_uploaded_path(uploaded_file)
  if not path:
    return gr.update(value="", visible=False)
  file_name = os.path.basename(path)
  try:
    size_kb = os.path.getsize(path) / 1024
    size_label = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB"
  except Exception:
    size_label = "Ready"
  return gr.update(
    value=f"""
    <div class='uploaded-file-inline-box'>
      <span class='uploaded-file-icon'></span>
      <div><b>{_ui_escape(file_name)}</b><small>Uploaded successfully {html.escape(size_label)}</small></div>
    </div>
    """,
    visible=True
  )

def build_placeholder_card(title="No result yet", text="Upload a report and click analyze to see your result here."):
  return f"""
  <div class="empty-result-card saas-empty-state fade-in-up">
    <div class="empty-emoji"></div>
    <h3>{_ui_escape(title)}</h3>
    <p>{_ui_escape(text)}</p>
    <div class="jump-text">All Set</div>
  </div>
  """

def build_patient_info_card(patient_info):
  rows = []
  labels = [
    ("Patient Name", "Name"),
    ("Age", "Age"),
    ("Gender", "Gender"),
    ("Age / Gender", "Age / Gender"),
    ("Report Date", "Report Date"),
    ("Sample Date", "Sample Date"),
    ("Lab No", "Lab No"),
  ]
  used = set()
  for key, label in labels:
    value = safe_str(patient_info.get(key, "")).strip()
    if value and value.lower() not in {"not found", "unknown", "none"} and label not in used:
      rows.append(f"""
      <div class="info-row">
        <span>{_ui_escape(label)}</span>
        <b>{_ui_escape(value)}</b>
      </div>
      """)
      used.add(label)

  if not rows:
    rows.append("""
    <div class="info-row">
      <span>Patient details</span>
      <b>Not detected from this file</b>
    </div>
    """)

  return f"""
  <div class="soft-card">
    <div class="card-title">Patient Information</div>
    {''.join(rows[:5])}
  </div>
  """

def build_report_type_card(report_category, report_subtype):
  return f"""
  <div class="soft-card">
    <div class="card-title">Detected Report Category and Type</div>
    <div class="info-row"><span>Category</span><b>{_ui_escape(report_category or "Unknown")}</b></div>
    <div class="info-row"><span>Type</span><b>{_ui_escape(report_subtype or "Unknown")}</b></div>
  </div>
  """

def build_ai_explanation_card(ai_explanation):
  text = safe_str(ai_explanation).strip()
  if not text:
    text = "AI explanation will appear here after analysis."
  text = _ui_escape(text).replace("\n", "<br>")
  return f"""
  <div class="soft-card magic-card ai-card">
    <div class="card-title">AI Medical Explanation</div>    <div class="ai-bubble">
      <div class="mini-robot"></div>
      <p>{text}</p>
    </div>
  </div>
  """

def build_health_gauge(total, normal, needs_attention):
  if total <= 0:
    score = 0
  else:
    score = int(round((normal / total) * 100))

  if total <= 0:
    label = "Waiting for report"
    face = ""
    tone = "muted"
  elif needs_attention <= 0:
    label = "Normal"
    face = ""
    tone = "success"
  elif needs_attention <= max(1, total * 0.25):
    label = "Needs Attention"
    face = ""
    tone = "warning"
  else:
    label = "Needs Doctor Review"
    face = ""
    tone = "danger"

  return f"""
  <div class="soft-card magic-card health-card">
    <div class="card-title">Your Health Summary</div>
    <div class="gauge-wrap">
      <div class="gauge gauge-{tone}" style="--score:{score * 1.8}deg;">
        <div class="gauge-face">{face}</div>
      </div>
    </div>
    <h3>{_ui_escape(label)}</h3>
    <p>{score}% values are marked normal based on detected reference ranges.</p>
  </div>
  """

def build_abnormal_panel(lab_records):
  if not lab_records:
    return """
    <div class="soft-card">
      <div class="card-title">Abnormal / Needs Attention</div>
      <div class="friendly-note">No structured lab values have been detected yet.</div>
    </div>
    """

  rows = []
  abnormal = []
  for idx, row in enumerate(lab_records, start=1):
    status = safe_str(row.get("Status", "Unknown")).title()
    if status != "Normal":
      abnormal.append((idx, row, status))

  if not abnormal:
    return """
    <div class="soft-card">
      <div class="card-title">Abnormal / Needs Attention</div>
      <div class="normal-box"> No abnormal lab value was detected in the extracted table.</div>
    </div>
    """

  for idx, row, status in abnormal[:8]:
    label, _ = get_display_parameter_labels(row.get("Parameter", ""), idx)
    value = compact_value_label(row)
    tone = status_tone(status)
    rows.append(f"""
    <div class="ab-row">
      <span>{_ui_escape(label)}</span>
      <b>{_ui_escape(value)}</b>
      <em class="pill pill-{tone}">{_ui_escape(status)}</em>
    </div>
    """)

  return f"""
  <div class="soft-card">
    <div class="card-title">Abnormal / Needs Attention</div>
    <div class="ab-list">{''.join(rows)}</div>
  </div>
  """


def build_report_suggestions(records):
  """Simple educational suggestions based on detected lab statuses."""
  if not records:
    return """
    <div class="soft-card magic-card suggestion-card">
      <div class="card-title">Suggestions</div>
      <ul><li>Upload a clear report and run analysis to receive report-based suggestions.</li></ul>
    </div>
    """

  high_items, low_items = [], []
  for idx, row in enumerate(records, start=1):
    status = safe_str(row.get("Status", "")).title()
    label, _ = get_display_parameter_labels(row.get("Parameter", ""), idx)
    if status == "High":
      high_items.append(label)
    elif status == "Low":
      low_items.append(label)

  suggestions = []
  if high_items:
    suggestions.append(f"Some values are high: <b>{_ui_escape(', '.join(high_items[:4]))}</b>. Discuss these with a doctor, especially if you have symptoms.")
    suggestions.append("Stay hydrated, avoid self-medication, and keep a copy of this report for your healthcare provider.")
  if low_items:
    suggestions.append(f"Some values are low: <b>{_ui_escape(', '.join(low_items[:4]))}</b>. Ask your doctor if diet, supplements, or follow-up testing is needed.")
  if not suggestions:
    suggestions.append("Most detected values look within range. Keep routine checkups and maintain healthy sleep, diet, and activity.")
  suggestions.append("This app is educational only and does not replace professional medical advice.")

  items = "".join(f"<li>{item}</li>" for item in suggestions)
  return f"""
  <div class="soft-card magic-card suggestion-card">
    <div class="card-title">Personalized Suggestions</div>
    <ul>{items}</ul>
  </div>
  """


def build_lab_dashboard_html(report_type_display, ai_explanation, lab_table_html):
  category = CURRENT_REPORT_CONTEXT.get("report_category", "") or "Unknown"
  subtype = CURRENT_REPORT_CONTEXT.get("report_subtype", "") or "Unknown"
  patient_info = CURRENT_REPORT_CONTEXT.get("patient_info", {}) or {}
  records = CURRENT_REPORT_CONTEXT.get("lab_records", []) or []

  total = len(records)
  normal = sum(1 for r in records if safe_str(r.get("Status", "")).title() == "Normal")
  high = sum(1 for r in records if safe_str(r.get("Status", "")).title() == "High")
  low = sum(1 for r in records if safe_str(r.get("Status", "")).title() == "Low")
  unknown = max(0, total - normal - high - low)
  needs_attention = high + low + unknown

  attention_rows = []
  for row in records:
    status = safe_str(row.get("Status", "Unknown")).title()
    if status != "Normal":
      parameter = safe_str(row.get("Parameter", "Value"))
      value = safe_str(row.get("Value", ""))
      ref = safe_str(row.get("Reference Range", row.get("Normal Range", "")))
      tone = "danger" if status == "High" else "warning" if status == "Low" else "muted"
      attention_rows.append(f"""
        <div class="lab-attention-item">
          <div><b>{_ui_escape(parameter)}</b><span>{_ui_escape(value)} Ref: {_ui_escape(ref or "Not listed")}</span></div>
          <em class="pill pill-{tone}">{_ui_escape(status)}</em>
        </div>
      """)
  if not attention_rows:
    attention_html = '<div class="normal-box"> No abnormal value was detected in the extracted lab table.</div>'
  else:
    attention_html = ''.join(attention_rows[:8])

  normal_width = int((normal / max(total, 1)) * 100)
  low_width = int((low / max(total, 1)) * 100)
  high_width = int((high / max(total, 1)) * 100)
  clean_ai = _ui_escape(ai_explanation or 'AI explanation will appear here after analysis.').replace(chr(10), '<br>')
  patient_card = build_patient_info_card(patient_info).replace('<div class="soft-card">', '<div class="inner-card">', 1)
  type_card = build_report_type_card(category, subtype).replace('<div class="soft-card">', '<div class="inner-card">', 1)

  ocr_quality = CURRENT_REPORT_CONTEXT.get("ocr_quality", {}) or compute_ocr_quality_score(CURRENT_REPORT_CONTEXT.get("raw_text", ""))
  risk_score = CURRENT_REPORT_CONTEXT.get("risk_score", {}) or build_risk_score(records)
  ml_classifier = CURRENT_REPORT_CONTEXT.get("ml_classifier", {}) or classify_report_type_ml_style(CURRENT_REPORT_CONTEXT.get("formatted_text", ""))
  health_suggestions = CURRENT_REPORT_CONTEXT.get("health_suggestions", []) or generate_health_suggestions(records)
  top_summary_html = build_polished_result_summary_html(
    category,
    subtype,
    ocr_quality,
    risk_score,
    ml_classifier,
    health_suggestions,
  )

  return f"""
  <div class="result-page lab-redesign">
    <div class="lab-hero-head">
      <div>
        <div class="step-kicker">Step 3 of 3 Analyze + Chat</div>
        <h2>Lab Report Analysis</h2>
        <p>Your uploaded medical report has been decoded into a clean, readable dashboard.</p>
      </div>
      <div class="complete-chip green-chip"> Analysis Completed</div>
    </div>

    <div class="wide-section visible-polish-summary">
      {top_summary_html}
    </div>

    <div class="stats-grid lab-stats-modern">
      <div class="stat-card green"><span>Total Parameter</span><b>{total}</b></div>
      <div class="stat-card green"><span>Normal</span><b>{normal}</b></div>
      <div class="stat-card orange"><span>Needs Attention</span><b>{needs_attention}</b></div>
      <div class="stat-card red"><span>High</span><b>{high}</b></div>
    </div>


    <div class="lab-main-grid">
      <div class="lab-info-stack">
        <div class="soft-card magic-card info-clean-card">
          <div class="card-title">Patient Information</div>
          {patient_card}
        </div>

        <div class="soft-card magic-card info-clean-card">
          <div class="card-title">Detected Report Category and Type</div>
          {type_card}
        </div>
      </div>

      <div class="soft-card magic-card attention-card">
        <div class="card-title">Abnormal / Needs Attention</div>
        <p class="card-subtitle">Values outside the extracted reference range are highlighted here first.</p>
        <div class="lab-attention-list">{attention_html}</div>
      </div>

      <div class="soft-card magic-card ai-card-wide">
        <div class="card-title">AI Medical Explanation</div>
        <div class="ai-bubble lab-ai-bubble">
          <div class="mini-robot"></div>
          <p>{clean_ai}</p>
        </div>
      </div>
    </div>

    <div class="wide-section soft-card table-card-modern">
      <div class="section-title">All Detected Values</div>
      {lab_table_html}
    </div>

    <div class="wide-section">
      {build_report_suggestions(records)}
    </div>
  </div>
  """

def analyze_lab_dashboard_ui(uploaded_file, pdf_language="English"):
  pdf_language = "English" # English-only PDF export
  old_outputs = analyze_medical_report_ui(uploaded_file, pdf_language)

  (
    status,
    report_type_display,
    patient_info_md,
    raw_text,
    formatted_text,
    lab_table_html,
    radiology_sections_md,
    ai_explanation,
    summary_html,
    raw_json,
    pdf_file,
    json_file,
  ) = old_outputs

  raw_json_clean = safe_str(raw_json).strip()
  if (not raw_json_clean) or raw_json_clean == "{}" or "successfully" not in safe_str(status).lower():
    reset_report_context()
    dashboard_html = build_placeholder_card(
      "Analysis not completed",
      status or "Please upload a valid PDF or image report."
    )
    pdf_output = gr.update(value=None, visible=False)
  else:
    dashboard_html = build_lab_dashboard_html(report_type_display, ai_explanation, lab_table_html)
    pdf_output = gr.update(value=pdf_file, visible=True)

  return (
    status,
    dashboard_html,
    lab_table_html,
    report_type_display,
    patient_info_md,
    raw_text,
    formatted_text,
    radiology_sections_md,
    ai_explanation,
    summary_html,
    raw_json,
    pdf_output,
    json_file,
  )


def build_xray_personalized_suggestions(result):
  """Create simple educational suggestions from the X-ray result."""
  result = normalize_xray_result(result)
  status = result.get("status", "")
  body_part = result.get("body_part", "the imaged area")
  body_area = _plain_body_area(body_part)
  body_area_text = _article_for_body_area(body_area)
  findings = result.get("key_findings", []) or []
  symptom_phrase = _body_part_advice_phrase(body_part)

  if _is_chest_body_part(body_part):
    suggestions = [
      "Show this Chest X-ray and summary to a qualified doctor or radiologist.",
      "Do not rely on pain medicine alone for chest pain or breathing trouble; get urgent medical help if these symptoms are present.",
      "Follow the treatment advice from your clinician, especially if you have fever, cough, shortness of breath, or worsening chest symptoms.",
      "Keep the uploaded image and this simple summary for your appointment.",
    ]
  elif _is_orthopedic_body_part(body_part):
    suggestions = [
      f"Rest {body_area_text} and avoid heavy use, sports, lifting, or putting weight on it until a clinician reviews the X-ray.",
      "For pain or swelling, use a cold pack wrapped in cloth for 1520 minutes at a time, especially during the first 2448 hours.",
      f"If swelling is present, keep {body_area_text} raised when possible and avoid tight wrapping that causes numbness or more pain.",
      "For pain relief, you may consider paracetamol/acetaminophen, or ibuprofen/naproxen only if these are safe for you. Avoid anti-inflammatory painkillers if you have stomach ulcers, kidney disease, blood thinners, pregnancy concerns, or an allergyask a doctor or pharmacist.",
      f"Get urgent medical help if you have {symptom_phrase}.",
    ]
  else:
    suggestions = [
      "Ask a qualified doctor or radiologist to review the original X-ray.",
      "If you have pain, swelling, fever, numbness, weakness, or worsening symptoms, seek medical advice promptly.",
      "For pain relief, use only medicines that are safe for you and follow the label or your clinicians advice.",
      "Keep the uploaded image and this simple summary for your appointment.",
    ]

  if status == "Limited / unclear image":
    suggestions.insert(0, "The image review was limited or unclear, so upload a clearer X-ray if available or use the official radiology report.")
  elif status == "Needs attention" and not _is_chest_body_part(body_part):
    suggestions.insert(0, f"Because this review found something that needs checking, avoid stressing {body_area_text} until a clinician reviews it.")

  if findings:
    suggestions.append("Main points to discuss with the doctor: " + "; ".join([safe_str(x) for x in findings[:3]]) + ".")

  items = "".join(f"<li>{_ui_escape(item)}</li>" for item in suggestions)
  return f"""
  <div class="soft-card magic-card xray-suggestion-card">
    <div class="card-title">Personalized Suggestions</div>
    <ul>{items}</ul>
  </div>
  """


def expand_xray_simple_explanation(result):
  """Make the simple explanation useful and easy to understand."""
  result = normalize_xray_result(result)
  base = safe_str(result.get("simple_explanation", "")).strip()
  body_part = safe_str(result.get("body_part", "the imaged area"))
  body_area = _plain_body_area(body_part)
  view = safe_str(result.get("view", "the uploaded view"))
  status = safe_str(result.get("status", "Limited / unclear image"))
  symptom_phrase = _body_part_advice_phrase(body_part)

  if status == "Needs attention":
    extra = (
      f" In simple words, this {body_area} X-ray review noticed something that may need a doctor to check. "
      f"This is not a diagnosis. Please get medical advice, especially if you have {symptom_phrase}."
    )
  elif status == "No obvious acute abnormality":
    extra = (
      f" In simple words, this educational review did not find an obvious urgent-looking issue in the {body_area} X-ray ({view}). "
      "Still, an X-ray cannot explain every symptom, so see a healthcare professional if pain or other symptoms continue."
    )
  else:
    extra = (
      f" In simple words, the {body_area} X-ray review was limited or unclear. "
      "A clearer image, the official radiology report, or review by a qualified doctor may be needed."
    )

  if base and extra.strip() not in base:
    return _xray_clean_generated_text(base + extra, view=view, body_part=body_part, max_sentences=6)
  return _xray_clean_generated_text(base or extra.strip(), view=view, body_part=body_part, max_sentences=6)


def _xray_easy_patient_words(text):
  """Turn generated X-ray wording into simpler patient-facing wording."""
  s = _xray_clean_generated_text(text, max_sentences=2)
  replacements = [
    (r"\bclinician\b", "doctor"),
    (r"\borthopedic doctor\b", "bone/joint doctor"),
    (r"\borthopedic clinician\b", "bone/joint doctor"),
    (r"\bradiologist\b", "X-ray specialist"),
    (r"\bpatella\b", "kneecap"),
    (r"\bmetallic fixation hardware\b", "metal repair hardware"),
    (r"\bmetallic hardware\b", "metal hardware"),
    (r"\bfixation\b", "surgical repair/fixation"),
    (r"\bprojected near\b", "seen near"),
    (r"\bfracture-healing status\b", "how the bone is healing"),
    (r"\bbone-healing status\b", "how the bone is healing"),
    (r"\bNo obvious acute abnormality\b", "No obvious urgent problem"),
  ]
  for pattern, repl in replacements:
    s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
  s = re.sub(r"\s+", " ", s).strip(" .")
  return s


def _xray_easy_summary_bullets(result, max_items=5):
  """Universal easy bullet summary for any uploaded X-ray body part."""
  body_part = safe_str(result.get("body_part", "Unknown")).strip() or "Unknown"
  body_area = _plain_body_area(body_part)
  view = safe_str(result.get("view", "Unknown")).strip() or "Unknown"
  status = safe_str(result.get("status", "Limited / unclear image")).strip()
  layer1 = result.get("layer_1_safety", {}) or {}
  layer2 = result.get("layer_2_findings", {}) or {}
  possible = _xray_as_list(layer2.get("possible_findings", []), max_items=4)
  hardware_present = layer2.get("hardware_present")
  hardware_desc = safe_str(layer2.get("hardware_description", "")).strip()
  symptom_phrase = _body_part_advice_phrase(body_part)

  bullets = []
  if view and "unknown" not in view.lower() and "uncertain" not in view.lower():
    bullets.append(f"This is a {view} {body_area} X-ray.")
  else:
    bullets.append(f"This is a {body_area} X-ray; the exact view is not clear.")

  if hardware_present is True:
    if hardware_desc:
      bullets.append(_xray_easy_patient_words(hardware_desc) + ".")
    else:
      bullets.append("Metal hardware is visible in the X-ray area.")
  elif hardware_present is False and status == "No obvious acute abnormality":
    bullets.append("No obvious metal hardware was found by this basic AI review.")

  if possible:
    bullets.append("Possible finding: " + _xray_easy_patient_words(possible[0]) + ".")
  elif status == "No obvious acute abnormality":
    bullets.append("No obvious urgent problem was found by this basic AI review.")
  elif status == "Limited / unclear image":
    bullets.append("The image is limited or unclear, so this result may be less reliable.")
  elif hardware_present is not True:
    bullets.append("Something in the image needs a doctor or X-ray specialist to check.")

  if status == "Needs attention":
    bullets.append(f"Please show this X-ray to a doctor or X-ray specialist, especially if you have {symptom_phrase}.")
  elif status == "Limited / unclear image":
    bullets.append("Use the official radiology report or upload a clearer X-ray if available.")
  else:
    bullets.append("This is educational only; a doctor should confirm the result if symptoms continue.")

  reason = safe_str(layer1.get("urgency_reason", "")).strip()
  if reason and status == "Needs attention" and len(bullets) < max_items:
    easy_reason = _xray_easy_patient_words(reason)
    if easy_reason and not any(easy_reason.lower() in b.lower() for b in bullets):
      bullets.append(easy_reason + ".")

  cleaned, seen = [], set()
  for bullet in bullets:
    bullet = _xray_clean_generated_text(bullet, view=view, body_part=body_part, max_sentences=1).strip(" -")
    if not bullet:
      continue
    if not bullet.endswith(('.', '!', '?')):
      bullet += "."
    key = re.sub(r"[^a-z0-9]+", " ", bullet.lower()).strip()
    if key in seen:
      continue
    seen.add(key)
    cleaned.append(bullet)
    if len(cleaned) >= max_items:
      break
  return cleaned


def _xray_summary_bullets_html(result):
  bullets = _xray_easy_summary_bullets(result)
  if not bullets:
    return '<p class="saas-summary-text">The visual summary will appear here after analysis.</p>'
  items = "".join(f"<li>{_ui_escape(item)}</li>" for item in bullets)
  return f'<ul class="saas-summary-bullets" style="margin:8px 0 0 18px; padding:0; line-height:1.55;">{items}</ul>'


def _xray_summary_bullets_text(result):
  bullets = _xray_easy_summary_bullets(result)
  return "\n".join(f" {item}" for item in bullets)



def _xray_display_type_label(result):
  """Return a safe dashboard type label without creating a fake default.

  Example: Chest X-ray, Knee X-ray, or simply X-ray image when the body area
  is not confirmed. This prevents labels like "Unknown body area X-ray" and
  prevents a stale model exam_type from overriding a corrected body_part.
  """
  result = result if isinstance(result, dict) else {}
  exam_type = safe_str(result.get("exam_type", "")).strip()
  body_part = safe_str(result.get("body_part", "")).strip()
  body_family = _body_part_family_label(body_part)
  exam_family = _body_part_family_label(exam_type)

  if body_part and body_family != "unknown":
    if exam_type and "x-ray" in exam_type.lower() and exam_family == body_family:
      return exam_type
    return f"{body_part} X-ray"

  if exam_type and "unknown" not in exam_type.lower() and "unclear" not in exam_type.lower() and "x-ray" in exam_type.lower():
    return exam_type
  return "X-ray image"

def build_xray_dashboard_html(result):
  result = normalize_xray_result(result)
  status_cls, status_icon = xray_status_classes(result["status"])

  findings_html = "".join(
    f"""
    <div class="saas-finding-card pop-card">
      <div class="finding-alert-icon"></div>
      <p>{_ui_escape(item)}</p>
    </div>
    """
    for item in result.get("key_findings", [])
  )
  if not findings_html:
    findings_html = '<div class="friendly-note">No key findings were returned.</div>'

  if result["status"] == "No obvious acute abnormality":
    attention_label = "Stable Review"
    attention_class = "ok"
    attention_icon = ""
  elif result["status"] == "Needs attention":
    attention_label = "Need Review"
    attention_class = "warn"
    attention_icon = ""
  else:
    attention_label = "Limited Review"
    attention_class = "soft"
    attention_icon = "!"

  simple = _ui_escape(expand_xray_simple_explanation(result))
  impression = _ui_escape(result.get("overall_impression", ""))
  summary_bullets_html = _xray_summary_bullets_html(result)
  return f"""
  <div class="saas-dashboard xray-saas-dashboard fade-in-up">
    <div class="saas-result-header">
      <div>
        <div class="saas-kicker">Radiology dashboard educational review</div>
        <h2>X-ray Analysis Result</h2>
        <p>Your uploaded X-ray is translated into a simple, review-friendly summary.</p>
      </div>
      <div class="saas-status-pill {attention_class}">{attention_icon} {attention_label}</div>
    </div>

    <div class="saas-widget-grid three">
      <div class="saas-widget magic-card"><span></span><b>Image Uploaded</b><small>Preview available</small></div>
      <div class="saas-widget magic-card"><span></span><b>Report Ready</b><small>English PDF summary</small></div>
      <div class="saas-widget magic-card pulse-soft"><span>{attention_icon}</span><b>{_ui_escape(result['status'])}</b><small>Attention level</small></div>
    </div>

    <div class="saas-two-col">
      <div class="saas-card magic-card summary-card slide-up-card">
        <div class="saas-card-title"> X-ray Visual Summary <em class="jump-text">AI Summary Ready</em> <span class="medibuddy-type ai-type-label" data-texts="Need Review||Visual Insight Ready"></span></div>
        {summary_bullets_html}
        <div class="xray-status-badge {status_cls}">{status_icon} {_ui_escape(result['status'])}</div>
      </div>

      <div class="saas-card magic-card slide-up-card">
        <div class="saas-card-title"> General Report Category and Type</div>
        <div class="saas-info-row"><span>Category</span><b>Radiology</b></div>
        <div class="saas-info-row"><span>Type</span><b>{_ui_escape(_xray_display_type_label(result))}</b></div>
        <div class="saas-info-row"><span>View</span><b>{_ui_escape(result['view'])}</b></div>
      </div>
    </div>

    <div class="saas-two-col">
      <div class="saas-card magic-card slide-up-card">
        <div class="saas-card-title"> Key Findings <em class="jump-text">Need Review</em></div>
        <div class="saas-findings-grid">{findings_html}</div>
      </div>

      <div class="saas-card magic-card slide-up-card">
        <div class="saas-card-title"> Simple Explanation</div>
        <p class="simple-readable">{simple}</p>
      </div>
    </div>

    <div class="saas-alert {attention_class}">
      <div class="saas-alert-icon">{attention_icon}</div>
      <div><b>Need Attention</b><p>{impression or 'Please review this result with a qualified medical professional if symptoms are present.'}</p></div>
    </div>

    <div class="saas-card magic-card suggestions-card slide-up-card">
      {build_xray_personalized_suggestions(result)}
    </div>

    <div class="saas-disclaimer"> Educational use only. This is not a confirmed diagnosis. Please consult a qualified healthcare professional.</div>
  </div>
  """

def analyze_xray_dashboard_ui(uploaded_xray_file, pdf_language="English", selected_xray_region="Auto-detect"):
  pdf_language = "English" # English-only PDF export
  try:
    image_path = get_uploaded_path(uploaded_xray_file)
    if not image_path:
      reset_report_context()
      empty = fallback_xray_result()
      return (
        "Please upload a PNG or JPG X-ray image.",
        None,
        build_placeholder_card("No X-ray uploaded", "Upload an X-ray image to see the visual summary, key findings, and simple explanation."),
        build_xray_markdown(empty),
        gr.update(value=None, visible=False),
      )

    result = analyze_xray_image(image_path, selected_xray_region)
    result = normalize_xray_result(result)

    layer1 = result.get("layer_1_safety", {}) or {}
    layer2 = result.get("layer_2_findings", {}) or {}
    layer3 = result.get("layer_3_report", {}) or {}
    summary_bullets_text = _xray_summary_bullets_text(result)
    CURRENT_REPORT_CONTEXT["raw_text"] = ""
    CURRENT_REPORT_CONTEXT["formatted_text"] = (
      f"X-ray analysis\n"
      f"Exam type: {result['exam_type']}\n"
      f"Body part: {result['body_part']}\n"
      f"View: {result['view']}\n"
      f"Status: {result['status']}\n"
      f"Layer 1 safety: {json.dumps(layer1, ensure_ascii=False)}\n"
      f"Layer 2 findings: {json.dumps(layer2, ensure_ascii=False)}\n"
      f"Layer 3 safe report: {json.dumps(layer3, ensure_ascii=False)}\n"
      f"Overall impression: {result['overall_impression']}\n"
      f"Key findings: {', '.join(result['key_findings'])}\n"
      f"Simple explanation: {result['simple_explanation']}"
    )
    CURRENT_REPORT_CONTEXT["report_category"] = "Radiology"
    xray_report_type_display = _xray_display_type_label(result)
    CURRENT_REPORT_CONTEXT["report_subtype"] = xray_report_type_display
    CURRENT_REPORT_CONTEXT["patient_info"] = {}
    CURRENT_REPORT_CONTEXT["lab_records"] = []
    CURRENT_REPORT_CONTEXT["radiology_sections"] = {
      "Layer 1 Safety": f"Status: {layer1.get('status', result['status'])}; Image quality: {layer1.get('image_quality', 'Not specified')}; Doctor review required: {layer1.get('doctor_review_required', True)}; Reason: {layer1.get('urgency_reason', '')}",
      "Layer 2 Visible Anatomy": ", ".join(layer2.get("visible_anatomy", []) or []),
      "Layer 2 Hardware Check": layer2.get("hardware_description", ""),
      "Layer 2 Possible Findings": "; ".join(layer2.get("possible_findings", []) or []),
      "Layer 2 Uncertainty": "; ".join(layer2.get("uncertainty", []) or []),
      "Layer 3 Safe Summary": layer3.get("patient_friendly_summary", result["simple_explanation"]),
      "Overall Impression": result["overall_impression"],
      "Key Findings": result["key_findings"],
      "Simple Explanation": result["simple_explanation"],
    }
    CURRENT_REPORT_CONTEXT["ai_explanation"] = result["simple_explanation"]
    CURRENT_REPORT_CONTEXT["summary"] = summary_bullets_text or result["overall_impression"]

    xray_payload = {
      "original_filename": os.path.basename(image_path),
      "image_path": image_path,
      "report_category": "Radiology",
      "report_subtype": xray_report_type_display,
      "patient_info": {},
      "lab_records": [],
      "radiology_sections": {
        "Exam Type": result.get("exam_type", "X-ray image"),
        "Body Part": result.get("body_part", "Unknown"),
        "View": result.get("view", "Unknown"),
        "Status": result.get("status", "Limited / unclear image"),
        "Layer 1 Safety": f"Status: {layer1.get('status', result.get('status', ''))}; Image quality: {layer1.get('image_quality', 'Not specified')}; Doctor review required: {layer1.get('doctor_review_required', True)}; Reason: {layer1.get('urgency_reason', '')}",
        "Visible Anatomy": ", ".join(layer2.get("visible_anatomy", []) or []),
        "Hardware Check": layer2.get("hardware_description", ""),
        "Possible Findings": "; ".join(layer2.get("possible_findings", []) or []),
        "Uncertainty": "; ".join(layer2.get("uncertainty", []) or []),
        "Safe Patient Summary": layer3.get("patient_friendly_summary", result.get("simple_explanation", "")),
        "Doctor Questions": "; ".join(layer3.get("what_to_ask_doctor", []) or []),
        "Key Findings": "; ".join(result.get("key_findings", []) or []),
        "Overall Impression": result.get("overall_impression", ""),
      },
      "ai_explanation": expand_xray_simple_explanation(result),
      "summary": summary_bullets_text or result.get("overall_impression", ""),
    }
    xray_pdf_file = save_pdf_report(xray_payload, pdf_language)

    return (
      "X-ray analyzed successfully. Review the educational summary below.",
      image_path,
      build_xray_dashboard_html(result),
      build_xray_markdown(result),
      gr.update(value=xray_pdf_file, visible=bool(xray_pdf_file)),
    )
  except Exception as e:
    empty = normalize_xray_result({
      "overall_impression": f"X-ray UI error: {str(e)}",
      "key_findings": ["The system could not render the X-ray summary."],
      "simple_explanation": "Please try uploading the image again.",
      "caution": "This is educational only and not a confirmed diagnosis."
    })
    return (
      f"Error during X-ray analysis: {e}",
      gr.update(value=image_path) if 'image_path' in locals() and image_path else None,
      build_xray_dashboard_html(empty),
      build_xray_markdown(empty),
      gr.update(value=None, visible=False),
    )

def fill_question(text):
  return text

def render_step_progress(active_step, page_label=""):
  """Clean green step tracker:  2 3 with labels below."""
  steps = [
    (1, "WELCOME"),
    (2, "CHOOSE"),
    (3, "ANALYZE"),
  ]
  items = []
  for number, title in steps:
    cls = "active" if number == active_step else "done" if number < active_step else "upcoming"
    icon = "" if number < active_step else str(number)
    items.append(
      f'<div class="progress-item {cls}">'
      f' <div class="progress-dot">{icon}</div>'
      f' <div class="progress-label">{title}</div>'
      f'</div>'
    )
  page_text = f'<div class="progress-page-label">{html.escape(page_label)}</div>' if page_label else ""
  return f"""
  <div class="progress-stage" aria-label="Workflow progress">
    <div class="progress-rail"></div>
    <div class="progress-items">{''.join(items)}</div>
    {page_text}
  </div>
  """

CUSTOM_CSS = r"""

/* Green upload/file polish + remove option */
.upload-card .wrap, .upload-card .block, .upload-card .file-preview {
  border-color: rgba(31, 107, 87, .18) !important;
}
.upload-card [data-testid="file"] button,
.upload-card .file button,
.upload-card button[aria-label*="Remove"],
.upload-card button[title*="Remove"] {
  color: #1f6b57 !important;
  border-color: rgba(31, 107, 87, .22) !important;
}
.remove-file-btn, #lab-clear-file-btn, #xray-clear-file-btn {
  background: #ffffff !important;
  color: #1f6b57 !important;
  border: 1px solid rgba(31, 107, 87, .22) !important;
  box-shadow: 0 8px 20px rgba(31, 107, 87, .08) !important;
}
#lab-clear-file-btn:hover, #xray-clear-file-btn:hover {
  background: rgba(31, 107, 87, .08) !important;
}
#pdf-download, #xray-pdf-download {
  border-radius: 14px !important;
}
.visible-polish-summary {
  margin: 18px 0 16px 0 !important;
}
.visible-polish-summary .summary-card {
  border-radius: 24px !important;
  border: 1px solid rgba(31, 107, 87, .18) !important;
  background: linear-gradient(135deg, #ffffff, #f3fff8) !important;
  box-shadow: 0 10px 28px rgba(31, 107, 87, .08) !important;
  padding: 20px !important;
}
.visible-polish-summary .summary-title {
  font-size: 20px !important;
  font-weight: 900 !important;
  color: #143f35 !important;
}
.visible-polish-summary .summary-meta {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 10px !important;
  margin: 14px 0 !important;
}
.visible-polish-summary .meta-pill {
  background: #eafff1 !important;
  border: 1px solid #bcebd0 !important;
  border-radius: 999px !important;
  padding: 8px 12px !important;
  font-weight: 800 !important;
  color: #14543f !important;
}
.visible-polish-summary .summary-item {
  margin-top: 12px !important;
  padding: 12px 14px !important;
  border-radius: 16px !important;
  background: #ffffff !important;
  border: 1px solid #dcf3e6 !important;
}
.visible-polish-summary .summary-footnote {
  margin-top: 14px !important;
  background: #fff7ed !important;
  border: 1px solid #fed7aa !important;
  color: #7c2d12 !important;
  padding: 12px 14px !important;
  border-radius: 14px !important;
  font-weight: 800 !important;
}

@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
  font-family: 'Nunito', sans-serif !important;
  background:
   radial-gradient(circle at 8% 8%, rgba(47, 143, 111, .13), transparent 26%),
   radial-gradient(circle at 90% 10%, rgba(31, 107, 87, .12), transparent 24%),
   radial-gradient(circle at 50% 100%, rgba(20, 184, 166, .09), transparent 30%),
   #f8fbff !important;
  color: #17203a !important;
}

.gradio-container {
  max-width: 1360px !important;
  margin: 0 auto !important;
  padding: 18px !important;
}

.app-shell {
  background: rgba(255,255,255,.72);
  border: 1px solid rgba(31, 107, 87, .16);
  border-radius: 28px;
  box-shadow: 0 24px 80px rgba(31, 107, 87, .12);
  backdrop-filter: blur(18px);
  overflow: hidden;
  padding: 18px;
}

.top-nav {
  align-items: center;
  justify-content: space-between;
  padding: 6px 4px 16px;
}

.logo-wrap {
  display: flex;
  align-items: center;
  gap: 10px;
}

.logo-icon {
  width: 42px;
  height: 42px;
  display: grid;
  place-items: center;
  border-radius: 14px;
  background: linear-gradient(135deg, #1f6b57, #2f8f6f);
  color: white;
  font-size: 24px;
  box-shadow: 0 12px 28px rgba(31, 107, 87, .25);
}

.logo-text {
  font-size: 22px;
  font-weight: 900;
  letter-spacing: -.04em;
  color: #17203a;
}

.logo-text span { color: #1f6b57; }

.nav-pills {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.nav-pills span {
  padding: 9px 13px;
  border-radius: 999px;
  background: rgba(255,255,255,.76);
  border: 1px solid rgba(31, 107, 87, .12);
  color: #56617c;
  font-weight: 800;
  font-size: 12px;
}

.gradio-container .tabs {
  border: none !important;
}

.gradio-container button {
  border-radius: 15px !important;
  font-weight: 900 !important;
}

button.primary, #lab-analyze-btn, #xray-analyze-btn, #choice-lab-btn, #choice-xray-btn, #ask-btn {
  background: linear-gradient(135deg, #2f8f6f, #2f8f6f) !important;
  border: none !important;
  color: white !important;
  box-shadow: 0 12px 30px rgba(31, 107, 87, .22) !important;
}

.hero-grid {
  display: grid;
  grid-template-columns: 1.05fr .95fr;
  gap: 30px;
  align-items: center;
  padding: 28px 12px 10px;
}

.hero-copy {
  padding: 22px 12px;
}

.micro-pill {
  width: fit-content;
  padding: 8px 13px;
  border-radius: 999px;
  background: rgba(31, 107, 87, .10);
  color: #1f6b57;
  font-weight: 900;
  font-size: 13px;
  margin-bottom: 18px;
}

.hero-copy h1 {
  margin: 0;
  font-size: clamp(34px, 5vw, 58px);
  line-height: .98;
  letter-spacing: -.06em;
  color: #111a34;
}

.hero-copy h1 span {
  color: #1f6b57;
}

.hero-copy p {
  max-width: 560px;
  margin: 18px 0 22px;
  color: #58627d;
  font-size: 18px;
  line-height: 1.55;
  font-weight: 700;
}

.feature-list {
  display: grid;
  gap: 13px;
  margin: 24px 0;
}

.feature-item {
  display: grid;
  grid-template-columns: 42px 1fr;
  gap: 12px;
  align-items: center;
}

.feature-icon {
  width: 42px;
  height: 42px;
  display: grid;
  place-items: center;
  border-radius: 14px;
  background: rgba(31, 107, 87, .10);
  color: #1f6b57;
  font-size: 20px;
}

.feature-item b {
  display: block;
  color: #17203a;
  font-size: 15px;
}

.feature-item span {
  color: #6a748d;
  font-size: 13px;
  font-weight: 700;
}

.hero-cta {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  align-items: center;
  margin-top: 12px;
}

.cta-main {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 14px 22px;
  border-radius: 16px;
  background: linear-gradient(135deg, #2f8f6f, #2f8f6f);
  color: white !important;
  font-weight: 900;
  text-decoration: none !important;
  box-shadow: 0 14px 35px rgba(31, 107, 87, .24);
}

.secure-line {
  color: #6a748d;
  font-weight: 900;
  font-size: 12px;
}

.hero-art {
  position: relative;
  min-height: 510px;
  border-radius: 30px;
  background:
   radial-gradient(circle at 50% 45%, rgba(31, 107, 87,.11), transparent 38%),
   rgba(255,255,255,.58);
  border: 1px solid rgba(31, 107, 87,.11);
  overflow: hidden;
}

.person-card {
  position: absolute;
  left: 50%;
  top: 48%;
  transform: translate(-50%, -50%);
  width: min(360px, 86%);
  min-height: 390px;
  border-radius: 34px;
  background: linear-gradient(180deg, #ffffff, #edf7f2);
  box-shadow: 0 32px 70px rgba(82, 65, 160, .16);
  border: 1px solid rgba(31, 107, 87,.12);
  display: grid;
  place-items: center;
  text-align: center;
  padding: 30px;
}

.person-face {
  width: 142px;
  height: 142px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, #fed7aa, #fdba74);
  font-size: 72px;
  margin-bottom: 18px;
  box-shadow: inset 0 -8px 0 rgba(0,0,0,.04);
}

.report-paper {
  width: 190px;
  background: white;
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 18px 40px rgba(23, 32, 58, .12);
}

.paper-line {
  height: 9px;
  border-radius: 999px;
  background: #dcefe8;
  margin: 8px 0;
}

.paper-line.short { width: 62%; background: #d7eee5; }

.float-bubble {
  position: absolute;
  background: white;
  border: 1px solid rgba(31, 107, 87, .13);
  border-radius: 18px;
  padding: 12px 15px;
  font-weight: 900;
  color: #17203a;
  box-shadow: 0 18px 35px rgba(31, 107, 87,.10);
}

.bubble-one { top: 115px; right: 36px; }
.bubble-two { top: 230px; left: 26px; }
.bubble-three { bottom: 92px; right: 46px; }

.mascots {
  display: flex;
  gap: 18px;
  align-items: center;
  justify-content: center;
  margin-top: 8px;
}

.mascot {
  width: 76px;
  height: 76px;
  border-radius: 24px;
  display: grid;
  place-items: center;
  font-size: 42px;
  background: white;
  box-shadow: 0 14px 35px rgba(31, 107, 87,.10);
  border: 1px solid rgba(31, 107, 87,.10);
}

.section-heading {
  text-align: center;
  padding: 24px 12px 16px;
}

.section-heading h2 {
  margin: 0;
  color: #17203a;
  font-size: clamp(26px, 3vw, 38px);
  letter-spacing: -.04em;
}

.section-heading p {
  margin: 8px 0 0;
  color: #6a748d;
  font-weight: 800;
}

.choice-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 24px;
  max-width: 860px;
  margin: 0 auto 22px;
}

.choice-card {
  background: rgba(255,255,255,.82);
  border: 1px solid rgba(31, 107, 87,.11);
  border-radius: 26px;
  padding: 24px;
  box-shadow: 0 18px 50px rgba(31, 107, 87,.10);
  text-align: center;
}

.choice-illustration {
  width: 150px;
  height: 150px;
  display: grid;
  place-items: center;
  margin: 0 auto 16px;
  border-radius: 28px;
  background: linear-gradient(180deg, #e8f4ef, #f7fbf8);
  font-size: 76px;
}

.choice-card h3 {
  font-size: 24px;
  margin: 8px 0 8px;
  color: #17203a;
  letter-spacing: -.03em;
}

.choice-card p {
  color: #6a748d;
  font-weight: 700;
  min-height: 48px;
}

.how-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  max-width: 900px;
  margin: 26px auto 8px;
  padding: 18px;
  border-radius: 26px;
  background: rgba(255,255,255,.55);
  border: 1px solid rgba(31, 107, 87,.10);
}

.how-step {
  text-align: center;
  padding: 12px;
}

.how-icon {
  width: 54px;
  height: 54px;
  display: grid;
  place-items: center;
  margin: 0 auto 10px;
  border-radius: 18px;
  background: rgba(31, 107, 87,.10);
  font-size: 26px;
}

.how-step b { display: block; color: #17203a; }
.how-step span { color: #6a748d; font-size: 12px; font-weight: 700; }

.upload-card {
  padding: 16px;
  border-radius: 24px;
  background: rgba(255,255,255,.78);
  border: 1px solid rgba(31, 107, 87,.12);
  box-shadow: 0 14px 40px rgba(31, 107, 87,.08);
}

.upload-title {
  font-size: 15px;
  font-weight: 900;
  color: #17203a;
  margin: 2px 0 10px;
}

.status-box textarea {
  border-radius: 16px !important;
  font-weight: 800 !important;
  border: 1px solid rgba(31, 107, 87,.14) !important;
  background: rgba(255,255,255,.78) !important;
}

.result-page {
  padding: 8px 2px 16px;
}

.page-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 18px;
}

.page-head h2 {
  margin: 8px 0 0;
  font-size: 28px;
  line-height: 1;
  color: #17203a;
  letter-spacing: -.04em;
}

.fake-back {
  pointer-events: none;
  padding: 8px 14px;
  border-radius: 999px !important;
  color: #53607a !important;
  background: #fff !important;
  border: 1px solid rgba(31, 107, 87,.12) !important;
  box-shadow: none !important;
}

.complete-chip {
  border-radius: 999px;
  background: #dcfce7;
  color: #15803d;
  font-weight: 900;
  padding: 10px 14px;
  font-size: 13px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 16px;
}

.stat-card {
  border-radius: 20px;
  padding: 18px 16px;
  text-align: center;
  font-weight: 900;
  box-shadow: 0 12px 36px rgba(31, 107, 87,.08);
  border: 1px solid rgba(255,255,255,.7);
}

.stat-card span {
  display: block;
  font-size: 13px;
  margin-bottom: 7px;
}

.stat-card b {
  display: block;
  font-size: 32px;
  line-height: 1;
}

.stat-card.blue { background: #eaf5ef; color: #1f6b57; }
.stat-card.green { background: #eafaf0; color: #16a34a; }
.stat-card.orange { background: #fff4e5; color: #f97316; }
.stat-card.red { background: #ffecec; color: #ef4444; }

.lab-layout,
.xray-layout-new {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.left-stack,
.right-stack {
  display: grid;
  gap: 16px;
  align-content: start;
}

.soft-card {
  background: rgba(255,255,255,.86);
  border: 1px solid rgba(31, 107, 87,.10);
  border-radius: 22px;
  padding: 18px;
  box-shadow: 0 15px 45px rgba(31, 107, 87,.08);
}

.card-title {
  color: #17203a;
  font-weight: 900;
  font-size: 15px;
  margin-bottom: 13px;
}

.info-row {
  display: grid;
  grid-template-columns: 120px 1fr;
  gap: 10px;
  padding: 9px 0;
  border-top: 1px solid rgba(31, 107, 87,.08);
}

.info-row:first-of-type {
  border-top: none;
}

.info-row span {
  color: #6a748d;
  font-weight: 800;
  font-size: 13px;
}

.info-row b {
  color: #17203a;
  font-size: 13px;
}

.ai-bubble {
  position: relative;
  display: grid;
  grid-template-columns: 58px 1fr;
  gap: 12px;
  align-items: start;
  background: #f7fbf8;
  border-radius: 18px;
  padding: 15px;
}

.mini-robot {
  width: 48px;
  height: 48px;
  display: grid;
  place-items: center;
  border-radius: 17px;
  background: white;
  box-shadow: 0 10px 25px rgba(31, 107, 87,.10);
  font-size: 28px;
}

.ai-bubble p {
  margin: 0;
  color: #45516d;
  font-weight: 750;
  line-height: 1.55;
  font-size: 14px;
}

.health-card {
  text-align: center;
}

.gauge-wrap {
  display: grid;
  place-items: center;
  margin: 4px 0 8px;
}

.gauge {
  width: 180px;
  height: 96px;
  border-radius: 180px 180px 0 0;
  position: relative;
  overflow: hidden;
  background:
   conic-gradient(from 270deg at 50% 100%, #ef4444 0deg 45deg, #f59e0b 45deg 95deg, #22c55e 95deg 180deg, #e5e7eb 180deg 360deg);
}

.gauge::after {
  content: "";
  position: absolute;
  left: 22px;
  right: 22px;
  bottom: -58px;
  height: 116px;
  border-radius: 999px;
  background: white;
}

.gauge-face {
  position: absolute;
  z-index: 2;
  left: 50%;
  bottom: 9px;
  transform: translateX(-50%);
  width: 68px;
  height: 68px;
  border-radius: 50%;
  background: #fef3c7;
  display: grid;
  place-items: center;
  font-size: 34px;
  box-shadow: 0 10px 30px rgba(31, 107, 87,.10);
}

.health-card h3 {
  margin: 6px 0 2px;
  color: #17203a;
  font-size: 20px;
}

.health-card p {
  margin: 0;
  color: #6a748d;
  font-weight: 750;
  font-size: 13px;
}

.friendly-note {
  color: #6a748d;
  font-weight: 750;
  line-height: 1.55;
}

.normal-box {
  background: #ecfdf5;
  color: #15803d;
  border-radius: 16px;
  padding: 14px;
  font-weight: 900;
}

.ab-list {
  display: grid;
  gap: 10px;
}

.ab-row {
  display: grid;
  grid-template-columns: 1fr auto auto;
  align-items: center;
  gap: 10px;
  padding: 12px;
  border-radius: 16px;
  background: #f8fbff;
}

.ab-row span {
  font-weight: 900;
  color: #17203a;
}

.ab-row b {
  color: #52607c;
  font-size: 13px;
}

.pill {
  font-style: normal;
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 12px;
  font-weight: 900;
}

.pill-danger { background: #fee2e2; color: #b91c1c; }
.pill-warning { background: #fef3c7; color: #b45309; }
.pill-success { background: #dcfce7; color: #166534; }
.pill-muted { background: #e2e8f0; color: #475569; }

.wide-section {
  margin-top: 16px;
}

.section-title {
  font-size: 18px;
  font-weight: 900;
  color: #17203a;
  margin: 20px 0 10px;
}

.empty-result-card {
  min-height: 360px;
  border-radius: 28px;
  background: rgba(255,255,255,.78);
  border: 1px dashed rgba(31, 107, 87,.25);
  display: grid;
  place-items: center;
  text-align: center;
  padding: 36px;
  box-shadow: 0 18px 50px rgba(31, 107, 87,.08);
}

.empty-result-card h3 {
  margin: 8px 0 4px;
  color: #17203a;
  font-size: 24px;
}

.empty-result-card p {
  margin: 0;
  color: #6a748d;
  font-weight: 800;
}

.empty-emoji {
  font-size: 64px;
}

.finding-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.finding-tile {
  display: grid;
  grid-template-columns: 38px 1fr;
  gap: 10px;
  align-items: center;
  background: #f8fbff;
  border-radius: 17px;
  padding: 13px;
}

.finding-tile span {
  width: 38px;
  height: 38px;
  display: grid;
  place-items: center;
  border-radius: 13px;
  background: white;
  box-shadow: 0 8px 20px rgba(31, 107, 87,.08);
}

.finding-tile p {
  margin: 0;
  color: #45516d;
  font-weight: 800;
  font-size: 13px;
  line-height: 1.45;
}

.simple-box {
  background: #f7fbf8;
  border-radius: 18px;
  padding: 15px;
  color: #45516d;
  font-weight: 800;
  line-height: 1.55;
}

.impression-card {
  text-align: center;
}

.impression-face {
  width: 86px;
  height: 86px;
  margin: 0 auto 10px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  font-size: 44px;
  background: #fef3c7;
}

.impression-card h3 {
  margin: 0 0 4px;
  color: #17203a;
}

.impression-card p {
  margin: 0;
  color: #6a748d;
  font-weight: 800;
  line-height: 1.5;
}

.disclaimer-card {
  margin-top: 16px;
  background: #fff7ed;
  border: 1px solid #fed7aa;
  color: #9a3412;
  border-radius: 20px;
  padding: 15px 18px;
  font-weight: 800;
  line-height: 1.5;
}

.chat-layout {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 16px;
}

.chat-history-card {
  background: rgba(255,255,255,.84);
  border: 1px solid rgba(31, 107, 87,.10);
  border-radius: 24px;
  padding: 18px;
  box-shadow: 0 15px 45px rgba(31, 107, 87,.08);
}

.chat-history-card h3 {
  margin: 0 0 14px;
  color: #17203a;
  font-size: 18px;
}

.quick-q {
  display: block;
  width: 100%;
  text-align: left;
  margin-bottom: 10px;
  border-radius: 16px;
  padding: 13px 14px;
  background: #edf7f2;
  color: #1f6b57;
  font-weight: 900;
}

.chat-main {
  background: rgba(255,255,255,.78);
  border: 1px solid rgba(31, 107, 87,.10);
  border-radius: 24px;
  padding: 16px;
  box-shadow: 0 15px 45px rgba(31, 107, 87,.08);
}

.chatbot-box {
  border-radius: 20px !important;
  overflow: hidden !important;
}

.chat-input-row {
  align-items: center;
  gap: 12px;
}


.chart-wrap {
  margin-top: 18px;
  background: #fff;
  border: 1px solid rgba(31, 107, 87,.10);
  border-radius: 20px;
  padding: 16px;
}
.status-chart { display: flex; flex-direction: column; gap: 12px; margin-top: 12px; }
.chart-row { display: grid; grid-template-columns: 80px 1fr 32px; align-items: center; gap: 10px; font-weight: 900; color: #475569; }
.chart-track { height: 14px; background: #e8f4ef; border-radius: 999px; overflow: hidden; }
.chart-fill { height: 100%; border-radius: 999px; }
.chart-normal { background: #22c55e; }
.chart-low { background: #f59e0b; }
.chart-high { background: #ef4444; }
.suggestion-card ul { margin: 10px 0 0; padding-left: 22px; color: #334155; font-weight: 800; line-height: 1.7; }
.suggestion-card li { margin-bottom: 8px; }


/* Next-level assistant panel at the end of analysis page */
.assistant-dock {
  margin-top: 26px;
  background: linear-gradient(135deg, rgba(255,255,255,.96), rgba(244,247,255,.94));
  border: 1px solid rgba(31, 107, 87,.14);
  border-radius: 30px;
  padding: 18px;
  box-shadow: 0 24px 70px rgba(31, 107, 87,.13);
}
.assistant-hero {
  display: grid;
  grid-template-columns: 70px 1fr;
  gap: 14px;
  align-items: center;
  margin-bottom: 14px;
}
.assistant-avatar {
  width: 64px;
  height: 64px;
  display: grid;
  place-items: center;
  border-radius: 22px;
  font-size: 34px;
  background: radial-gradient(circle at 35% 25%, #ffffff, #dcefe8 35%, #2f8f6f 100%);
  box-shadow: 0 16px 35px rgba(31, 107, 87,.25);
}
.assistant-hero h2 {
  margin: 0;
  font-size: 26px;
  color: #17203a;
}
.assistant-hero p {
  margin: 4px 0 0;
  color: #64748b;
  font-weight: 800;
  line-height: 1.45;
}
.assistant-prompt-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 14px;
}
.assistant-chip {
  border-radius: 999px;
  background: #edf7f2;
  color: #1f6b57;
  padding: 9px 13px;
  font-weight: 900;
  font-size: 13px;
}
.bottom-action-row {
  margin-top: 16px;
  display: grid !important;
  grid-template-columns: 1fr;
  gap: 12px;
}
#lab-bottom-back-btn, #xray-bottom-back-btn {
  background: linear-gradient(135deg, #e8f4ef, #ffffff) !important;
  color: #1f6b57 !important;
  border: 1px solid rgba(31, 107, 87,.22) !important;
  border-radius: 16px !important;
  font-weight: 900 !important;
  box-shadow: 0 12px 30px rgba(31, 107, 87,.08) !important;
}
#lab-bottom-back-btn:hover, #xray-bottom-back-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 16px 35px rgba(31, 107, 87,.14) !important;
}

.app-footer {
  text-align: center;
  padding: 18px 0 4px;
  color: #7a849c;
  font-weight: 900;
  font-size: 12px;
}

#pdf-download {
  border-radius: 18px !important;
}

/* Existing helper HTML styling, refreshed to match new UI */
.result-card,
.summary-card,
.xray-card {
  background: rgba(255,255,255,.86);
  border: 1px solid rgba(31, 107, 87,.10);
  border-radius: 22px;
  padding: 18px;
  box-shadow: 0 15px 45px rgba(31, 107, 87,.08);
  margin-top: 0;
}

.result-card-head,
.summary-head,
.xray-head {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  margin-bottom:12px;
}

.result-card-head h3,
.summary-title,
.xray-title {
  margin:0;
  font-size:18px;
  color:#17203a;
  font-weight:900;
}

.mini-note,
.xray-subtitle,
.xray-note {
  font-size:12px;
  color:#6a748d;
  font-weight:800;
}

.metric-table {
  display:flex;
  flex-direction:column;
  gap:0;
  border:1px solid rgba(31, 107, 87,.10);
  border-radius:18px;
  overflow:hidden;
}

.metric-row {
  display:grid;
  grid-template-columns:1.4fr 1fr 1fr .8fr 1.4fr;
  align-items:center;
  background:#fff;
  border-bottom:1px solid #eef2f7;
}

.metric-row:last-child { border-bottom:none; }
.metric-header { background:#f6fbf8; font-size:12px; font-weight:900; color:#475569; }
.metric-cell { padding:12px 14px; font-size:14px; color:#17203a; }
.metric-param { font-weight:900; }
.metric-value, .metric-range, .metric-status { font-size:13px; }

.status-pill {
  display:inline-flex;
  align-items:center;
  border-radius:999px;
  padding:5px 10px;
  font-size:12px;
  font-weight:900;
}

.status-high { background:#fee2e2; color:#b91c1c; }
.status-low { background:#fef3c7; color:#b45309; }
.status-normal { background:#dcfce7; color:#166534; }
.status-unknown { background:#e2e8f0; color:#475569; }

.visual-wrap { display:flex; align-items:center; gap:10px; }
.visual-track { flex:1; height:8px; background:#e5e7eb; border-radius:999px; overflow:hidden; }
.visual-fill { height:100%; border-radius:999px; }
.bar-high { background:#ef4444; }
.bar-low { background:#f59e0b; }
.bar-normal { background:#22c55e; }
.bar-unknown { background:#94a3b8; }
.visual-pct { min-width:36px; text-align:right; font-size:12px; font-weight:900; color:#64748b; }

.glance-title { font-size:13px; color:#64748b; margin:14px 0 10px 2px; font-weight:900; }
.glance-grid { display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:10px; }
.glance-card { background:#fff; border-radius:16px; border:1px solid rgba(31, 107, 87,.10); padding:10px 10px 12px; }
.glance-bar { height:56px; border-radius:12px; margin-bottom:8px; opacity:.95; }
.glance-label { text-align:center; font-size:12px; font-weight:900; color:#475569; }
.mini-high .glance-bar { background:#ef4444; }
.mini-low .glance-bar { background:#f59e0b; }
.mini-normal .glance-bar { background:#22c55e; }
.mini-unknown .glance-bar { background:#94a3b8; }
.empty-state { padding:18px; border-radius:16px; background:#f8fafc; color:#64748b; font-size:14px; font-weight:800; }

.xray-status-badge {
  display:inline-flex;
  align-items:center;
  gap:7px;
  width: fit-content;
  border-radius:999px;
  padding:9px 13px;
  font-size:12px;
  font-weight:900;
}

.xray-green { background:#dcfce7; color:#166534; }
.xray-red { background:#fee2e2; color:#b91c1c; }
.xray-amber { background:#fef3c7; color:#b45309; }



/* ---------- Hide Gradio Page Tabs: button-only navigation flow ---------- */
/* Keeps the Gradio tab system working internally, but removes the visible top page navigation. */
.gradio-container [role="tablist"],
.gradio-container .tabs > .tab-nav,
.gradio-container .tab-nav,
.gradio-container .tabs button[role="tab"],
.gradio-container button[role="tab"] {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
}

.gradio-container .tabs,
.gradio-container .tabitem,
.gradio-container [role="tabpanel"] {
  border-top: 0 !important;
  margin-top: 0 !important;
}

/* ---------- Updated Welcome Flow + Get Started Button ---------- */
.gradio-container {
  scroll-behavior: smooth;
}

.hero-cta {
  margin-top: 30px;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 12px;
}

#get-started-btn {
  max-width: 330px !important;
  min-height: 64px !important;
  border: 0 !important;
  border-radius: 999px !important;
  padding: 16px 30px !important;
  background: linear-gradient(135deg, #1f6b57 0%, #1f6b57 52%, #06b6d4 100%) !important;
  color: #ffffff !important;
  font-size: 18px !important;
  font-weight: 950 !important;
  letter-spacing: .2px !important;
  box-shadow: 0 18px 35px rgba(31, 107, 87, .35), 0 7px 18px rgba(31, 107, 87, .22) !important;
  transform: translateY(0);
  transition: transform .2s ease, box-shadow .2s ease, filter .2s ease !important;
  position: relative;
  overflow: hidden;
}

#get-started-btn:hover {
  transform: translateY(-3px) scale(1.015);
  filter: brightness(1.05);
  box-shadow: 0 24px 48px rgba(31, 107, 87, .43), 0 12px 24px rgba(31, 107, 87, .25) !important;
}

#get-started-btn:active {
  transform: translateY(0) scale(.99);
}

#get-started-btn::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(120deg, transparent 0%, rgba(255,255,255,.28) 45%, transparent 70%);
  transform: translateX(-110%);
}

@keyframes button-shimmer {
  0% { transform: translateX(-110%); }
  55% { transform: translateX(120%); }
  100% { transform: translateX(120%); }
}

.welcome-trust-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 18px;
}

.trust-chip {
  border: 1px solid rgba(31, 107, 87,.12);
  background: rgba(255,255,255,.74);
  color: #475569;
  border-radius: 999px;
  padding: 9px 12px;
  font-size: 12px;
  font-weight: 850;
  box-shadow: 0 10px 24px rgba(31, 107, 87,.07);
}

.next-step-banner {
  margin: 18px 0 22px;
  border: 1px solid rgba(31, 107, 87,.12);
  background: linear-gradient(135deg, rgba(31, 107, 87,.10), rgba(6,182,212,.10));
  border-radius: 22px;
  padding: 16px 18px;
  color: #334155;
  font-weight: 800;
}

.next-step-banner b {
  color: #17203a;
}

.choice-card {
  transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
}

.choice-card:hover {
  transform: translateY(-5px);
  border-color: rgba(31, 107, 87,.24);
  box-shadow: 0 22px 50px rgba(31, 107, 87,.14);
}


/* ---------- Choose-only Page 2 Flow ---------- */
.page-action-row {
  margin: 8px 0 10px !important;
  justify-content: flex-start !important;
}
#back-to-welcome-btn {
  max-width: 120px !important;
  min-height: 42px !important;
  border-radius: 999px !important;
  background: rgba(255,255,255,.86) !important;
  border: 1px solid rgba(31, 107, 87, .20) !important;
  color: #27304f !important;
  box-shadow: 0 10px 22px rgba(31, 41, 55, .08) !important;
}
.choose-only-grid {
  align-items: stretch !important;
}
.choose-only-card {
  cursor: pointer;
  min-height: 360px;
  transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
}
.choose-only-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 22px 48px rgba(31, 107, 87, .18);
  border-color: rgba(31, 107, 87, .28);
}
#choice-lab-btn, #choice-xray-btn {
  min-height: 50px !important;
  font-size: 15px !important;
  margin-top: auto !important;
}



/* ---------- Breadcrumb / Progress Tracker ---------- */
.progress-wrap {
  position: relative;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  align-items: stretch;
  background: rgba(255,255,255,.72);
  border: 1px solid rgba(31, 107, 87,.12);
  border-radius: 22px;
  padding: 16px;
  margin: 12px 0 22px;
  box-shadow: 0 12px 34px rgba(31, 107, 87,.07);
  overflow: hidden;
}
.progress-line {
  position: absolute;
  left: 11%;
  right: 11%;
  top: 38px;
  height: 4px;
  background: linear-gradient(90deg, rgba(47, 143, 111,.32), rgba(31, 107, 87,.18));
  border-radius: 999px;
  z-index: 0;
}
.progress-step {
  position: relative;
  z-index: 1;
  display: flex;
  gap: 12px;
  align-items: center;
  border-radius: 18px;
  padding: 12px;
  background: rgba(248,250,252,.86);
  border: 1px solid rgba(31, 107, 87,.08);
}
.progress-step.active {
  background: linear-gradient(135deg, rgba(47, 143, 111,.13), rgba(31, 107, 87,.11));
  border-color: rgba(31, 107, 87,.26);
  box-shadow: 0 12px 30px rgba(31, 107, 87,.12);
}
.progress-step.done {
  background: rgba(236,253,245,.88);
  border-color: rgba(34,197,94,.20);
}
.progress-dot {
  width: 38px;
  height: 38px;
  display: grid;
  place-items: center;
  border-radius: 14px;
  color: white;
  font-weight: 900;
  background: linear-gradient(135deg, #2f8f6f, #2f8f6f);
  box-shadow: 0 10px 24px rgba(31, 107, 87,.18);
}
.progress-step.done .progress-dot { background: linear-gradient(135deg, #22c55e, #14b8a6); }
.progress-step.upcoming .progress-dot { background: #cbd5e1; color:#334155; }
.progress-copy b { display:block; color:#17203a; font-size:13px; }
.progress-copy span { display:block; color:#64748b; font-size:11px; font-weight:800; margin-top:2px; }
.progress-page-label {
  grid-column: 1 / -1;
  text-align: center;
  color: #1f6b57;
  font-weight: 900;
  font-size: 12px;
  letter-spacing: .02em;
}

/* ---------- Smooth page loading overlay ---------- */
#page-loader {
  position: fixed;
  inset: 0;
  z-index: 999999;
  display: none;
  place-items: center;
  background: rgba(248,251,255,.72);
  backdrop-filter: blur(10px);
}
#page-loader.is-visible { display: grid; }
.loader-card {
  width: min(340px, 88vw);
  background: white;
  border: 1px solid rgba(31, 107, 87,.16);
  border-radius: 26px;
  padding: 26px;
  text-align: center;
  box-shadow: 0 28px 80px rgba(31, 107, 87,.20);
}
.loader-spinner {
  width: 54px;
  height: 54px;
  margin: 0 auto 14px;
  border-radius: 999px;
  border: 6px solid rgba(47, 143, 111,.18);
  border-top-color: #1f6b57;
  animation: medibuddy-spin .8s linear infinite;
}
.loader-card b { display:block; color:#17203a; font-size:18px; }
.loader-card span { display:block; color:#64748b; font-weight:800; margin-top:5px; }
@keyframes medibuddy-spin { to { transform: rotate(360deg); } }

.back-btn, #chat-back-btn {
  background: #f8fafc !important;
  color: #17203a !important;
  border: 1px solid rgba(31, 107, 87,.14) !important;
  box-shadow: 0 8px 20px rgba(31, 107, 87,.07) !important;
}
.back-btn:hover, #chat-back-btn:hover { background:#e8f4ef !important; }



/* ---------- Next-level X-ray Result Enhancements ---------- */
.xray-preview-center {
  align-items: center !important;
  justify-content: center !important;
  background: rgba(255,255,255,.76);
  border: 1px solid rgba(31, 107, 87,.12);
  border-radius: 24px;
  padding: 18px;
  margin: 12px auto 18px;
  box-shadow: 0 18px 45px rgba(31, 107, 87,.08);
}
#xray-preview-image {
  max-width: 760px !important;
  margin: 0 auto !important;
}
#xray-preview-image img {
  object-fit: contain !important;
  max-height: 540px !important;
  border-radius: 18px !important;
}
#xray-pdf-download {
  border-radius: 18px !important;
  margin-top: 10px !important;
}
#xray-bottom-chat-btn {
  background: linear-gradient(135deg, #1f6b57, #1f6b57) !important;
  color: white !important;
  border: 0 !important;
  border-radius: 16px !important;
  font-weight: 900 !important;
  box-shadow: 0 16px 34px rgba(31, 107, 87,.20) !important;
}
#xray-bottom-chat-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 20px 44px rgba(31, 107, 87,.28) !important;
}
.xray-suggestion-card ul {
  margin: 8px 0 0 20px;
  padding: 0;
  color: #334155;
  line-height: 1.7;
  font-weight: 750;
}
.xray-suggestion-card li { margin-bottom: 8px; }
.simple-box {
  line-height: 1.75;
  font-size: 14px;
}

@media (max-width: 980px) {
 .hero-grid,
 .choice-grid,
 .lab-layout,
 .xray-layout-new,
 .chat-layout {
  grid-template-columns: 1fr;
 }
 .stats-grid,
 .how-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
 }
 .hero-art { min-height: 430px; }
}

@media (max-width: 720px) {
 .gradio-container { padding: 8px !important; }
 .app-shell { padding: 12px; border-radius: 22px; }
 .nav-pills { display: none; }
 .stats-grid,
 .how-grid,
 .finding-grid,
 .glance-grid {
  grid-template-columns: 1fr;
 }
 .metric-row { grid-template-columns: 1fr; }
 .metric-header { display: none; }
 .metric-cell { padding: 8px 12px; }
 .page-head { align-items: flex-start; flex-direction: column; }
 .progress-wrap { grid-template-columns: 1fr; }
 .progress-line { display:none; }
}


/* ---------- Next-level animated welcome page ---------- */
.welcome-animated {
  position: relative;
  overflow: hidden;
  border-radius: 28px;
  padding: 34px 22px 22px;
  background:
   radial-gradient(circle at 12% 18%, rgba(47, 143, 111,.22), transparent 28%),
   radial-gradient(circle at 82% 8%, rgba(31, 107, 87,.20), transparent 28%),
   radial-gradient(circle at 75% 86%, rgba(6,182,212,.16), transparent 32%),
   linear-gradient(135deg, rgba(255,255,255,.72), rgba(248,250,252,.86));
}
.welcome-animated::before,
.welcome-animated::after {
  content: "";
  position: absolute;
  width: 260px;
  height: 260px;
  border-radius: 999px;
  filter: blur(2px);
  opacity: .38;
  z-index: 0;
  animation: orbFloat 8s ease-in-out infinite;
}
.welcome-animated::before {
  left: -90px;
  top: -80px;
  background: linear-gradient(135deg, #2f8f6f, #06b6d4);
}
.welcome-animated::after {
  right: -95px;
  bottom: -110px;
  background: linear-gradient(135deg, #2f8f6f, #8fcfb9);
  animation-delay: -3s;
}
.welcome-animated > * {
  position: relative;
  z-index: 1;
}
.welcome-animated .hero-copy {
  animation: slideFadeUp .8s ease both;
}
.welcome-animated .micro-pill {
  animation: pillPulse 2.4s ease-in-out infinite;
}
.welcome-title-line {
  display: inline-block;
  background: linear-gradient(90deg, #111827, #1f6b57, #06b6d4);
  background-size: 220% auto;
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent !important;
  animation: shimmerText 4.5s linear infinite;
}
.welcome-subtitle {
  position: relative;
}
.live-dot {
  display: inline-block;
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #22c55e;
  margin-right: 8px;
  box-shadow: 0 0 0 rgba(34,197,94,.55);
  animation: dotPulse 1.8s infinite;
}
.animated-feature {
  opacity: 0;
  transform: translateY(18px);
  animation: slideFadeUp .72s ease both;
}
.animated-feature:nth-child(1) { animation-delay: .12s; }
.animated-feature:nth-child(2) { animation-delay: .25s; }
.animated-feature:nth-child(3) { animation-delay: .38s; }
.animated-feature:nth-child(4) { animation-delay: .51s; }
.welcome-visual-card {
  position: relative;
  width: min(430px, 100%);
  min-height: 420px;
  margin: 0 auto;
  border-radius: 32px;
  background: rgba(255,255,255,.58);
  border: 1px solid rgba(31, 107, 87,.16);
  box-shadow: 0 28px 80px rgba(30,41,59,.13);
  backdrop-filter: blur(15px);
  animation: visualFloat 4.5s ease-in-out infinite;
  overflow: hidden;
}
.welcome-visual-card::before {
  content: "";
  position: absolute;
  inset: 18px;
  border-radius: 24px;
  border: 1px dashed rgba(31, 107, 87,.22);
}
.medical-orbit {
  position: absolute;
  inset: 44px;
  border-radius: 999px;
  border: 2px solid rgba(31, 107, 87,.13);
  animation: orbitSpin 16s linear infinite;
}
.orbit-icon {
  position: absolute;
  display: grid;
  place-items: center;
  width: 48px;
  height: 48px;
  border-radius: 17px;
  background: white;
  box-shadow: 0 14px 30px rgba(15,23,42,.12);
  font-size: 24px;
}
.orbit-icon:nth-child(1) { top: -18px; left: 44%; }
.orbit-icon:nth-child(2) { right: -18px; top: 44%; }
.orbit-icon:nth-child(3) { bottom: -18px; left: 44%; }
.orbit-icon:nth-child(4) { left: -18px; top: 44%; }
.center-assistant {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  width: 185px;
  min-height: 210px;
  border-radius: 30px;
  background: linear-gradient(180deg, #ffffff, #e8f4ef);
  box-shadow: 0 22px 60px rgba(79,70,229,.18);
  display: grid;
  place-items: center;
  text-align: center;
  padding: 20px;
}
.assistant-face {
  width: 94px;
  height: 94px;
  display: grid;
  place-items: center;
  border-radius: 32px;
  background: linear-gradient(135deg, #2f8f6f, #2f8f6f);
  color: white;
  font-size: 48px;
  margin-bottom: 12px;
  animation: faceBounce 2.6s ease-in-out infinite;
}
.center-assistant b {
  color: #17203a;
  font-size: 16px;
}
.center-assistant span {
  color: #64748b;
  font-weight: 800;
  font-size: 12px;
}
.ecg-line {
  position: absolute;
  left: 36px;
  right: 36px;
  bottom: 42px;
  height: 42px;
  overflow: hidden;
  opacity: .92;
}
.ecg-line svg {
  width: 100%;
  height: 42px;
}
.ecg-line path {
  fill: none;
  stroke: #22c55e;
  stroke-width: 4;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-dasharray: 420;
  stroke-dashoffset: 420;
  animation: drawLine 2.4s ease-in-out infinite;
}
.float-bubble {
  animation: bubbleFloat 4s ease-in-out infinite;
}
.bubble-two { animation-delay: -1.3s; }
.bubble-three { animation-delay: -2.2s; }
.welcome-trust-row .trust-chip {
  animation: softPop .65s ease both;
}
.welcome-trust-row .trust-chip:nth-child(2) { animation-delay: .15s; }
.welcome-trust-row .trust-chip:nth-child(3) { animation-delay: .30s; }

@keyframes orbFloat {
  0%, 100% { transform: translate(0,0) scale(1); }
  50% { transform: translate(24px, 18px) scale(1.08); }
}
@keyframes slideFadeUp {
  from { opacity: 0; transform: translateY(24px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes pillPulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(31, 107, 87,.0); }
  50% { box-shadow: 0 0 0 9px rgba(31, 107, 87,.08); }
}
@keyframes shimmerText {
  0% { background-position: 0% center; }
  100% { background-position: 220% center; }
}
@keyframes dotPulse {
  0% { box-shadow: 0 0 0 0 rgba(34,197,94,.55); }
  70% { box-shadow: 0 0 0 10px rgba(34,197,94,0); }
  100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}
@keyframes visualFloat {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-12px); }
}
@keyframes orbitSpin {
  to { transform: rotate(360deg); }
}
@keyframes faceBounce {
  0%, 100% { transform: translateY(0) rotate(0deg); }
  50% { transform: translateY(-8px) rotate(2deg); }
}
@keyframes drawLine {
  0% { stroke-dashoffset: 420; }
  55% { stroke-dashoffset: 0; }
  100% { stroke-dashoffset: -420; }
}
@keyframes bubbleFloat {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-12px); }
}
@keyframes softPop {
  from { opacity: 0; transform: scale(.92); }
  to { opacity: 1; transform: scale(1); }
}



/* ---------- Exact animated welcome landing page v2 ---------- */
.welcome-animated {
  background: #f7fbf8 !important;
  border: 1px solid rgba(16, 86, 68, .08) !important;
  border-radius: 0 !important;
  padding: 0 !important;
  box-shadow: none !important;
  overflow: hidden;
  color: #12342b;
}
.welcome-landing {
  min-height: 900px;
  padding: 28px 32px 24px;
  background:
   radial-gradient(circle at 82% 15%, rgba(191, 220, 195, .55), transparent 24%),
   linear-gradient(180deg, #f9fdf9 0%, #f7fbf8 100%);
  animation: landingFade .65s ease both;
}
.welcome-mini-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 88px;
}
.welcome-brand {
  display:flex;
  align-items:center;
  gap:8px;
  font-size:13px;
  font-weight:800;
  color:#163c31;
}
.welcome-brand-mark {
  width:24px;
  height:24px;
  border-radius:8px;
  display:grid;
  place-items:center;
  background:#1f4c3f;
  color:white;
  box-shadow:0 8px 18px rgba(32,76,63,.16);
  animation: softPulse 2.8s ease-in-out infinite;
}
.welcome-secure-tag {
  font-size:12px;
  color:#345b50;
  font-weight:700;
}
.landing-hero {
  display:grid;
  grid-template-columns: 1fr .9fr;
  gap:44px;
  align-items:center;
  margin-bottom:110px;
}
.hero-kicker {
  display:inline-flex;
  align-items:center;
  gap:7px;
  padding:7px 12px;
  border-radius:999px;
  border:1px solid rgba(58,129,105,.18);
  background:#eff8f1;
  color:#397661;
  font-size:12px;
  font-weight:800;
  margin-bottom:18px;
}
.kicker-dot {
  width:7px;
  height:7px;
  border-radius:50%;
  background:#3a9b78;
  box-shadow:0 0 0 5px rgba(58,155,120,.12);
}
.landing-title {
  font-size:56px;
  line-height:1.03;
  letter-spacing:-2.3px;
  margin:0;
  color:#17372f;
  font-weight:950;
}
.landing-title .highlight {
  color:#3f8b76;
  position:relative;
  display:inline-block;
}
.landing-title .highlight:before,
.landing-title .highlight:after {
  content:"";
  position:absolute;
  left:-4px;
  right:-4px;
  height:18px;
  background:rgba(146, 194, 169, .35);
  z-index:-1;
}
.landing-title .highlight:before { top:6px; }
.landing-title .highlight:after { bottom:2px; }
.landing-copy {
  max-width:535px;
  margin:22px 0 0;
  font-size:15px;
  line-height:1.72;
  color:#668078;
  font-weight:600;
}
.welcome-actions {
  display:flex;
  align-items:center;
  gap:14px;
  margin-top:26px;
  flex-wrap:wrap;
}
.fake-start-pill {
  display:inline-flex;
  align-items:center;
  gap:12px;
  padding:16px 21px;
  border-radius:999px;
  background:#204f42;
  color:white;
  font-size:14px;
  font-weight:900;
  box-shadow:0 18px 35px rgba(32,79,66,.25);
  animation: buttonFloat 3s ease-in-out infinite;
}
.fake-start-pill span {
  width:25px;
  height:25px;
  border-radius:50%;
  background:rgba(255,255,255,.13);
  display:grid;
  place-items:center;
}
.welcome-action-chip {
  color:#5e796f;
  font-size:12px;
  font-weight:800;
}
.landing-visual-card {
  position:relative;
  min-height:315px;
  border-radius:14px;
  overflow:hidden;
  background:
    radial-gradient(circle at 82% 18%, #91c09e 0 19%, transparent 20%),
    radial-gradient(circle at 14% 12%, #e5c59a 0 18%, transparent 19%),
    linear-gradient(135deg, #d8e9db 0%, #9ac7a8 42%, #6fa78c 100%);
  box-shadow:0 22px 45px rgba(58, 94, 76, .18);
  animation: cardRise .8s ease both, visualBreath 5s ease-in-out infinite;
}
.landing-visual-card:before {
  content:"";
  position:absolute;
  width:170px;
  height:170px;
  border-radius:42px;
  background:rgba(255,255,255,.42);
  left:118px;
  top:80px;
  transform:rotate(-17deg);
  filter:blur(.2px);
}
.medical-device {
  position:absolute;
  width:150px;
  height:96px;
  left:145px;
  top:112px;
  border-radius:18px;
  background:#f8fbf8;
  transform:rotate(-16deg);
  box-shadow:0 22px 28px rgba(28,66,55,.18);
  animation: deviceFloat 4s ease-in-out infinite;
}
.medical-device:before {
  content:"";
  position:absolute;
  left:23px;
  right:23px;
  top:22px;
  height:18px;
  border-radius:12px;
  background:linear-gradient(90deg,#89ba9f,#d8eadc);
  box-shadow:0 28px 0 #ddebe0;
}
.medical-device:after {
  content:"";
  position:absolute;
  width:42px;
  height:42px;
  border-radius:50%;
  right:18px;
  bottom:14px;
  background:conic-gradient(#346e5e 0 35%, #d9eadf 35% 100%);
}
.status-card {
  position:absolute;
  left:26px;
  bottom:28px;
  display:flex;
  align-items:center;
  gap:12px;
  padding:13px 17px;
  border-radius:14px;
  background:rgba(255,255,255,.86);
  box-shadow:0 16px 30px rgba(31,74,61,.18);
  backdrop-filter:blur(8px);
  animation: bubbleFloat 3.5s ease-in-out infinite;
}
.status-card .status-icon {
  width:34px;
  height:34px;
  display:grid;
  place-items:center;
  border-radius:50%;
  background:#e8f6ee;
  color:#347a64;
  font-weight:900;
}
.status-card b { display:block; color:#183d33; font-size:13px; }
.status-card span { display:block; color:#6c837b; font-size:11px; font-weight:700; }
.section-divider { height:1px; background:#e7efeb; margin-bottom:48px; }
.landing-lower {
  display:grid;
  grid-template-columns: 1.05fr 1.15fr;
  gap:44px;
  align-items:start;
}
.landing-section-title {
  margin:0 0 28px;
  color:#17372f;
  font-size:17px;
  font-weight:950;
}
.how-roadmap {
  position:relative;
  min-height:235px;
  padding-left:0;
}
.road-line {
  position:absolute;
  left:185px;
  top:22px;
  width:2px;
  height:190px;
  background:#dbe8e2;
}
.road-step {
  position:absolute;
  width:185px;
  padding:18px 18px;
  border-radius:12px;
  background:white;
  border:1px solid #dce7e2;
  box-shadow:0 10px 22px rgba(32,79,66,.06);
  animation: softPop .65s ease both;
}
.road-step:nth-child(2) { left:218px; top:0; animation-delay:.08s; }
.road-step:nth-child(3) { left:0; top:82px; animation-delay:.17s; }
.road-step:nth-child(4) { left:218px; top:166px; animation-delay:.25s; }
.road-num {
  position:absolute;
  width:26px;
  height:26px;
  border-radius:50%;
  background:#2a6756;
  color:white;
  display:grid;
  place-items:center;
  font-size:12px;
  font-weight:900;
}
.road-num.n1 { left:180px; top:23px; }
.road-num.n2 { left:180px; top:106px; }
.road-num.n3 { left:180px; top:190px; }
.road-step b { display:block; color:#21463c; font-size:13px; margin-bottom:5px; }
.road-step span { display:block; color:#778b84; font-size:11px; line-height:1.45; font-weight:650; }
.why-grid {
  display:grid;
  grid-template-columns:repeat(2, minmax(0,1fr));
  gap:18px;
}
.why-card {
  background:white;
  border:1px solid #dce7e2;
  border-radius:12px;
  padding:20px 18px;
  min-height:105px;
  box-shadow:0 10px 22px rgba(32,79,66,.05);
  animation: cardRise .7s ease both;
}
.why-card:nth-child(2){animation-delay:.05s}.why-card:nth-child(3){animation-delay:.1s}.why-card:nth-child(4){animation-delay:.15s}
.why-icon {
  width:32px;
  height:32px;
  border-radius:10px;
  display:grid;
  place-items:center;
  background:#eff7f2;
  margin-bottom:13px;
}
.why-card b { color:#21463c; font-size:13px; display:block; margin-bottom:5px; }
.why-card span { color:#758b83; font-size:11px; line-height:1.45; font-weight:650; }
.landing-footer {
  border-top:1px solid #e7efeb;
  margin-top:66px;
  padding-top:22px;
  display:flex;
  justify-content:center;
  gap:32px;
  color:#688178;
  font-weight:800;
  font-size:12px;
}
#get-started-btn {
  max-width: 178px !important;
  min-height: 54px !important;
  border: 0 !important;
  border-radius: 999px !important;
  padding: 0 20px !important;
  background: #204f42 !important;
  color: #ffffff !important;
  font-size: 13px !important;
  font-weight: 900 !important;
  box-shadow: 0 18px 35px rgba(32,79,66,.25) !important;
  position: relative;
  margin-top: -603px !important;
  margin-left: 32px !important;
  z-index: 8;
  animation: buttonFloat 3s ease-in-out infinite;
}
#get-started-btn::after {
  content: "";
  display:inline-grid;
  place-items:center;
  margin-left:10px;
  width:23px;
  height:23px;
  border-radius:50%;
  background:rgba(255,255,255,.13);
}
.secure-line {
  display:none !important;
}
.hero-cta {
  margin-top: 0 !important;
  min-height: 0 !important;
  display:block !important;
}
@keyframes landingFade { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
@keyframes cardRise { from { opacity:0; transform:translateY(18px); } to { opacity:1; transform:translateY(0); } }
@keyframes visualBreath { 0%,100% { transform:translateY(0) scale(1); } 50% { transform:translateY(-5px) scale(1.01); } }
@keyframes deviceFloat { 0%,100% { transform:rotate(-16deg) translateY(0); } 50% { transform:rotate(-13deg) translateY(-10px); } }
@keyframes buttonFloat { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-4px); } }
@keyframes softPulse { 0%,100% { box-shadow:0 8px 18px rgba(32,76,63,.16); } 50% { box-shadow:0 8px 24px rgba(32,76,63,.30); } }

@media (max-width: 900px) {
  .welcome-landing { min-height:auto; padding:20px; }
  .welcome-mini-nav { margin-bottom:44px; }
  .landing-hero, .landing-lower { grid-template-columns:1fr; margin-bottom:60px; }
  .landing-title { font-size:40px; }
  #get-started-btn { margin-top: -900px !important; margin-left: 20px !important; }
  .how-roadmap { min-height:520px; }
  .road-line, .road-num { display:none; }
  .road-step { position:relative !important; left:auto !important; top:auto !important; width:auto; margin-bottom:14px; }
  .why-grid { grid-template-columns:1fr; }
}

/* ---------- Final welcome polish: button placement, orbit visual, title pop animation ---------- */
.landing-title {
  animation: welcomeTitlePop .95s cubic-bezier(.16, 1.35, .32, 1) both;
  transform-origin: left center;
}
.landing-title .highlight {
  animation: highlightJump 1.15s cubic-bezier(.16, 1.35, .32, 1) .18s both;
}
.welcome-orbit-visual {
  min-height: 420px !important;
  border-radius: 32px !important;
  background:
   radial-gradient(circle at 20% 20%, rgba(31, 107, 87, .08), transparent 26%),
   radial-gradient(circle at 78% 70%, rgba(31, 107, 87, .09), transparent 28%),
   rgba(255,255,255,.62) !important;
  border: 1px solid rgba(31, 107, 87,.16) !important;
  box-shadow: 0 28px 80px rgba(30,41,59,.13) !important;
}
.hero-cta {
  margin-top: -74px !important;
  margin-left: 38px !important;
  margin-bottom: 22px !important;
  position: relative !important;
  z-index: 20 !important;
  width: fit-content !important;
}
#get-started-btn {
  transform: translateY(-12px) !important;
}
#get-started-btn:hover {
  transform: translateY(-16px) scale(1.03) !important;
}
@keyframes welcomeTitlePop {
  0% { opacity: 0; transform: translateY(28px) scale(.92); filter: blur(7px); }
  58% { opacity: 1; transform: translateY(-8px) scale(1.035); filter: blur(0); }
  78% { transform: translateY(3px) scale(.99); }
  100% { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes highlightJump {
  0% { transform: translateY(18px) scale(.96); opacity: .72; }
  55% { transform: translateY(-10px) scale(1.04); opacity: 1; }
  78% { transform: translateY(3px) scale(.99); }
  100% { transform: translateY(0) scale(1); opacity: 1; }
}
@media (max-width: 900px) {
  .hero-cta {
    margin-top: -54px !important;
    margin-left: 28px !important;
    margin-bottom: 18px !important;
  }
  #get-started-btn { transform: translateY(-8px) !important; }
}

/* ---------- Exact redesigned Page 2: choose report type ---------- */
.choose-page-v2 {
  max-width: 960px;
  margin: 2px auto 0;
}
.choose-progress-v2 {
  position: relative;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  align-items: start;
  max-width: 820px;
  margin: 0 auto 28px;
  padding: 0 6px;
}
.choose-progress-line {
  position: absolute;
  left: 15%;
  right: 15%;
  top: 20px;
  height: 2px;
  background: linear-gradient(90deg, #215c4c 0 52%, rgba(33,92,76,.15) 52% 100%);
  z-index: 0;
}
.choose-progress-item {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  text-align: center;
  color: rgba(33, 92, 76, .38);
  text-transform: uppercase;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: .12em;
}
.choose-progress-dot {
  width: 42px;
  height: 42px;
  border-radius: 999px;
  display: grid;
  place-items: center;
  background: rgba(255,255,255,.92);
  border: 1px solid rgba(33, 92, 76, .14);
  box-shadow: 0 12px 26px rgba(33, 92, 76, .08);
  color: rgba(33, 92, 76, .38);
  font-size: 16px;
}
.choose-progress-item.done .choose-progress-dot,
.choose-progress-item.active .choose-progress-dot {
  background: #215c4c;
  color: white;
  box-shadow: 0 14px 28px rgba(33, 92, 76, .20);
}
.choose-progress-item.done,
.choose-progress-item.active { color: #215c4c; }
.choose-progress-item.upcoming .choose-progress-dot { background: #f7fbf8; }
.choose-back-row-v2 {
  max-width: 960px;
  margin: -4px auto 2px !important;
  justify-content: flex-start !important;
}
#back-to-welcome-btn.choose-back-btn-v2,
.choose-back-btn-v2 button,
#back-to-welcome-btn {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  color: #2f7a68 !important;
  font-weight: 900 !important;
  min-height: 30px !important;
  padding: 0 !important;
  max-width: 130px !important;
  text-align: left !important;
}
.choose-copy-v2 {
  text-align: center;
  color: #17392f;
  margin: 0 auto;
}
.choose-kicker-v2 {
  color: #399276;
  font-size: 12px;
  letter-spacing: .12em;
  font-weight: 900;
  margin-bottom: 28px;
}
.choose-copy-v2 h2 {
  margin: 0;
  font-size: clamp(34px, 4vw, 48px);
  line-height: .98;
  letter-spacing: -.055em;
  font-weight: 950;
  color: #193f34;
}
.choose-copy-v2 p {
  margin: 18px 0 22px;
  color: #607b72;
  font-size: 16px;
  line-height: 1.55;
  font-weight: 800;
}
.choose-info-v2 {
  max-width: 650px;
  margin: 0 auto 24px;
  padding: 13px 18px;
  border-radius: 14px;
  border: 1px solid rgba(33,92,76,.18);
  background: rgba(255,255,255,.62);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.75);
  color: #54746b;
  font-weight: 800;
  text-align: left;
}
.choose-info-v2 span {
  display: inline-grid;
  place-items: center;
  width: 22px;
  height: 22px;
  border-radius: 8px;
  margin-right: 8px;
  background: rgba(33,92,76,.10);
  color: #215c4c;
}
.choice-grid-v2 {
  max-width: 1020px;
  margin: 0 auto 28px !important;
  display: grid !important;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 24px !important;
  align-items: stretch !important;
}
.choice-grid-v2 > div {
  min-width: 0 !important;
}
.choice-card-v2 {
  position: relative;
  min-height: 318px;
  padding: 40px 28px 26px !important;
  background: rgba(255,255,255,.90);
  border: 1px solid rgba(33,92,76,.12);
  border-radius: 20px;
  box-shadow: 0 22px 50px rgba(33,92,76,.08);
  text-align: center;
  transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
}
.choice-card-v2:hover {
  transform: translateY(-7px);
  box-shadow: 0 30px 60px rgba(33,92,76,.14);
  border-color: rgba(33,92,76,.24);
}
.choice-badge-v2 {
  position: absolute;
  top: 28px;
  right: 28px;
  padding: 4px 14px;
  border-radius: 999px;
  background: rgba(33,92,76,.10);
  color: #53736a;
  font-size: 10px;
  font-weight: 950;
  letter-spacing: .06em;
}
.choice-icon-v2 {
  width: 82px;
  height: 82px;
  border-radius: 22px;
  display: grid;
  place-items: center;
  margin: 28px auto 20px;
  font-size: 38px;
  font-weight: 950;
}
.lab-icon-v2 { background: #d9f1e8; color: #215c4c; }
.xray-icon-v2 { background: #e4efff; color: #27466c; }
.choice-card-v2 h3 {
  margin: 0 0 10px;
  color: #193f34;
  font-size: 20px;
  font-weight: 950;
}
.choice-card-v2 p {
  min-height: 70px;
  margin: 0 0 18px;
  color: #648078;
  font-size: 14px;
  line-height: 1.45;
  font-weight: 800;
}
#choice-lab-btn, #choice-xray-btn {
  min-height: 54px !important;
  border-radius: 14px !important;
  font-size: 15px !important;
  font-weight: 950 !important;
  border: none !important;
  box-shadow: 0 14px 24px rgba(33,92,76,.16) !important;
}
#choice-lab-btn { background: #225b4b !important; }
#choice-xray-btn { background: #284766 !important; }
.choose-mini-steps-v2 {
  max-width: 650px;
  margin: 0 auto 10px;
  padding: 18px 22px;
  border-radius: 18px;
  background: rgba(255,255,255,.76);
  border: 1px solid rgba(33,92,76,.12);
  box-shadow: 0 20px 48px rgba(33,92,76,.07);
  display: grid;
  grid-template-columns: 1fr 20px 1fr 20px 1fr 20px 1fr;
  gap: 8px;
  align-items: start;
  text-align: center;
}
.choose-mini-steps-v2 em {
  color: rgba(33,92,76,.20);
  font-style: normal;
  font-weight: 950;
  padding-top: 10px;
}
.mini-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 3px;
  color: #9aaca6;
  font-weight: 900;
}
.mini-step b {
  width: 32px;
  height: 32px;
  border-radius: 999px;
  display: grid;
  place-items: center;
  background: #edf6f2;
  color: #9aaca6;
}
.mini-step.active b { background: #215c4c; color: white; }
.mini-step span { color: #56746b; font-size: 12px; }
.mini-step small { color: #9aaca6; font-size: 10px; font-weight: 800; }
@media (max-width: 760px) {
  .choice-grid-v2 { grid-template-columns: 1fr; max-width: 420px; }
  .choose-mini-steps-v2 { grid-template-columns: 1fr; }
  .choose-mini-steps-v2 em { display: none; }
  .choose-progress-v2 { margin-bottom: 18px; }
  .choose-copy-v2 h2 { font-size: 34px; }
}




/* ===== Redesigned Lab Report Page ===== */
.lab-redesign {
  background: linear-gradient(180deg, #f7fffb 0%, #eef8f3 100%);
  border-radius: 28px;
  padding: 24px;
  border: 1px solid rgba(26, 95, 79, .12);
}
.lab-hero-head {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: flex-start;
  padding: 18px 18px 20px;
  border-radius: 24px;
  background: rgba(255,255,255,.82);
  border: 1px solid rgba(28, 105, 86, .14);
  box-shadow: 0 18px 44px rgba(20, 82, 67, .08);
  margin-bottom: 16px;
}
.lab-hero-head h2 { margin: 6px 0 4px; color: #173f35; font-size: 30px; font-weight: 900; }
.lab-hero-head p, .card-subtitle { margin: 0; color: #6a817a; font-weight: 800; }
.step-kicker { color: #2f8b70; text-transform: uppercase; letter-spacing: .08em; font-size: 12px; font-weight: 900; }
.green-chip { background: #d9fae8 !important; color: #0f7a51 !important; border-color: rgba(15, 122, 81, .16) !important; }
.lab-stats-modern .stat-card { border: 1px solid rgba(23, 63, 53, .08); box-shadow: 0 12px 26px rgba(23, 63, 53, .07); }
.chart-card { margin: 16px 0; }
.bar-row { display: grid; grid-template-columns: 120px 1fr 32px; align-items: center; gap: 12px; margin: 12px 0; color: #173f35; font-weight: 900; }
.bar-row div { height: 10px; background: #e8f0ec; border-radius: 999px; overflow: hidden; }
.bar-row i { display: block; height: 100%; border-radius: 999px; background: #20c878; }
.bar-row.low i { background: #f59e0b; }
.bar-row.high i { background: #ef4444; }
.lab-main-grid { display: grid; grid-template-columns: .9fr 1.1fr; gap: 16px; align-items: start; }
.lab-info-stack { display: grid; gap: 16px; align-content: start; }
.lab-main-grid .soft-card { min-height: 100%; }
.attention-card { grid-column: 2; grid-row: 1; }
.ai-card-wide { grid-column: 1 / -1; }
.lab-attention-list { display: grid; gap: 10px; margin-top: 14px; }
.lab-attention-item { display: flex; justify-content: space-between; gap: 12px; align-items: center; padding: 13px 14px; background: #fbfefc; border: 1px solid rgba(28, 105, 86, .10); border-radius: 16px; }
.lab-attention-item b { display: block; color: #173f35; font-size: 14px; }
.lab-attention-item span { display: block; color: #71857f; font-size: 12px; font-weight: 800; margin-top: 2px; }
.lab-ai-bubble { background: #f4fbf8 !important; border-color: rgba(28, 105, 86, .12) !important; }
.inner-card { background: transparent !important; border: 0 !important; padding: 0 !important; box-shadow: none !important; }
.info-clean-card .inner-card > .card-title { display:none; }
.table-card-modern { overflow: hidden; }
.table-card-modern table { border-radius: 16px; overflow: hidden; }
@media (max-width: 850px) { .lab-hero-head, .lab-main-grid { grid-template-columns: 1fr; display: grid; } .ai-card-wide { grid-column: auto; } }


/* === FIXED GREEN PROGRESS TRACKER (matches reference image) === */
.progress-stage {
  position: relative;
  width: min(760px, 92%);
  margin: 10px auto 22px auto;
  padding: 0 0 6px;
  min-height: 72px;
}
.progress-stage .progress-rail {
  position: absolute;
  top: 18px;
  left: 42px;
  right: 42px;
  height: 3px;
  background: #1f6b57;
  opacity: 1;
  border-radius: 999px;
  z-index: 0;
}
.progress-stage .progress-items {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  align-items: start;
}
.progress-stage .progress-item {
  text-align: center;
  color: #1c2f2b;
}
.progress-stage .progress-dot {
  width: 42px;
  height: 42px;
  border-radius: 50%;
  margin: 0 auto 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 900;
  font-size: 17px;
  line-height: 1;
  background: #eff7f4;
  color: #88aaa1;
  border: 3px solid #eff7f4;
  box-shadow: 0 6px 18px rgba(31,107,87,.10);
}
.progress-stage .progress-item.done .progress-dot,
.progress-stage .progress-item.active .progress-dot {
  background: #1f6b57;
  border-color: #1f6b57;
  color: #ffffff;
}
.progress-stage .progress-item.upcoming .progress-dot {
  background: #f5fbf8;
  border-color: #e4f1ed;
  color: #9fbeb5;
}
.progress-stage .progress-label {
  font-size: 12px;
  font-weight: 900;
  letter-spacing: .12em;
  color: #182d29;
}
.progress-stage .progress-item.upcoming .progress-label {
  color: #7f9690;
}
.progress-page-label {
  text-align: center;
  margin-top: 2px;
  color: #1f6b57;
  font-size: 12px;
  font-weight: 900;
  text-transform: none;
}

/* Hide older/broken progress fragments if they remain anywhere */
.clean-progress-wrap,
.clean-progress-dots,
.clean-progress-labels,
.clean-step,
.clean-dot,
.clean-label,
.clean-page-label {
  display: none !important;
}

/* === FORCE GRADIO FILE UPLOAD FROM BLUE TO GREEN === */
.gradio-container {
  --color-accent: #1f6b57 !important;
  --color-accent-soft: #e7f4ef !important;
  --button-primary-background-fill: linear-gradient(135deg, #1f6b57, #2f8f6f) !important;
  --button-primary-background-fill-hover: linear-gradient(135deg, #195948, #267c61) !important;
  --button-primary-text-color: #ffffff !important;
  --link-text-color: #1f6b57 !important;
  --link-text-color-hover: #174c3f !important;
}
.upload-card label,
.upload-card .label-wrap,
.upload-card .file-label,
.upload-card .file-title,
.upload-card .upload-title {
  color: #172d29 !important;
}
.upload-card button,
.upload-card .file-preview button,
.upload-card [role="button"],
.upload-card a,
.upload-card svg,
.upload-card .icon,
.upload-card .upload-icon {
  color: #1f6b57 !important;
  border-color: rgba(31,107,87,.22) !important;
}
.upload-card .file-preview,
.upload-card .wrap,
.upload-card .dropzone,
.upload-card [data-testid="file-upload"],
.upload-card [data-testid="file"],
.upload-card [data-testid="file-upload-label"],
.upload-card .upload-container {
  border-color: rgba(31,107,87,.35) !important;
  background: #ffffff !important;
}
.upload-card .dropzone:hover,
.upload-card [data-testid="file-upload"]:hover,
.upload-card [data-testid="file"]:hover {
  border-color: #1f6b57 !important;
  background: #f2faf6 !important;
}
.upload-card .file-preview a,
.upload-card .file-preview span,
.upload-card [data-testid="file"] span,
.upload-card [data-testid="file-upload"] span,
.upload-card .upload-text,
.upload-card p {
  color: #1f6b57 !important;
}
button.primary,
button[variant="primary"],
.primary {
  background: linear-gradient(135deg, #1f6b57, #2f8f6f) !important;
  border-color: #1f6b57 !important;
  color: #ffffff !important;
}



/* ================= ADVANCED GREEN SAAS DASHBOARD OVERRIDES ================= */
:root { --mb-green:#0f8f68; --mb-dark:#1f6b57; --mb-mint:#eaf8f2; --mb-line:rgba(31,107,87,.16); --mb-shadow:0 18px 50px rgba(31,107,87,.12); }
.gradio-container { background: linear-gradient(180deg,#f6fbf8 0%,#eef7f2 100%) !important; }
button, .gr-button { border-radius: 14px !important; font-weight: 900 !important; transition: transform .18s ease, box-shadow .18s ease, background .18s ease !important; }
button:hover, .gr-button:hover { transform: translateY(-2px); box-shadow: 0 12px 28px rgba(15,143,104,.18) !important; }
#lab-analyze-btn, #xray-analyze-btn, #ask-btn, #lab-ask-btn { background: linear-gradient(135deg,#0f8f68,#047857) !important; color:#fff !important; border:0 !important; }
#pdf-download, #xray-pdf-download { background:#fff !important; border:1px solid var(--mb-line) !important; border-radius:16px !important; box-shadow:0 10px 26px rgba(31,107,87,.07) !important; color:var(--mb-dark) !important; }
.upload-card { background:rgba(255,255,255,.92) !important; border:1px solid var(--mb-line) !important; border-radius:28px !important; box-shadow:var(--mb-shadow) !important; padding:18px !important; overflow:hidden; }
.premium-upload-title { font-size:20px !important; color:#173c35 !important; letter-spacing:-.02em; margin-bottom:12px !important; }
.premium-drop-helper { min-height:150px; border:2px dashed rgba(15,143,104,.34); border-radius:24px; display:grid; place-items:center; text-align:center; padding:20px; background:linear-gradient(180deg,rgba(234,248,242,.72),rgba(255,255,255,.94)); color:#173c35; margin-bottom:14px; }
.premium-drop-helper b { display:block; font-size:17px; margin-top:8px; }
.premium-drop-helper span { display:block; color:#6b7f78; font-weight:800; font-size:13px; }
.premium-drop-helper em { display:inline-block; margin-top:8px; font-style:normal; color:var(--mb-green); font-weight:900; animation: jumpPop 1.7s ease-in-out infinite; }
.cloud-pulse { width:54px; height:54px; display:grid; place-items:center; font-size:30px; color:var(--mb-green); border-radius:50%; background:#e8fff4; animation:pulseGlow 1.9s ease-in-out infinite; }
.upload-card input, .upload-card textarea, .upload-card .wrap, .upload-card .block, .upload-card [data-testid="file"] { border-color:rgba(15,143,104,.25)!important; color:var(--mb-dark)!important; }
.upload-card .file, .upload-card .file-preview, .upload-card [data-testid="file"] { background:#f7fcfa!important; border-radius:18px!important; }
.upload-card [data-testid="file"] button, .upload-card button[aria-label*="Remove"], .upload-card button[title*="Remove"] { color:var(--mb-green)!important; background:#eaf8f2!important; border:1px solid rgba(15,143,104,.18)!important; }
.remove-file-btn, #lab-clear-file-btn, #xray-clear-file-btn { background:#fff!important; color:var(--mb-green)!important; border:2px solid rgba(15,143,104,.65)!important; border-radius:16px!important; box-shadow:none!important; }
.status-box textarea { border:1px solid rgba(15,143,104,.25)!important; background:#f7fcfa!important; color:#173c35!important; }
.saas-dashboard { display:grid; gap:18px; margin-top:10px; }
.saas-result-header { display:flex; justify-content:space-between; align-items:center; gap:18px; background:linear-gradient(135deg,#fff,#f0fbf6); border:1px solid var(--mb-line); border-radius:26px; padding:22px; box-shadow:var(--mb-shadow); }
.saas-kicker { color:var(--mb-green); font-size:12px; text-transform:uppercase; letter-spacing:.08em; font-weight:1000; }
.saas-result-header h2 { margin:6px 0; font-size:30px; color:#173c35; letter-spacing:-.04em; }
.saas-result-header p { margin:0; color:#65756f; font-weight:800; }
.saas-status-pill { padding:12px 16px; border-radius:999px; font-weight:1000; white-space:nowrap; }
.saas-status-pill.ok { background:#dcfce7; color:#15803d; } .saas-status-pill.warn { background:#fff7ed; color:#c2410c; } .saas-status-pill.soft { background:#eef7f2; color:var(--mb-dark); }
.saas-widget-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
.saas-widget { background:#fff; border:1px solid var(--mb-line); border-radius:22px; padding:18px; box-shadow:0 14px 35px rgba(31,107,87,.08); display:grid; gap:4px; }
.saas-widget span { width:38px; height:38px; border-radius:14px; display:grid; place-items:center; background:#e8fff4; color:var(--mb-green); font-size:20px; }
.saas-widget b { color:#173c35; font-size:15px; } .saas-widget small { color:#73847e; font-weight:800; }
.saas-two-col { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.saas-card { background:rgba(255,255,255,.94); border:1px solid var(--mb-line); border-radius:24px; padding:20px; box-shadow:0 15px 40px rgba(31,107,87,.08); }
.saas-card-title { color:#173c35; font-size:16px; font-weight:1000; margin-bottom:12px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.saas-info-row { display:grid; grid-template-columns:120px 1fr; padding:10px 0; border-top:1px solid rgba(31,107,87,.09); }
.saas-info-row:first-of-type { border-top:0; } .saas-info-row span { color:#6b7f78; font-weight:900; } .saas-info-row b { color:#173c35; }
.saas-findings-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
.saas-finding-card { display:flex; gap:10px; align-items:flex-start; padding:13px; border:1px solid rgba(244,63,94,.16); background:#fffafa; border-radius:18px; }
.finding-alert-icon { width:28px; height:28px; flex:0 0 auto; display:grid; place-items:center; border-radius:10px; background:#fff1f2; color:#e11d48; font-weight:1000; }
.saas-finding-card p, .simple-readable, .saas-summary-text { margin:0; color:#34433e; line-height:1.55; font-weight:800; }
.saas-alert { display:flex; gap:14px; align-items:center; border-radius:22px; padding:18px; border:1px solid rgba(245,158,11,.40); background:linear-gradient(135deg,#fff7ed,#fff); color:#9a3412; box-shadow:0 14px 35px rgba(245,158,11,.10); }
.saas-alert-icon { width:46px; height:46px; border-radius:50%; display:grid; place-items:center; background:#ffedd5; font-size:24px; animation:pulseGlow 2s infinite; }
.saas-alert p { margin:3px 0 0; font-weight:800; color:#7c2d12; }
.saas-disclaimer { border:1px solid rgba(245,158,11,.32); background:#fffaf0; color:#9a3412; border-radius:18px; padding:14px 16px; font-weight:900; }
.premium-chat-shell, .assistant-dock { background:linear-gradient(135deg,#ffffff,#f0fbf6)!important; border:1px solid var(--mb-line)!important; border-radius:28px!important; box-shadow:var(--mb-shadow)!important; padding:22px!important; }
.assistant-avatar { background:linear-gradient(135deg,#dffbea,#fff)!important; border:1px solid rgba(15,143,104,.2)!important; box-shadow:0 10px 26px rgba(15,143,104,.12)!important; }
.assistant-chip { background:#f2fbf7!important; color:var(--mb-green)!important; border:1px solid rgba(15,143,104,.20)!important; border-radius:999px!important; padding:10px 16px!important; font-weight:1000!important; box-shadow:0 8px 20px rgba(31,107,87,.06)!important; }
.chatbot-box { border:1px solid var(--mb-line)!important; border-radius:24px!important; box-shadow:0 15px 40px rgba(31,107,87,.08)!important; overflow:hidden!important; background:#fff!important; }
.chat-input-row textarea { border-radius:18px!important; border:1px solid rgba(15,143,104,.25)!important; background:#fff!important; }
.xray-preview-center { background:#fff!important; border:1px solid var(--mb-line)!important; border-radius:28px!important; padding:22px!important; box-shadow:var(--mb-shadow)!important; align-items:center!important; }
#xray-preview-image img { object-fit:contain!important; max-height:560px!important; margin:auto!important; border-radius:12px!important; box-shadow:0 16px 45px rgba(0,0,0,.14)!important; }
.fade-in-up, .slide-up-card { animation: fadeUp .55s ease both; }
.pop-card:hover, .saas-card:hover, .saas-widget:hover { transform:translateY(-3px); box-shadow:0 20px 52px rgba(31,107,87,.12); transition:.2s ease; }
.jump-text { display:inline-block; color:var(--mb-green); background:#e8fff4; border:1px solid rgba(15,143,104,.18); padding:4px 10px; border-radius:999px; font-size:12px; font-weight:1000; animation: jumpPop 1.9s ease-in-out infinite; }
.pulse-soft { animation:pulseGlow 2.2s ease-in-out infinite; }
@keyframes fadeUp { from { opacity:0; transform:translateY(18px); } to { opacity:1; transform:translateY(0); } }
@keyframes jumpPop { 0%,100%{ transform:translateY(0) scale(1); } 45%{ transform:translateY(-4px) scale(1.04); } }
@keyframes pulseGlow { 0%,100%{ box-shadow:0 0 0 0 rgba(15,143,104,.18); } 50%{ box-shadow:0 0 0 12px rgba(15,143,104,0); } }


/* =============== X-RAY SINGLE UPLOAD + OPEN CHATBOT POLISH =============== */
.single-upload-card { padding: 22px !important; }
.single-upload-card .premium-upload-title { margin-bottom: 14px !important; }
.xray-upload-only .premium-drop-helper { display:none !important; }
.single-green-file,
.single-green-file .wrap,
.single-green-file .block,
.single-green-file [data-testid="file-upload"],
.single-green-file [data-testid="file"],
.single-green-file [data-testid="file-upload-label"] {
  border: 2px dashed rgba(15,143,104,.36) !important;
  background: linear-gradient(180deg, rgba(238,251,245,.85), #ffffff) !important;
  border-radius: 24px !important;
  min-height: 188px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  text-align: center !important;
  color: #0f8f68 !important;
}
.single-green-file:hover,
.single-green-file [data-testid="file-upload"]:hover,
.single-green-file [data-testid="file"]:hover {
  border-color: #0f8f68 !important;
  background: linear-gradient(180deg, #ecfff6, #ffffff) !important;
  box-shadow: inset 0 0 0 1px rgba(15,143,104,.08), 0 12px 30px rgba(15,143,104,.08) !important;
}
.single-green-file .file-preview,
.single-green-file [data-testid="file"] {
  min-height: auto !important;
  justify-content: space-between !important;
  padding: 16px !important;
  background: #f8fffb !important;
  border: 1px solid rgba(15,143,104,.16) !important;
  border-radius: 18px !important;
}
.single-green-file button,
.single-green-file a,
.single-green-file span,
.single-green-file p,
.single-green-file svg {
  color: #0f8f68 !important;
}
.xray-open-chat-shell { margin-top: 18px !important; }
#xray-bottom-chat-btn,
.assistant-open-btn {
  background: linear-gradient(135deg, #0f8f68, #047857) !important;
  color: #ffffff !important;
  border: 0 !important;
  border-radius: 18px !important;
  min-height: 48px !important;
  box-shadow: 0 14px 34px rgba(15,143,104,.18) !important;
}
[role="alert"] svg { display: none !important; }
.xray-bottom-actions { margin-top: 12px !important; }

@media (max-width:900px){ .saas-two-col,.saas-widget-grid,.saas-findings-grid{ grid-template-columns:1fr!important; } .saas-result-header{ flex-direction:column; align-items:flex-start; } }
"""

CUSTOM_CSS += '\n/* =========================================================\n  FINAL SINGLE UPLOAD FIX\n  Keeps only the real Gradio upload box visible.\n  Removes fake duplicate drop areas from Lab + X-ray cards.\n  ========================================================= */\n.premium-drop-helper {\n  display: none !important;\n}\n\n/* Reset wrappers so they don\'t become a second upload panel */\n.single-green-file,\n.single-green-file .wrap,\n.single-green-file .block,\n.single-green-file > div {\n  background: transparent !important;\n  border: 0 !important;\n  box-shadow: none !important;\n  min-height: auto !important;\n}\n\n/* Style ONLY the actual Gradio file dropzone / upload label */\n.single-green-file [data-testid="file-upload"],\n.single-green-file [data-testid="file-upload-label"],\n.single-green-file label[data-testid="file-upload-label"],\n.single-green-file .file-preview,\n.single-green-file .file,\n.single-green-file .upload-container {\n  border: 2px dashed rgba(15, 143, 104, .38) !important;\n  background: linear-gradient(180deg, rgba(238, 251, 245, .92), #ffffff) !important;\n  border-radius: 24px !important;\n  min-height: 185px !important;\n  display: flex !important;\n  align-items: center !important;\n  justify-content: center !important;\n  text-align: center !important;\n  color: #0f8f68 !important;\n}\n\n/* Uploaded file row stays connected under same top upload card */\n.single-green-file [data-testid="file"] {\n  background: #f7fcfa !important;\n  border: 1px solid rgba(15,143,104,.18) !important;\n  border-radius: 16px !important;\n  margin-top: 12px !important;\n  color: #176b56 !important;\n}\n\n/* Strong green upload text/buttons */\n.single-green-file button,\n.single-green-file a,\n.single-green-file span,\n.single-green-file p {\n  color: #0f7f61 !important;\n}\n\n/* Make upload cards clean and single-panel */\n.upload-card {\n  overflow: hidden !important;\n}\n'

CUSTOM_CSS += '\n/* =========================================================\n  CLEAN GREEN UPLOAD CARD FIX\n  Removes ugly native "Choose file / No file chosen" look\n  Keeps one upload area only\n  ========================================================= */\n\n/* Upload card shell */\n.upload-card {\n  background: #ffffff !important;\n  border-radius: 28px !important;\n  border: 1px solid rgba(13, 116, 88, 0.12) !important;\n  box-shadow: 0 18px 45px rgba(18, 74, 61, 0.08) !important;\n  padding: 24px !important;\n  overflow: hidden !important;\n}\n\n/* Hide duplicate helper if it exists */\n.premium-drop-helper {\n  display: none !important;\n}\n\n/* Make Gradio file component become one clean upload panel */\n.single-green-file {\n  background: transparent !important;\n  border: none !important;\n  box-shadow: none !important;\n}\n\n/* Main file component wrapper */\n.single-green-file .wrap,\n.single-green-file .block,\n.single-green-file > div,\n.single-green-file div[data-testid="file"],\n.single-green-file div[data-testid="file-upload"] {\n  background: transparent !important;\n  border: none !important;\n  box-shadow: none !important;\n}\n\n/* The real clickable upload label / drop zone */\n.single-green-file label,\n.single-green-file label[data-testid="file-upload-label"],\n.single-green-file [data-testid="file-upload-label"] {\n  width: 100% !important;\n  min-height: 210px !important;\n  border: 2px dashed rgba(15, 143, 104, 0.36) !important;\n  border-radius: 24px !important;\n  background:\n    radial-gradient(circle at 50% 30%, rgba(22, 163, 127, 0.13), transparent 32%),\n    linear-gradient(180deg, rgba(240, 253, 248, 0.95), #ffffff) !important;\n  display: flex !important;\n  align-items: center !important;\n  justify-content: center !important;\n  text-align: center !important;\n  color: #08785d !important;\n  font-weight: 800 !important;\n  cursor: pointer !important;\n  transition: all .25s ease !important;\n}\n\n/* Hover effect */\n.single-green-file label:hover,\n.single-green-file label[data-testid="file-upload-label"]:hover,\n.single-green-file [data-testid="file-upload-label"]:hover {\n  border-color: #0f8f68 !important;\n  box-shadow: 0 18px 35px rgba(15, 143, 104, 0.12) !important;\n  transform: translateY(-2px) !important;\n}\n\n/* Hide native browser file input text */\n.single-green-file input[type="file"] {\n  opacity: 0 !important;\n  width: 0.1px !important;\n  height: 0.1px !important;\n  position: absolute !important;\n  overflow: hidden !important;\n  z-index: -1 !important;\n}\n\n/* Hide any visible native button if browser exposes it */\n.single-green-file input::file-selector-button {\n  display: none !important;\n}\n\n/* Remove small default file button look */\n.single-green-file button[aria-label="Upload"],\n.single-green-file .file-preview button,\n.single-green-file .upload-button {\n  background: #e8f8f1 !important;\n  color: #08785d !important;\n  border: 1px solid rgba(15, 143, 104, 0.22) !important;\n  border-radius: 12px !important;\n}\n\n/* Uploaded file row */\n.single-green-file .file-preview,\n.single-green-file [data-testid="file-preview"],\n.single-green-file .file {\n  margin-top: 14px !important;\n  background: #f8fffb !important;\n  border: 1px solid rgba(15, 143, 104, 0.16) !important;\n  border-radius: 16px !important;\n  padding: 12px 14px !important;\n  color: #176b56 !important;\n}\n\n/* Strong green text */\n.single-green-file span,\n.single-green-file p,\n.single-green-file a {\n  color: #08785d !important;\n}\n\n/* Fix oversized inner empty box */\n.single-green-file .empty,\n.single-green-file .upload-container {\n  min-height: 0 !important;\n}\n\n/* Optional cloud icon feeling for supported Gradio text */\n.single-green-file label::before,\n.single-green-file [data-testid="file-upload-label"]::before {\n  content: "";\n  width: 58px;\n  height: 58px;\n  border-radius: 50%;\n  background: #dcf8ec;\n  color: #078f69;\n  display: inline-flex;\n  align-items: center;\n  justify-content: center;\n  font-size: 30px;\n  margin-right: 16px;\n  box-shadow: 0 10px 28px rgba(15, 143, 104, 0.14);\n}\n'

CUSTOM_CSS += '\n/* =========================================================\n  POLISHED LAB + XRAY UPLOAD LAYOUT FIX\n  Goal: clean single upload card, no cutting, no overflow.\n  Keeps Gradio upload logic untouched.\n  ========================================================= */\n\n/* Upload card should be tall enough and never clip content */\n.upload-card {\n  background: #ffffff !important;\n  border-radius: 28px !important;\n  border: 1px solid rgba(13, 116, 88, 0.12) !important;\n  box-shadow: 0 18px 45px rgba(18, 74, 61, 0.08) !important;\n  padding: 26px 28px 28px 28px !important;\n  overflow: visible !important;\n  min-height: 330px !important;\n}\n\n/* Title spacing */\n.premium-upload-title,\n.upload-title {\n  font-size: 22px !important;\n  font-weight: 900 !important;\n  color: #17362e !important;\n  margin-bottom: 18px !important;\n}\n\n/* Remove fake duplicate helper if present */\n.premium-drop-helper {\n  display: none !important;\n}\n\n/* Keep upload component full width and stable */\n.single-green-file,\n.lab-single-green-file,\n.xray-single-green-file {\n  width: 100% !important;\n  max-width: 100% !important;\n  overflow: visible !important;\n  background: transparent !important;\n  border: 0 !important;\n  box-shadow: none !important;\n}\n\n/* Prevent Gradio internal wrappers from clipping */\n.single-green-file *,\n.lab-single-green-file *,\n.xray-single-green-file * {\n  box-sizing: border-box !important;\n}\n\n.single-green-file .wrap,\n.single-green-file .block,\n.single-green-file > div,\n.single-green-file div[data-testid="file"],\n.single-green-file div[data-testid="file-upload"] {\n  max-width: 100% !important;\n  overflow: visible !important;\n  background: transparent !important;\n  border: 0 !important;\n  box-shadow: none !important;\n}\n\n/* Main visible upload area */\n.single-green-file label,\n.single-green-file label[data-testid="file-upload-label"],\n.single-green-file [data-testid="file-upload-label"] {\n  width: 100% !important;\n  min-height: 190px !important;\n  border: 2px dashed rgba(15, 143, 104, 0.38) !important;\n  border-radius: 24px !important;\n  background:\n    radial-gradient(circle at 50% 30%, rgba(22, 163, 127, 0.14), transparent 30%),\n    linear-gradient(180deg, rgba(240, 253, 248, 0.97), #ffffff) !important;\n  display: flex !important;\n  flex-direction: column !important;\n  gap: 8px !important;\n  align-items: center !important;\n  justify-content: center !important;\n  text-align: center !important;\n  color: #08785d !important;\n  font-weight: 800 !important;\n  cursor: pointer !important;\n  padding: 30px 18px !important;\n  transition: all .25s ease !important;\n  overflow: hidden !important;\n}\n\n/* Hide native browser input */\n.single-green-file input[type="file"] {\n  opacity: 0 !important;\n  width: 0.1px !important;\n  height: 0.1px !important;\n  position: absolute !important;\n  overflow: hidden !important;\n  z-index: -1 !important;\n}\n.single-green-file input::file-selector-button {\n  display: none !important;\n}\n\n/* Hover polish */\n.single-green-file label:hover,\n.single-green-file label[data-testid="file-upload-label"]:hover,\n.single-green-file [data-testid="file-upload-label"]:hover {\n  border-color: #0f8f68 !important;\n  box-shadow: 0 18px 35px rgba(15, 143, 104, 0.12) !important;\n  transform: translateY(-2px) !important;\n}\n\n/* Cloud icon */\n.single-green-file label::before,\n.single-green-file [data-testid="file-upload-label"]::before {\n  content: "";\n  width: 58px !important;\n  height: 58px !important;\n  min-width: 58px !important;\n  border-radius: 999px !important;\n  background: #dcf8ec !important;\n  color: #078f69 !important;\n  display: inline-flex !important;\n  align-items: center !important;\n  justify-content: center !important;\n  font-size: 30px !important;\n  margin: 0 0 8px 0 !important;\n  box-shadow: 0 10px 28px rgba(15, 143, 104, 0.14) !important;\n}\n\n/* Uploaded file preview must sit below the drop area, full width, not side-cut */\n.single-green-file .file-preview,\n.single-green-file [data-testid="file-preview"],\n.single-green-file .file,\n.single-green-file [data-testid="file"],\n.single-green-file .file-preview-holder {\n  width: 100% !important;\n  max-width: 100% !important;\n  margin-top: 14px !important;\n  background: #f8fffb !important;\n  border: 1px solid rgba(15, 143, 104, 0.18) !important;\n  border-radius: 16px !important;\n  padding: 12px 14px !important;\n  color: #176b56 !important;\n  overflow: hidden !important;\n  white-space: normal !important;\n}\n\n/* File name row: no horizontal scroll / no right cut */\n.single-green-file .file-preview *,\n.single-green-file [data-testid="file-preview"] *,\n.single-green-file .file * {\n  max-width: 100% !important;\n  overflow: hidden !important;\n  text-overflow: ellipsis !important;\n}\n\n/* Disable horizontal scrollbars inside file component */\n.single-green-file ::-webkit-scrollbar {\n  height: 0px !important;\n}\n\n/* Make small "File" button green/mint and not dominant */\n.single-green-file button,\n.single-green-file .upload-button,\n.single-green-file button[aria-label="Upload"] {\n  background: #e8f8f1 !important;\n  color: #08785d !important;\n  border: 1px solid rgba(15, 143, 104, 0.22) !important;\n  border-radius: 12px !important;\n  font-weight: 700 !important;\n}\n\n/* Strong green text */\n.single-green-file span,\n.single-green-file p,\n.single-green-file a {\n  color: #08785d !important;\n}\n\n/* Remove weird extra boxes caused by gradio default file wrapper */\n.single-green-file .empty,\n.single-green-file .upload-container {\n  min-height: auto !important;\n  height: auto !important;\n}\n\n/* Responsive */\n@media (max-width: 768px) {\n  .upload-card {\n    padding: 20px !important;\n    min-height: 300px !important;\n  }\n  .single-green-file label,\n  .single-green-file [data-testid="file-upload-label"] {\n    min-height: 170px !important;\n  }\n}\n'

CUSTOM_CSS += '\n/* =========================================================\n  XRAY PREMIUM CHATBOT UI - open like Lab Report\n  ========================================================= */\n#xray-bottom-chat-btn,\n.assistant-open-btn {\n  display: none !important;\n}\n\n.xray-open-chat-shell {\n  margin-top: 18px !important;\n  margin-bottom: 12px !important;\n  background: linear-gradient(180deg, #f2fffa 0%, #ffffff 100%) !important;\n  border: 1px solid rgba(15,143,104,.16) !important;\n  border-radius: 28px !important;\n  box-shadow: 0 14px 34px rgba(18,74,61,.08) !important;\n}\n\n.xray-chatbot-box,\n.chatbot-box {\n  border-radius: 26px !important;\n  background: #ffffff !important;\n  border: 1px solid rgba(15,143,104,.13) !important;\n  box-shadow: 0 13px 32px rgba(18,74,61,.07) !important;\n  overflow: hidden !important;\n}\n\n.xray-chat-input-row {\n  margin-top: 10px !important;\n  align-items: stretch !important;\n}\n\n.xray-chat-input-row textarea {\n  min-height: 54px !important;\n  border-radius: 18px !important;\n  border: 1px solid rgba(15,143,104,.25) !important;\n  background: #ffffff !important;\n  box-shadow: inset 0 1px 0 rgba(15,143,104,.05) !important;\n}\n\n#xray-ask-btn {\n  background: linear-gradient(135deg, #0b8f68, #078457) !important;\n  color: #ffffff !important;\n  border: none !important;\n  border-radius: 18px !important;\n  font-size: 18px !important;\n  font-weight: 900 !important;\n  box-shadow: 0 12px 26px rgba(15,143,104,.24) !important;\n  transition: all .22s ease !important;\n}\n\n#xray-ask-btn:hover {\n  transform: translateY(-2px) !important;\n  box-shadow: 0 16px 32px rgba(15,143,104,.28) !important;\n}\n\n.xray-bottom-actions {\n  margin-top: 12px !important;\n}\n'
CUSTOM_CSS += '\n/* =========================================================\n  VISIBLE REACTBITS EFFECTS - REAL MEDIBUDDY CLASSES\n  Directly targets actual notebook classes:\n  soft-card, saas-card, saas-widget, xray-preview-center, chat-main, upload-card.\n  ========================================================= */\n\n/* Global premium card glow */\n.soft-card,\n.saas-card,\n.saas-widget,\n.upload-card,\n.chat-main,\n.chat-history-card,\n.xray-preview-center,\n.assistant-dock,\n.premium-chat-shell,\n.xray-open-chat-shell,\n.health-card,\n.ai-card,\n.suggestion-card,\n.disclaimer-card,\n.info-card,\n.table-card {\n  position: relative !important;\n  overflow: visible !important;\n  border-radius: 26px !important;\n  border: 1px solid rgba(15, 143, 104, 0.16) !important;\n  background: linear-gradient(180deg, #ffffff 0%, #f6fffb 100%) !important;\n  box-shadow:\n    0 18px 46px rgba(18, 74, 61, 0.09),\n    inset 0 1px 0 rgba(255,255,255,0.95) !important;\n  transition:\n    transform .25s ease,\n    box-shadow .25s ease,\n    border-color .25s ease !important;\n}\n\n/* Animated BorderGlow */\n.soft-card::before,\n.saas-card::before,\n.saas-widget::before,\n.upload-card::before,\n.chat-main::before,\n.chat-history-card::before,\n.xray-preview-center::before,\n.assistant-dock::before,\n.premium-chat-shell::before,\n.xray-open-chat-shell::before,\n.health-card::before,\n.ai-card::before,\n.suggestion-card::before,\n.disclaimer-card::before,\n.info-card::before,\n.table-card::before {\n  content: "";\n  position: absolute;\n  inset: -2px;\n  border-radius: inherit;\n  padding: 2px;\n  background:\n    linear-gradient(120deg,\n      rgba(15,143,104,.10),\n      rgba(38,197,133,.65),\n      rgba(202,255,230,.70),\n      rgba(15,143,104,.16));\n  -webkit-mask:\n    linear-gradient(#fff 0 0) content-box,\n    linear-gradient(#fff 0 0);\n  -webkit-mask-composite: xor;\n  mask-composite: exclude;\n  pointer-events: none;\n  z-index: 0;\n  opacity: .55;\n  animation: medibuddyBorderGlow 5.5s linear infinite;\n}\n\n.soft-card > *,\n.saas-card > *,\n.saas-widget > *,\n.upload-card > *,\n.chat-main > *,\n.chat-history-card > *,\n.xray-preview-center > *,\n.assistant-dock > *,\n.premium-chat-shell > *,\n.xray-open-chat-shell > *,\n.health-card > *,\n.ai-card > *,\n.suggestion-card > *,\n.disclaimer-card > *,\n.info-card > *,\n.table-card > * {\n  position: relative !important;\n  z-index: 1 !important;\n}\n\n@keyframes medibuddyBorderGlow {\n  0% { opacity: .30; filter: hue-rotate(0deg); }\n  45% { opacity: .82; }\n  100% { opacity: .35; filter: hue-rotate(18deg); }\n}\n\n/* MagicBento hover feel */\n.soft-card:hover,\n.saas-card:hover,\n.saas-widget:hover,\n.upload-card:hover,\n.chat-main:hover,\n.chat-history-card:hover,\n.xray-preview-center:hover,\n.assistant-dock:hover,\n.premium-chat-shell:hover,\n.xray-open-chat-shell:hover {\n  transform: translateY(-4px) !important;\n  border-color: rgba(15,143,104,.34) !important;\n  box-shadow:\n    0 24px 60px rgba(18, 74, 61, 0.14),\n    0 0 32px rgba(15, 143, 104, 0.11) !important;\n}\n\n/* Warm glow only for attention / abnormal cards */\n.need-card,\n.abnormal-card,\n.alert-card,\n.needs-attention,\n.warning-card {\n  position: relative !important;\n  border-color: rgba(245, 158, 11, .28) !important;\n}\n.need-card::before,\n.abnormal-card::before,\n.alert-card::before,\n.needs-attention::before,\n.warning-card::before {\n  content:"";\n  position:absolute;\n  inset:-2px;\n  border-radius:inherit;\n  padding:2px;\n  background:linear-gradient(120deg,rgba(15,143,104,.12),rgba(245,158,11,.55),rgba(255,237,213,.78),rgba(15,143,104,.14));\n  -webkit-mask:linear-gradient(#fff 0 0) content-box,linear-gradient(#fff 0 0);\n  -webkit-mask-composite:xor;\n  mask-composite:exclude;\n  pointer-events:none;\n  animation:medibuddyBorderGlow 5.5s linear infinite;\n}\n\n/* X-ray preview should visibly glow */\n.xray-preview-center {\n  max-width: 850px !important;\n  margin-left: auto !important;\n  margin-right: auto !important;\n  padding: 26px !important;\n}\n.xray-preview-center::after {\n  content: "";\n  position: absolute;\n  inset: -16px;\n  border-radius: inherit;\n  background: radial-gradient(circle at 50% 50%, rgba(15,143,104,.23), transparent 66%);\n  filter: blur(18px);\n  opacity: .48;\n  pointer-events: none;\n  z-index: -1;\n}\n\n/* Chatbot visible premium glow */\n.chat-main,\n.chatbot-box,\n.xray-chatbot-box {\n  border-radius: 28px !important;\n  background: linear-gradient(180deg, #ffffff, #f7fffb) !important;\n}\n\n/* TextType look */\n.typewriter-line,\n.medibuddy-type,\n.ai-type-label {\n  color: #08785d !important;\n  font-weight: 900 !important;\n  display: inline-block !important;\n}\n.typewriter-line::after,\n.medibuddy-type::after,\n.ai-type-label::after {\n  content: "|";\n  margin-left: 4px;\n  color: #0fa878;\n  animation: medibuddyBlink .75s steps(1) infinite;\n}\n@keyframes medibuddyBlink { 0%,50%{opacity:1} 51%,100%{opacity:0} }\n\n/* Text pop/jump */\n.jump-text,\n.saas-card-title em,\n.assistant-chip,\n.quick-q {\n  animation: medibuddyPop .95s ease both;\n}\n@keyframes medibuddyPop {\n  0% { opacity: 0; transform: translateY(8px) scale(.96); }\n  65% { opacity: 1; transform: translateY(-2px) scale(1.04); }\n  100% { opacity: 1; transform: translateY(0) scale(1); }\n}\n\n/* ImageTrail-style welcome background accent */\n.welcome-hero,\n.welcome-card,\n.hero-card,\n.hero-section {\n  position: relative !important;\n  overflow: hidden !important;\n}\n.welcome-hero::after,\n.welcome-card::after,\n.hero-card::after,\n.hero-section::after {\n  content: "";\n  position: absolute;\n  width: 450px;\n  height: 450px;\n  right: -160px;\n  top: -180px;\n  border-radius: 999px;\n  background: radial-gradient(circle, rgba(15,143,104,.22), rgba(15,143,104,.08) 34%, transparent 70%);\n  animation: medibuddyTrail 8s ease-in-out infinite;\n  pointer-events: none;\n}\n@keyframes medibuddyTrail {\n  0%, 100% { transform: translate3d(0,0,0) scale(1); }\n  50% { transform: translate3d(26px,18px,0) scale(1.08); }\n}\n\n/* Upload card clearer glow */\n.single-green-file label,\n.single-green-file [data-testid="file-upload-label"] {\n  box-shadow:\n    inset 0 0 0 1px rgba(15,143,104,.08),\n    0 12px 30px rgba(15,143,104,.08) !important;\n}\n\n/* Bento layout polish on existing grids */\n.saas-widget-grid,\n.saas-two-col,\n.lab-grid,\n.result-grid {\n  gap: 18px !important;\n}\n\n@media (prefers-reduced-motion: reduce) {\n  .soft-card::before,\n  .saas-card::before,\n  .saas-widget::before,\n  .upload-card::before,\n  .chat-main::before,\n  .xray-preview-center::before,\n  .welcome-card::after,\n  .welcome-hero::after,\n  .jump-text {\n    animation: none !important;\n  }\n}\n'


CUSTOM_CSS += '\n/* =========================================================\n  FINAL LAB UPLOAD UI POLISH\n  - File preview stays below the upload area\n  - No oversized side preview box\n  - Download report area/button is clearer\n  - Remove/Re-upload is controlled by Gradio visibility\n  ========================================================= */\n\n.lab-single-green-file,\n.lab-single-green-file .wrap,\n.lab-single-green-file .block,\n.lab-single-green-file > div,\n.lab-single-green-file [data-testid="file"] {\n  width: 100% !important;\n  max-width: 100% !important;\n}\n\n.lab-single-green-file [data-testid="file"],\n.lab-single-green-file .file,\n.lab-single-green-file .file-preview,\n.lab-single-green-file [data-testid="file-preview"] {\n  display: flex !important;\n  flex-direction: column !important;\n  align-items: stretch !important;\n  gap: 12px !important;\n}\n\n.lab-single-green-file label,\n.lab-single-green-file label[data-testid="file-upload-label"],\n.lab-single-green-file [data-testid="file-upload-label"] {\n  width: 100% !important;\n  min-height: 180px !important;\n  margin: 0 !important;\n}\n\n.lab-single-green-file .file-preview,\n.lab-single-green-file [data-testid="file-preview"],\n.lab-single-green-file .file-preview-holder,\n.lab-single-green-file [data-testid="file"] > div:not(:first-child) {\n  width: 100% !important;\n  max-width: 100% !important;\n  margin: 10px 0 0 0 !important;\n  border-radius: 16px !important;\n  background: #f8fffb !important;\n  border: 1px solid rgba(15,143,104,.20) !important;\n  padding: 12px 14px !important;\n}\n\n.lab-single-green-file .file-preview *,\n.lab-single-green-file [data-testid="file-preview"] *,\n.lab-single-green-file [data-testid="file"] * {\n  max-width: 100% !important;\n  white-space: normal !important;\n  overflow-wrap: anywhere !important;\n}\n\n/* Reduce unwanted empty/grey vertical space under upload card */\n.lab-single-green-file .empty,\n.lab-single-green-file .upload-container,\n.lab-single-green-file [data-testid="file-upload"] {\n  min-height: auto !important;\n}\n\n/* Make the generated report download look like an intentional action */\n#pdf-download label,\n#pdf-download .label-wrap,\n#pdf-download .file-label {\n  font-weight: 900 !important;\n  color: #123f35 !important;\n}\n\n#pdf-download .file,\n#pdf-download .file-preview,\n#pdf-download [data-testid="file"],\n#pdf-download [data-testid="file-preview"] {\n  border: 1px solid rgba(15,143,104,.24) !important;\n  border-radius: 14px !important;\n  background: linear-gradient(180deg, #ffffff, #f5fffb) !important;\n  padding: 10px 12px !important;\n}\n\n#pdf-download a,\n#pdf-download button,\n#pdf-download [role="button"] {\n  font-size: 15px !important;\n  font-weight: 900 !important;\n  color: #08785d !important;\n}\n\n#pdf-download svg {\n  width: 22px !important;\n  height: 22px !important;\n  color: #08785d !important;\n  stroke-width: 2.8 !important;\n}\n'



CUSTOM_CSS += '\n/* =========================================================\n  FINAL REQUEST: CLEAN LAB UPLOAD UI ONLY\n  - No empty grey space before upload\n  - Hide generated report download until analysis creates it\n  - Hide native file preview card; show filename in one inline row\n  ========================================================= */\n\n/* Keep the lab upload card compact before a file is uploaded */\n.lab-single-green-file label,\n.lab-single-green-file label[data-testid="file-upload-label"],\n.lab-single-green-file [data-testid="file-upload-label"] {\n  min-height: 150px !important;\n}\n\n/* The lab upload card should not reserve a large blank area */\n.upload-card:has(.lab-single-green-file) {\n  min-height: auto !important;\n  padding-bottom: 22px !important;\n}\n\n/* Hide Gradio\'s oversized native uploaded-file preview for this one upload box only.\n  The actual file value remains unchanged; we show a cleaner filename row below. */\n.lab-single-green-file .file-preview,\n.lab-single-green-file [data-testid="file-preview"],\n.lab-single-green-file .file-preview-holder,\n.lab-single-green-file [data-testid="file"] > div:not(:first-child) {\n  display: none !important;\n}\n\n.uploaded-file-inline-box {\n  margin-top: 14px;\n  width: 100%;\n  display: grid;\n  grid-template-columns: 42px 1fr;\n  align-items: center;\n  gap: 12px;\n  padding: 12px 14px;\n  border-radius: 16px;\n  border: 1px solid rgba(15,143,104,.22);\n  background: linear-gradient(180deg, #ffffff, #f5fffb);\n  color: #143f35;\n  box-shadow: 0 8px 22px rgba(15,143,104,.07);\n}\n.uploaded-file-inline-box .uploaded-file-icon {\n  width: 42px;\n  height: 42px;\n  border-radius: 14px;\n  display: grid;\n  place-items: center;\n  background: #e7f8f1;\n  color: #08785d;\n  font-size: 20px;\n}\n.uploaded-file-inline-box b {\n  display: block;\n  color: #143f35;\n  font-size: 14px;\n  line-height: 1.25;\n  overflow-wrap: anywhere;\n}\n.uploaded-file-inline-box small {\n  display: block;\n  margin-top: 3px;\n  color: #5b756b;\n  font-weight: 800;\n}\n\n/* Download report should look like an action, not a blank file card */\n#pdf-download {\n  margin-top: 12px !important;\n}\n#pdf-download label,\n#pdf-download .label-wrap,\n#pdf-download .file-label {\n  font-weight: 950 !important;\n  color: #123f35 !important;\n  font-size: 15px !important;\n}\n#pdf-download .file,\n#pdf-download .file-preview,\n#pdf-download [data-testid="file"],\n#pdf-download [data-testid="file-preview"] {\n  min-height: auto !important;\n  border: 1px solid rgba(15,143,104,.28) !important;\n  background: #f8fffb !important;\n  border-radius: 14px !important;\n}\n#pdf-download a,\n#pdf-download button,\n#pdf-download [role="button"] {\n  font-size: 15px !important;\n  font-weight: 950 !important;\n  color: #08785d !important;\n}\n#pdf-download svg {\n  width: 24px !important;\n  height: 24px !important;\n  color: #08785d !important;\n  stroke-width: 3 !important;\n}\n#lab-clear-file-btn {\n  margin-top: 12px !important;\n}\n'


CUSTOM_CSS += r'''

/* === MEDIBUDDY FINAL LAB UPLOAD UI PATCH === */
/* Keep Lab upload compact and remove the extra uploaded-file preview card. */
#lab-upload-file {
  min-height: 0 !important;
}
#lab-upload-file .wrap,
#lab-upload-file .block,
#lab-upload-file [data-testid="file"],
#lab-upload-file .upload-container {
  min-height: 160px !important;
  height: auto !important;
  padding: 14px !important;
  border-radius: 22px !important;
}
/* Hide Gradio's separate internal selected-file preview/card; our clean file row below shows the filename. */
#lab-upload-file .file-preview,
#lab-upload-file [data-testid="file-preview"],
#lab-upload-file .file-preview-wrapper,
#lab-upload-file .file-preview-holder,
#lab-upload-file .file-preview-container,
#lab-upload-file .file-preview-row,
#lab-upload-file .file-item,
#lab-upload-file .uploaded-file,
#lab-upload-file .upload-file-list,
#lab-upload-file .file-list,
#lab-upload-file ul,
#lab-upload-file li {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  min-height: 0 !important;
  max-height: 0 !important;
  width: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}
.uploaded-file-inline {
  margin-top: 12px !important;
}
.uploaded-file-inline-box {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid rgba(15,143,104,.22);
  background: linear-gradient(135deg,#ffffff,#f2fff9);
  box-shadow: 0 8px 22px rgba(15,143,104,.08);
}
.uploaded-file-inline-box b {
  color: #143f35 !important;
  font-size: 14px !important;
  font-weight: 950 !important;
}
.uploaded-file-inline-box small {
  color: #5b756b !important;
  font-weight: 800 !important;
}
/* File output appears after analysis and should look like a clear download action. */
#pdf-download {
  margin-top: 12px !important;
}
#pdf-download label,
#pdf-download .label-wrap,
#pdf-download .file-label {
  font-size: 15px !important;
  font-weight: 950 !important;
  color: #123f35 !important;
}
#pdf-download a,
#pdf-download button,
#pdf-download [role="button"] {
  font-size: 15px !important;
  font-weight: 950 !important;
  color: #08785d !important;
}
#pdf-download svg {
  width: 24px !important;
  height: 24px !important;
  stroke-width: 3 !important;
}
#lab-clear-file-btn {
  margin-top: 12px !important;
}

'''

CUSTOM_CSS += '\n/* =========================================================\n  REFERENCE-STYLE UPLOAD UI FOR LAB + XRAY\n  Applies the same clean design to both upload pages.\n  ========================================================= */\n.ref-upload-head { display:flex; align-items:center; gap:14px; margin-bottom:18px; }\n.upload-head-icon { width:52px; height:52px; display:grid; place-items:center; border-radius:16px; background:#eafaf3; color:#08785d; font-size:25px; box-shadow:0 10px 24px rgba(15,143,104,.08); }\n.ref-upload-head .premium-upload-title, .ref-upload-head .upload-title { margin:0 0 4px 0 !important; font-size:24px !important; line-height:1.1 !important; }\n.upload-subtitle { color:#65748a; font-weight:850; font-size:15px; line-height:1.35; }\n.upload-secure-strip { margin-top:18px; display:grid; grid-template-columns:44px 1fr; gap:12px; align-items:center; border-radius:18px; border:1px solid rgba(15,143,104,.10); background:linear-gradient(135deg,#f4fff9,#ffffff); padding:14px 16px; }\n.upload-secure-strip span { width:42px; height:42px; border-radius:14px; display:grid; place-items:center; background:#dcf8ec; color:#078f69; font-size:22px; }\n.upload-secure-strip b { display:block; color:#162033; font-size:14px; font-weight:950; }\n.upload-secure-strip small { display:block; color:#65748a; font-size:13px; font-weight:800; margin-top:3px; }\n.action-panel-ref { text-align:center; padding:4px 4px 18px; }\n.action-title-ref { text-align:left; font-size:24px; font-weight:950; color:#17203a; margin-bottom:20px; }\n.action-illustration-ref { width:116px; height:116px; margin:0 auto 18px; border-radius:24px; display:grid; place-items:center; background:radial-gradient(circle at 70% 20%,rgba(20,184,166,.18),transparent 25%),linear-gradient(135deg,#e8f8f1,#ffffff); font-size:52px; box-shadow:0 16px 36px rgba(15,143,104,.10); }\n.action-panel-ref h3 { margin:0 0 8px; color:#17203a; font-size:22px; font-weight:950; }\n.action-panel-ref p { margin:0 auto; color:#66748a; font-size:15px; line-height:1.5; font-weight:850; max-width:260px; }\n#lab-analyze-btn, #xray-analyze-btn { min-height:54px !important; font-size:16px !important; margin-top:8px !important; border-radius:15px !important; }\n#pdf-download, #xray-pdf-download { margin-top:14px !important; }\n#pdf-download label, #xray-pdf-download label, #pdf-download .label-wrap, #xray-pdf-download .label-wrap, #pdf-download .file-label, #xray-pdf-download .file-label { font-size:15px !important; font-weight:950 !important; color:#123f35 !important; }\n#pdf-download a, #xray-pdf-download a, #pdf-download button, #xray-pdf-download button, #pdf-download [role="button"], #xray-pdf-download [role="button"] { font-size:15px !important; font-weight:950 !important; color:#08785d !important; }\n#pdf-download svg, #xray-pdf-download svg { width:24px !important; height:24px !important; stroke-width:3 !important; }\n#lab-clear-file-btn, #xray-clear-file-btn { margin-top:12px !important; min-height:46px !important; font-size:15px !important; }\n#lab-upload-file .file-preview, #xray-upload-file .file-preview, #lab-upload-file [data-testid="file-preview"], #xray-upload-file [data-testid="file-preview"], #lab-upload-file .file-preview-wrapper, #xray-upload-file .file-preview-wrapper, #lab-upload-file .file-preview-holder, #xray-upload-file .file-preview-holder, #lab-upload-file .file-item, #xray-upload-file .file-item, #lab-upload-file ul, #xray-upload-file ul, #lab-upload-file li, #xray-upload-file li { display:none !important; visibility:hidden !important; height:0 !important; min-height:0 !important; max-height:0 !important; margin:0 !important; padding:0 !important; overflow:hidden !important; }\n#lab-upload-file .wrap, #xray-upload-file .wrap, #lab-upload-file .block, #xray-upload-file .block, #lab-upload-file [data-testid="file"], #xray-upload-file [data-testid="file"], #lab-upload-file .upload-container, #xray-upload-file .upload-container { min-height:170px !important; height:auto !important; }\n.uploaded-file-inline-box .uploaded-file-icon { flex:0 0 auto; }\n'



CUSTOM_CSS += """
/* =========================================================
  FINISH BUTTON NAVIGATION
  Finish returns the user directly to Page 1 Welcome.
  ========================================================= */
.finish-btn,
#choose-finish-btn,
#lab-finish-btn,
#xray-finish-btn,
#chat-finish-btn,
#comparison-finish-btn {
  background: linear-gradient(135deg, #0f8f68, #075f49) !important;
  color: #ffffff !important;
  border: 1px solid rgba(15, 143, 104, .22) !important;
  border-radius: 16px !important;
  font-weight: 950 !important;
  box-shadow: 0 12px 30px rgba(15, 143, 104, .18) !important;
}
.finish-btn:hover,
#choose-finish-btn:hover,
#lab-finish-btn:hover,
#xray-finish-btn:hover,
#chat-finish-btn:hover,
#comparison-finish-btn:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 16px 35px rgba(15, 143, 104, .24) !important;
}
.bottom-action-row {
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)) !important;
}
.page-action-row,
.choose-back-row-v2 {
  gap: 12px !important;
  align-items: center !important;
}
#choose-finish-btn {
  max-width: 120px !important;
  min-height: 34px !important;
  padding: 6px 14px !important;
  border-radius: 999px !important;
  text-align: center !important;
}
/* Keep Comparison Dashboard navigation at the bottom like Lab/X-ray pages */
.comparison-bottom-actions {
  margin-top: 22px !important;
  margin-bottom: 8px !important;
}
.comparison-bottom-actions button {
  min-height: 48px !important;
}

"""



CUSTOM_CSS += """
/* =========================================================
  FINAL SUPERVISOR POLISH
  - Smaller navigation buttons
  - Left / right placement with clean middle spacing
  - Clear developer credit at the end
  ========================================================= */
.bottom-action-row,
.comparison-bottom-actions,
.xray-bottom-actions {
  display: flex !important;
  grid-template-columns: none !important;
  justify-content: space-between !important;
  align-items: center !important;
  gap: clamp(34px, 18vw, 360px) !important;
  width: 100% !important;
  max-width: 960px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding: 4px 10px !important;
}

.bottom-action-row > *,
.bottom-action-row > div,
.bottom-action-row .form,
.comparison-bottom-actions > *,
.comparison-bottom-actions > div,
.comparison-bottom-actions .form,
.xray-bottom-actions > *,
.xray-bottom-actions > div,
.xray-bottom-actions .form {
  flex: 0 0 auto !important;
  width: auto !important;
  max-width: max-content !important;
  min-width: 0 !important;
}

#lab-bottom-back-btn,
#lab-bottom-back-btn button,
#lab-finish-btn,
#lab-finish-btn button,
#xray-bottom-back-btn,
#xray-bottom-back-btn button,
#xray-finish-btn,
#xray-finish-btn button,
#comparison-back-btn,
#comparison-back-btn button,
#comparison-finish-btn,
#comparison-finish-btn button,
#chat-back-btn,
#chat-back-btn button,
#chat-finish-btn,
#chat-finish-btn button {
  width: 180px !important;
  max-width: 180px !important;
  min-width: 155px !important;
  min-height: 42px !important;
  padding: 8px 16px !important;
  font-size: 14px !important;
  border-radius: 14px !important;
  line-height: 1.1 !important;
  flex: 0 0 180px !important;
}

#xray-bottom-chat-btn,
#xray-bottom-chat-btn button {
  width: 290px !important;
  max-width: 290px !important;
  min-height: 42px !important;
  font-size: 14px !important;
  border-radius: 14px !important;
}

.page-action-row {
  display: flex !important;
  justify-content: space-between !important;
  align-items: center !important;
  gap: clamp(28px, 16vw, 320px) !important;
  max-width: 960px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

.app-footer {
  margin: 18px auto 0 !important;
  padding: 12px 16px 6px !important;
  max-width: 980px !important;
  text-align: center !important;
  border-top: 1px solid rgba(15, 143, 104, .16) !important;
  color: #7a849c !important;
  font-weight: 850 !important;
  font-size: 12px !important;
}
.developer-credit {
  display: none !important;
}
.footer-subtext {
  margin-top: 0 !important;
  color: #7a849c !important;
  font-size: 12px !important;
  font-weight: 850 !important;
  line-height: 1.6 !important;
}

@media (max-width: 720px) {
  .bottom-action-row,
  .comparison-bottom-actions,
  .xray-bottom-actions,
  .page-action-row {
    gap: 12px !important;
    padding: 4px 0 !important;
  }
  #lab-bottom-back-btn,
  #lab-bottom-back-btn button,
  #lab-finish-btn,
  #lab-finish-btn button,
  #xray-bottom-back-btn,
  #xray-bottom-back-btn button,
  #xray-finish-btn,
  #xray-finish-btn button,
  #comparison-back-btn,
  #comparison-back-btn button,
  #comparison-choices-bottom-btn,
  #comparison-choices-bottom-btn button,
  #comparison-finish-btn,
  #comparison-finish-btn button,
  #chat-back-btn,
  #chat-back-btn button,
  #chat-finish-btn,
  #chat-finish-btn button {
    width: 150px !important;
    max-width: 150px !important;
    min-width: 135px !important;
    flex-basis: 150px !important;
    font-size: 13px !important;
  }
}

/* Clean comparison card button: keep arrow inline, no broken second-line arrow. */
#choice-compare-btn,
#choice-compare-btn button,
#choice-compare-btn span {
  white-space: nowrap !important;
}
#choice-compare-btn,
#choice-compare-btn button {
  min-height: 54px !important;
  border-radius: 14px !important;
  font-size: 14px !important;
  font-weight: 950 !important;
  padding-left: 12px !important;
  padding-right: 12px !important;
  border: none !important;
  box-shadow: 0 14px 24px rgba(33,92,76,.16) !important;
  background: #248b68 !important;
}


/* Move the original working Comparison Choices button to the bottom center. */
.comparison-choice-action-slot {
  flex: 0 0 190px !important;
  width: 190px !important;
  max-width: 190px !important;
  min-height: 42px !important;
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
}
.comparison-choice-action-slot:empty {
  visibility: hidden !important;
}
.comparison-choice-action-slot .comparison-moved-original-btn,
.comparison-choice-action-slot .comparison-moved-original-btn > div,
.comparison-choice-action-slot .comparison-moved-original-btn button {
  width: 190px !important;
  max-width: 190px !important;
  min-width: 165px !important;
  min-height: 42px !important;
  margin: 0 !important;
  padding: 0 !important;
}
.comparison-choice-action-slot .comparison-moved-original-btn button {
  padding: 8px 16px !important;
  border-radius: 14px !important;
  font-size: 14px !important;
  line-height: 1.1 !important;
  font-weight: 950 !important;
  white-space: nowrap !important;
}

/* Three-button professional layout for Comparison Dashboard only. */
.comparison-bottom-actions {
  max-width: 900px !important;
  gap: clamp(18px, 8vw, 120px) !important;
  justify-content: space-between !important;
}
#comparison-choices-bottom-btn,
#comparison-choices-bottom-btn button {
  width: 190px !important;
  max-width: 190px !important;
  min-width: 165px !important;
  min-height: 42px !important;
  padding: 8px 16px !important;
  font-size: 14px !important;
  border-radius: 14px !important;
  line-height: 1.1 !important;
  flex: 0 0 190px !important;
}



/* Original comparison dashboard back button is kept only as a hidden working action. */
.top-comparison-choices-hidden,
.top-comparison-choices-hidden *,
#original-comparison-choices-hidden,
#original-comparison-choices-hidden * {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  min-height: 0 !important;
  max-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  overflow: hidden !important;
}

/* Hide the original top comparison-choices button; the working action is proxied by the bottom middle button. */
[data-medibuddy-hidden-comparison-choices-wrap="1"],
[data-medibuddy-original-comparison-choices="1"] {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  min-height: 0 !important;
  max-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  overflow: hidden !important;
}

/* Bottom middle Comparison Choices button only on the comparison dashboard. */
#comparison-choices-bottom-btn,
#comparison-choices-bottom-btn button {
  width: 190px !important;
  max-width: 190px !important;
  min-width: 165px !important;
  min-height: 42px !important;
  padding: 8px 16px !important;
  font-size: 14px !important;
  border-radius: 14px !important;
  line-height: 1.1 !important;
  flex: 0 0 190px !important;
  white-space: nowrap !important;
}
#comparison-choices-bottom-btn button {
  background: #ffffff !important;
  color: #163b35 !important;
  border: 1px solid rgba(15,143,104,.28) !important;
  box-shadow: 0 12px 26px rgba(33,92,76,.10) !important;
  font-weight: 950 !important;
}
#comparison-choices-bottom-btn button:hover {
  background: #f4fffb !important;
  border-color: rgba(15,143,104,.45) !important;
}
"""



CUSTOM_CSS += """
/* =========================================================
  XRAY BODY REGION DROPDOWN DESIGN FIX ONLY
  Keeps dropdown logic exactly the same; only cleans the UI.
  ========================================================= */
#xray-region-input {
  margin-top: 12px !important;
  margin-bottom: 14px !important;
  position: relative !important;
  z-index: 60 !important;
  width: 100% !important;
}

#xray-region-input,
#xray-region-input * {
  box-sizing: border-box !important;
}

#xray-region-input label,
#xray-region-input .label-wrap,
#xray-region-input .label-wrap span {
  color: #08785d !important;
  font-size: 13px !important;
  font-weight: 950 !important;
  line-height: 1.2 !important;
  background: transparent !important;
  padding: 0 !important;
  margin-bottom: 6px !important;
}

#xray-region-input .info,
#xray-region-input .secondary-wrap,
#xray-region-input small {
  color: #6b7d76 !important;
  font-size: 11px !important;
  font-weight: 750 !important;
  line-height: 1.35 !important;
  margin-top: 2px !important;
}

#xray-region-input .wrap,
#xray-region-input .block,
#xray-region-input > div,
#xray-region-input [data-testid="dropdown"],
#xray-region-input [role="combobox"],
#xray-region-input input {
  width: 100% !important;
  max-width: 100% !important;
}

#xray-region-input [role="combobox"],
#xray-region-input input,
#xray-region-input .container,
#xray-region-input .wrap > div {
  min-height: 46px !important;
  border-radius: 14px !important;
  border: 1px solid rgba(15, 143, 104, .20) !important;
  background: #ffffff !important;
  color: #17362e !important;
  font-size: 14px !important;
  font-weight: 850 !important;
  box-shadow: 0 8px 22px rgba(15, 143, 104, .06) !important;
}

#xray-region-input [role="combobox"]:focus,
#xray-region-input input:focus,
#xray-region-input .container:focus-within {
  border-color: rgba(15, 143, 104, .45) !important;
  box-shadow: 0 0 0 3px rgba(15, 143, 104, .10) !important;
}

#xray-region-input svg,
#xray-region-input button {
  color: #08785d !important;
}

/* Open menu: keep it inside a neat rounded scroll panel instead of a long white box */
#xray-region-input .options,
#xray-region-input ul.options,
#xray-region-input [role="listbox"],
#xray-region-input .dropdown-options {
  max-height: 178px !important;
  overflow-y: auto !important;
  border-radius: 14px !important;
  border: 1px solid rgba(15, 143, 104, .22) !important;
  background: #ffffff !important;
  box-shadow: 0 18px 36px rgba(15, 143, 104, .14) !important;
  z-index: 99999 !important;
  padding: 6px !important;
}

#xray-region-input [role="option"],
#xray-region-input .options li,
#xray-region-input ul.options li {
  min-height: 36px !important;
  padding: 8px 12px !important;
  border-radius: 10px !important;
  color: #17362e !important;
  font-size: 13px !important;
  font-weight: 800 !important;
}

#xray-region-input [role="option"]:hover,
#xray-region-input .options li:hover,
#xray-region-input ul.options li:hover {
  background: #eefbf6 !important;
  color: #08785d !important;
}
"""

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
  gr.HTML("\n<script>\n(function(){\n if(window.__medibuddyVisibleTypeLoaded) return;\n window.__medibuddyVisibleTypeLoaded = true;\n\n function startType(el){\n  var raw = el.getAttribute('data-texts') || el.textContent || '';\n  var parts = raw.indexOf('||') >= 0 ? raw.split('||') : [raw];\n  var i=0, c=0, del=false, speed=55, pause=1400;\n  function tick(){\n   var s = parts[i] || '';\n   if(!del){\n    c++;\n    el.textContent = s.slice(0,c);\n    if(c >= s.length){\n     del = parts.length > 1;\n     setTimeout(tick, pause);\n     return;\n    }\n   } else {\n    c--;\n    el.textContent = s.slice(0,c);\n    if(c <= 0){\n     del = false;\n     i = (i+1) % parts.length;\n    }\n   }\n   setTimeout(tick, del ? 30 : speed);\n  }\n  el.textContent = '';\n  tick();\n }\n\n function init(){\n  document.querySelectorAll('.medibuddy-type[data-texts]:not([data-ready])').forEach(function(el){\n   el.setAttribute('data-ready','1');\n   startType(el);\n  });\n }\n new MutationObserver(init).observe(document.documentElement,{childList:true,subtree:true});\n document.addEventListener('DOMContentLoaded', init);\n setTimeout(init, 800);\n})();\n</script>\n")
  with gr.Group(elem_classes=["app-shell"]):

    gr.HTML("""
    <div class="top-nav">
      <div class="logo-wrap">
        <div class="logo-icon"></div>
        <div class="logo-text">MediBuddy <span>AI</span></div>
      </div>
    </div>
    """)

    gr.HTML("""
    <div id="page-loader">
      <div class="loader-card">
        <div class="loader-spinner"></div>
        <b>Loading next step...</b>
        <span>Please wait a moment</span>
      </div>
    </div>
    <script>
    window.medibuddyShowLoader = function(){
      const loader = document.getElementById('page-loader');
      if (loader) {
        loader.classList.add('is-visible');
        setTimeout(() => loader.classList.remove('is-visible'), 850);
      }
    };
    </script>
    <script>
    (function(){
      // Versioned installer so updated code still runs even if an older Gradio page is open.
      window.__medibuddyComparisonChoicesProxyVersion = 'v5-bottom-only';

      function normalizeText(el){
        return (el && el.textContent ? el.textContent : '').replace(/\s+/g, ' ').trim().toLowerCase();
      }

      function forceHide(el){
        if (!el) return;
        el.setAttribute('data-medibuddy-hidden-comparison-choices-wrap', '1');
        el.style.setProperty('display', 'none', 'important');
        el.style.setProperty('visibility', 'hidden', 'important');
        el.style.setProperty('height', '0px', 'important');
        el.style.setProperty('min-height', '0px', 'important');
        el.style.setProperty('max-height', '0px', 'important');
        el.style.setProperty('margin', '0px', 'important');
        el.style.setProperty('padding', '0px', 'important');
        el.style.setProperty('border', '0px', 'important');
        el.style.setProperty('overflow', 'hidden', 'important');
      }

      function isBottomArea(el){
        return !!(el && (el.closest('#comparison-choices-bottom-btn') || el.closest('.comparison-bottom-actions')));
      }

      function buttonLooksLikeTopComparisonChoices(btn){
        if (!btn || isBottomArea(btn)) return false;
        const text = normalizeText(btn);
        return text.includes('back to comparison choices') || text.includes('comparison choices');
      }

      function findOriginalComparisonChoicesButton(){
        const panel = document.querySelector('#comparison-dashboard-inner') || document;
        const buttons = Array.from(panel.querySelectorAll('button, [role="button"]'));
        return buttons.find(buttonLooksLikeTopComparisonChoices) || null;
      }

      function hideTopComparisonChoicesButtons(){
        const panel = document.querySelector('#comparison-dashboard-inner');
        if (!panel) return;

        const buttons = Array.from(panel.querySelectorAll('button, [role="button"]'));
        buttons.forEach(function(btn){
          if (!buttonLooksLikeTopComparisonChoices(btn)) return;
          btn.setAttribute('data-medibuddy-original-comparison-choices', '1');
          forceHide(btn);

          // Hide only the small top button wrapper. Stop before the main dashboard container.
          let node = btn.parentElement;
          let safety = 0;
          while (node && node !== panel && safety < 6) {
            const nodeText = normalizeText(node);
            const hasOnlyTopButtonText = nodeText.includes('back to comparison choices') && nodeText.length < 120;
            if (hasOnlyTopButtonText) {
              forceHide(node);
            }
            node = node.parentElement;
            safety += 1;
          }
        });

        // Extra fallback: sometimes Gradio places the label in a span/div outside the button.
        Array.from(panel.querySelectorAll('*')).forEach(function(el){
          if (isBottomArea(el)) return;
          const text = normalizeText(el);
          if (!text.includes('back to comparison choices')) return;
          if (text.length > 120) return;
          const possibleButton = el.matches('button, [role="button"]') ? el : (el.closest('button, [role="button"]') || el.querySelector('button, [role="button"]'));
          if (possibleButton && !isBottomArea(possibleButton)) {
            possibleButton.setAttribute('data-medibuddy-original-comparison-choices', '1');
            forceHide(possibleButton);
          }
          forceHide(el);
        });
      }

      window.medibuddyClickOriginalComparisonChoices = function(){
        const panel = document.querySelector('#comparison-dashboard-inner') || document;
        let original = panel.querySelector('#original-comparison-choices-hidden button, #original-comparison-choices-hidden, .top-comparison-choices-hidden button, .top-comparison-choices-hidden');
        if (!original) original = panel.querySelector('button[data-medibuddy-original-comparison-choices="1"], [role="button"][data-medibuddy-original-comparison-choices="1"]');
        if (!original) original = findOriginalComparisonChoicesButton();
        if (original) {
          original.click();
          return true;
        }
        return false;
      };

      function init(){
        hideTopComparisonChoicesButtons();
      }

      // Run repeatedly because Gradio can rebuild tab content after navigation.
      try { new MutationObserver(init).observe(document.documentElement, {childList: true, subtree: true, characterData: true}); } catch(e) {}
      document.addEventListener('DOMContentLoaded', init);
      document.addEventListener('click', function(event){
        const bottom = event.target.closest('#comparison-choices-bottom-btn button, #comparison-choices-bottom-btn');
        if (!bottom) return;
        const clicked = window.medibuddyClickOriginalComparisonChoices && window.medibuddyClickOriginalComparisonChoices();
        if (clicked) {
          event.preventDefault();
          event.stopPropagation();
        }
      }, true);
      [50, 200, 500, 1000, 1800, 3000].forEach(function(ms){ setTimeout(init, ms); });
      [4500, 6500].forEach(function(ms){ setTimeout(init, ms); });
    })();
    </script>
    """)

    with gr.Tabs(selected="welcome") as main_tabs:
      with gr.Tab("Page 1 Welcome", id="welcome"):
        gr.HTML("""
        <div class="welcome-animated">
         <div class="welcome-landing">
          <div class="welcome-mini-nav">
           <div class="welcome-brand"><span class="welcome-brand-mark"></span> MediBuddy AI</div>
           <div class="welcome-secure-tag"> Private &amp; Secure</div>
          </div>

          <div class="landing-hero">
           <div>
            <div class="hero-kicker"><span class="kicker-dot"></span>Your Personal Health Companion</div>
            <h1 class="landing-title">Understand your<br><span class="highlight">medical report</span><br>in simple words.</h1>
            <p class="landing-copy">Upload your lab results or medical documents. MediBuddy instantly translates complex medical jargon into clear, reassuring insights you can actually understand.</p>
            <div class="welcome-actions">
             <div class="welcome-action-chip"> 100% Educational</div>
             <div class="welcome-action-chip"> Private</div>
            </div>
           </div>

           <div class="welcome-visual-card welcome-orbit-visual">
            <div class="medical-orbit">
             <div class="orbit-icon"></div>
             <div class="orbit-icon"></div>
             <div class="orbit-icon"></div>
             <div class="orbit-icon"></div>
            </div>
            <div class="center-assistant">
             <div class="assistant-face"></div>
             <b>MediBuddy AI</b>
             <span>Report assistant ready</span>
            </div>
            <div class="ecg-line">
             <svg viewBox="0 0 420 48" preserveAspectRatio="none">
              <path d="M4 26 L70 26 L88 26 L104 8 L126 42 L148 20 L168 26 L230 26 L248 26 L266 12 L286 38 L306 24 L416 24"></path>
             </svg>
            </div>
           </div>
          </div>

          <div class="section-divider"></div>

          <div class="landing-lower">
           <div>
            <h3 class="landing-section-title">How it works</h3>
            <div class="how-roadmap">
             <div class="road-line"></div>
             <div class="road-step"><b>Welcome</b><span>Securely start your session with our private environment.</span></div>
             <div class="road-step"><b>Choose &amp; Upload</b><span>Select your lab report, blood test, or clinical image.</span></div>
             <div class="road-step"><b>Analyze + Chat</b><span>Read your simplified report and ask questions.</span></div>
             <div class="road-num n1">1</div>
             <div class="road-num n2">2</div>
             <div class="road-num n3">3</div>
            </div>
           </div>

           <div>
            <h3 class="landing-section-title">Why MediBuddy?</h3>
            <div class="why-grid">
             <div class="why-card"><div class="why-icon"></div><b>Clean Report Reading</b><span>Accurate text extraction from complex PDFs.</span></div>
             <div class="why-card"><div class="why-icon"></div><b>Visual Health Insights</b><span>See your results in clean, easy-to-read charts.</span></div>
             <div class="why-card"><div class="why-icon"></div><b>Smart AI Assistant</b><span>Ask questions about your results safely.</span></div>
             <div class="why-card"><div class="why-icon"></div><b>Downloadable PDF</b><span>Save your simplified summary for later.</span></div>
            </div>
           </div>
          </div>

          <div class="landing-footer">
           <span> Private &amp; Secure</span>
           <span> Educational Use Only</span>
           <span> Step-by-step Guidance</span>
          </div>
         </div>
        </div>
        """)

        with gr.Row(elem_classes=["hero-cta"]):
          get_started_btn = gr.Button(
            "Get Started",
            variant="primary",
            elem_id="get-started-btn"
          )
          gr.HTML('<span class="secure-line">100% educational private easy to use</span>')

      with gr.Tab("Page 2 Choose Option", id="choose"):
        gr.HTML("""
        <div class="choose-page-v2">
          <div class="choose-progress-v2">
            <div class="choose-progress-line"></div>
            <div class="choose-progress-item done">
              <div class="choose-progress-dot"></div>
              <span>Welcome</span>
            </div>
            <div class="choose-progress-item active">
              <div class="choose-progress-dot">2</div>
              <span>Choose</span>
            </div>
            <div class="choose-progress-item upcoming">
              <div class="choose-progress-dot">3</div>
              <span>Analyze</span>
            </div>
          </div>
        </div>
        """)

        with gr.Row(elem_classes=["choose-back-row-v2"]):
          back_to_welcome_btn = gr.Button(" Go back", elem_id="back-to-welcome-btn", elem_classes=["choose-back-btn-v2"])

        gr.HTML("""
        <div class="choose-copy-v2">
          <div class="choose-kicker-v2">STEP 2 OF 3 CHOOSE YOUR REPORT TYPE</div>
          <h2>Which report would you like<br>us to decode?</h2>
          <p>Pick your report type well open the right page instantly<br>and guide you through every step.</p>
          <div class="choose-info-v2">
            <span></span>
            <b>Youre one tap away.</b> The correct upload page opens the moment you choose.
          </div>
        </div>
        """)

        with gr.Row(elem_classes=["choice-grid-v2"]):
          with gr.Column(elem_classes=["choice-card-v2 choice-lab-v2"]):
            gr.HTML("""
            <div class="choice-icon-v2 lab-icon-v2"></div>
            <h3>Lab Report</h3>
            <p>CBC, LFT, KFT, thyroid, lipid panel, urine if it came from a lab, we can make it clear.</p>
            """)
            choice_lab_btn = gr.Button("Start with Lab Report ", variant="primary", elem_id="choice-lab-btn")

          with gr.Column(elem_classes=["choice-card-v2 choice-xray-v2"]):
            gr.HTML("""
            <div class="choice-icon-v2 xray-icon-v2"></div>
            <h3>X-ray Imaging</h3>
            <p>Upload your X-ray and well translate the findings into plain, reassuring language.</p>
            """)
            choice_xray_btn = gr.Button("Start with X-ray ", variant="primary", elem_id="choice-xray-btn")

          with gr.Column(elem_classes=["choice-card-v2 choice-compare-v2"]):
            gr.HTML("""
            <div class="choice-icon-v2"></div>
            <h3>Compare Reports</h3>
            <p>Compare previous and current lab reports, or view X-ray history side by side.</p>
            """)
            choice_compare_btn = gr.Button("Open Dashboard ", variant="primary", elem_id="choice-compare-btn")

        gr.HTML("""
        <div class="choose-mini-steps-v2">
          <div class="mini-step active"><b>1</b><span>Choose</span><small>Pick your type</small></div>
          <em></em>
          <div class="mini-step"><b>2</b><span>Upload</span><small>One easy upload</small></div>
          <em></em>
          <div class="mini-step"><b>3</b><span>Analyze</span><small>AI reads it for you</small></div>
          <em></em>
          <div class="mini-step"><b>4</b><span>Understand</span><small>Crystal-clear results</small></div>
        </div>
        """)

      with gr.Tab("Page 3 Lab Report Result", id="lab_result"):
        gr.HTML(render_step_progress(3, "Lab report analysis"))
        # Top navigation buttons removed for cleaner flow. The user continues naturally,
        # then uses the bottom Back button after reviewing the result and chatbot.
        with gr.Row(equal_height=False):
          with gr.Column(scale=3, elem_classes=["upload-card"]):
            gr.HTML("""
            <div class="upload-section-head ref-upload-head">
              <div class="upload-head-icon"></div>
              <div>
                <div class="upload-title premium-upload-title">Upload Medical Report</div>
                <div class="upload-subtitle">Upload your lab report to get AI-powered insights and analysis.</div>
              </div>
            </div>
            """)
            lab_file_input = gr.File(
              label="",
              file_types=[".pdf", ".png", ".jpg", ".jpeg"],
              type="filepath",
              elem_id="lab-upload-file",
              elem_classes=["single-green-file", "lab-single-green-file", "clean-upload-only"]
            )
            lab_uploaded_file_info = gr.HTML(value="", visible=False, elem_classes=["uploaded-file-inline"])
            gr.HTML("""
            <div class="upload-secure-strip">
              <span></span>
              <div><b>Your data is secure and private</b><small>We keep your uploaded report inside this local project workflow.</small></div>
            </div>
            """)
          with gr.Column(scale=1, elem_classes=["upload-card"]):
            gr.HTML("""
            <div class="action-panel-ref">
              <div class="action-title-ref">Action</div>
              <div class="action-illustration-ref"></div>
              <h3>Analyze Lab Report</h3>
              <p>Our AI will analyze your report and provide detailed insights.</p>
            </div>
            """)
            lab_analyze_btn = gr.Button(" Analyze Lab Report", variant="primary", elem_id="lab-analyze-btn")
            pdf_file_out = gr.File(label="Download Report", elem_id="pdf-download", visible=False)
            lab_clear_file_btn = gr.Button("Remove / Re-upload Report", elem_id="lab-clear-file-btn", elem_classes=["remove-file-btn"], visible=False)

        lab_status_out = gr.Textbox(
          label="",
          placeholder="Status will appear here after analysis...",
          interactive=False,
          lines=1,
          elem_classes=["status-box"]
        )

        lab_dashboard_out = gr.HTML(value=build_placeholder_card())

        gr.HTML("""
        <div class="assistant-dock premium-chat-shell" id="lab-ai-assistant">
          <div class="assistant-hero">
            <div class="assistant-avatar pulse-soft"></div>
            <div>
              <div class="saas-kicker">Smart AI assistant</div>
              <h2>MediBuddy AI Assistant</h2>
              <p>Ask simple questions about your uploaded report. I will answer using the analyzed report context.</p>
            </div>
          </div>
          <div class="assistant-prompt-chips">
            <span class="assistant-chip"> Is my report normal?</span>
            <span class="assistant-chip"> Explain abnormal values</span>
            <span class="assistant-chip"> What should I do next?</span>
            <span class="assistant-chip"> Create a simple table</span>
          </div>
        </div>
        """)
        lab_chatbot = gr.Chatbot(
            label="",
          
            value=[
              {
                "role": "assistant",
                "content": "Hi, I am MediBuddy AI. Upload and analyze your report, then ask me anything about the results."
                   }
                    
                    ]

                     )

        with gr.Row(elem_classes=["chat-input-row"]):
          lab_question_input = gr.Textbox(
            placeholder="Ask: What does this report mean? What should I do next?",
            label="",
            scale=6,
            lines=1
          )
          lab_ask_btn = gr.Button("", scale=1, min_width=70, elem_id="lab-ask-btn")
        with gr.Row(elem_classes=["bottom-action-row"]):
          lab_back_btn = gr.Button(" Back to Options", elem_id="lab-bottom-back-btn", elem_classes=["back-btn"])
          lab_finish_btn = gr.Button("Finish", elem_id="lab-finish-btn", elem_classes=["finish-btn"])

        # Hidden developer outputs required by the existing backend callback.
        # These are intentionally hidden from the patient interface.
        lab_table_out = gr.HTML(visible=False)
        report_type_out = gr.Textbox(visible=False)
        patient_info_out = gr.Markdown(visible=False)
        raw_text_out = gr.Textbox(visible=False)
        formatted_text_out = gr.Textbox(visible=False)
        radiology_sections_out = gr.Markdown(visible=False)
        ai_explanation_out = gr.Textbox(visible=False)
        summary_out = gr.HTML(visible=False)
        raw_json_out = gr.Code(language="json", visible=False)
        json_file_out = gr.File(visible=False)

      with gr.Tab("Page 4 X-ray Result", id="xray_result"):
        gr.HTML(render_step_progress(3, "X-ray analysis"))
        with gr.Row(equal_height=False):
          with gr.Column(scale=3, elem_classes=["upload-card", "single-upload-card", "xray-upload-only"]):
            gr.HTML("""
            <div class="upload-section-head ref-upload-head">
              <div class="upload-head-icon"></div>
              <div>
                <div class="upload-title premium-upload-title">Upload X-ray Image</div>
                <div class="upload-subtitle">Upload your X-ray image to get an AI-powered educational visual summary.</div>
              </div>
            </div>
            """)
            xray_file_input = gr.File(
              label="",
              file_types=[".png", ".jpg", ".jpeg"],
              type="filepath",
              elem_id="xray-upload-file",
              elem_classes=["single-green-file", "xray-single-green-file", "clean-upload-only"]
            )
            xray_region_input = gr.Dropdown(
              choices=XRAY_BODY_REGION_CHOICES,
              value="Auto-detect",
              label="X-ray body region",
              info="Choose the body region if you know it. Auto-detect will show Needs review if confidence is low.",
              elem_id="xray-region-input"
            )
            xray_uploaded_file_info = gr.HTML(value="", visible=False, elem_classes=["uploaded-file-inline"])
            gr.HTML("""
            <div class="upload-secure-strip">
              <span></span>
              <div><b>Your X-ray image stays private</b><small>Use this as educational support, not a confirmed diagnosis.</small></div>
            </div>
            """)
          with gr.Column(scale=1, elem_classes=["upload-card"]):
            gr.HTML("""
            <div class="action-panel-ref">
              <div class="action-title-ref">Action</div>
              <div class="action-illustration-ref"></div>
              <h3>Analyze X-ray</h3>
              <p>Our AI will review your X-ray image and create a simple educational summary.</p>
            </div>
            """)
            xray_analyze_btn = gr.Button(" Analyze X-ray", variant="primary", elem_id="xray-analyze-btn")
            xray_pdf_file = gr.File(label="Download Report", elem_id="xray-pdf-download", visible=False)
            xray_clear_file_btn = gr.Button("Remove / Re-upload Report", elem_id="xray-clear-file-btn", elem_classes=["remove-file-btn"], visible=False)
            gr.HTML("""
            <div class="disclaimer-card" style="margin-top:10px;">
              <b>Note:</b> Educational image explanation only. This is not a diagnosis.
            </div>
            """)

        xray_status_out = gr.Textbox(
          label="",
          placeholder="Status will appear here after analysis...",
          interactive=False,
          lines=1,
          elem_classes=["status-box"]
        )

        with gr.Column(elem_classes=["xray-preview-center"]):
          xray_preview = gr.Image(
            label="Uploaded X-ray Preview",
            interactive=False,
            type="filepath",
            height=560,
            elem_id="xray-preview-image"
          )

        xray_dashboard_out = gr.HTML(
          value=build_placeholder_card(
            "No X-ray result yet",
            "Upload an X-ray image and click analyze to see the visual summary."
          )
        )

        gr.HTML("""
        <div class="assistant-dock premium-chat-shell xray-open-chat-shell" id="xray-ai-assistant-preview">
          <div class="assistant-hero">
            <div class="assistant-avatar pulse-soft"></div>
            <div>
              <div class="saas-kicker">Radiology AI assistant</div>
              <h2>MediBuddy AI Assistant</h2>
              <p>Ask simple questions about your X-ray result. I will answer using the analyzed report context.</p>
            </div>
          </div>
          <div class="assistant-prompt-chips">
            <span class="assistant-chip"> Is this serious?</span>
            <span class="assistant-chip"> Explain the finding</span>
            <span class="assistant-chip"> What should I do next?</span>
            <span class="assistant-chip"> Create a simple table</span>
          </div>
        </div>
        """)
        xray_chatbot = gr.Chatbot(
          label="",
          height=360,
          elem_classes=["chatbot-box", "xray-chatbot-box"],
          value=[{"role": "assistant", "content": "Hi, I am MediBuddy AI. Upload and analyze your X-ray, then ask me anything about the results."}]
        )
        with gr.Row(elem_classes=["chat-input-row", "xray-chat-input-row"]):
          xray_question_input = gr.Textbox(
            placeholder="Ask: What does this X-ray mean? What should I do next?",
            label="",
            scale=6,
            lines=1
          )
          xray_ask_btn = gr.Button("", scale=1, min_width=70, elem_id="xray-ask-btn")
        with gr.Row(elem_classes=["bottom-action-row", "xray-bottom-actions"]):
          xray_back_btn = gr.Button(" Back to Options", elem_id="xray-bottom-back-btn", elem_classes=["back-btn"])
          xray_finish_btn = gr.Button("Finish", elem_id="xray-finish-btn", elem_classes=["finish-btn"])
          xray_chat_btn = gr.Button(" Open MediBuddy AI Assistant", visible=False, elem_id="xray-bottom-chat-btn", elem_classes=["assistant-open-btn"])

        xray_text_hidden = gr.Markdown(visible=False)

      with gr.Tab("Page 5 AI Chatbot", id="ai_chat"):
        gr.HTML(render_step_progress(3, "AI Assistant"))
        with gr.Row(elem_classes=["page-action-row"]):
          chat_back_btn = gr.Button(" Back to Options", elem_id="chat-back-btn", elem_classes=["back-btn"])
          chat_finish_btn = gr.Button("Finish", elem_id="chat-finish-btn", elem_classes=["finish-btn"])
        gr.HTML("""
        <div class="section-heading">
          <h2>AI Assistant</h2>
          <p>Ask about your lab report or X-ray result after analysis</p>
        </div>
        """)

        with gr.Row(elem_classes=["chat-layout"]):
          with gr.Column(elem_classes=["chat-history-card"]):
            gr.HTML("""
            <h3>Chat Ideas</h3>
            <div class="quick-q">Why is my hemoglobin high?</div>
            <div class="quick-q">What should I do to reduce it?</div>
            <div class="quick-q">Is my report serious?</div>
            <div class="quick-q">What foods should I avoid?</div>
            """)
            sample_q1 = gr.Button("Is my report serious?")
            sample_q2 = gr.Button("What foods should I avoid?")
            sample_q3 = gr.Button("How can I improve my health?")

          with gr.Column(elem_classes=["chat-main"]):
            chatbot = gr.Chatbot(
              label="",
              height=430,
              elem_classes=["chatbot-box"],
              value=[{"role": "assistant", "content": "Hi, I am MediBuddy AI. How can I help you understand your uploaded lab report or X-ray result?"}]
            )
            with gr.Row(elem_classes=["chat-input-row"]):
              question_input = gr.Textbox(
                placeholder="Type your question here...",
                label="",
                scale=6,
                lines=1
              )
              ask_btn = gr.Button("", scale=1, min_width=70, elem_id="ask-btn")

      with gr.Tab("Page 6 Comparison Dashboard", id="comparison"):
        gr.HTML(render_step_progress(3, "Health progress comparison"))

        with gr.Column(elem_id="comparison-dashboard-inner", elem_classes=["comparison-dashboard-inner"]):
          if build_full_comparison_dashboard_ui is not None:
            # Keep the original dashboard's "Back to comparison choices" action,
            # but tag it as hidden so only the bottom middle button is visible.
            _medibuddy_original_gr_button = gr.Button
            def _medibuddy_hidden_comparison_button(*args, **kwargs):
              _label = str(args[0] if args else kwargs.get("value", ""))
              if "Back to comparison choices" in _label or "back to comparison choices" in _label.lower():
                classes = kwargs.get("elem_classes", [])
                if isinstance(classes, str):
                  classes = [classes]
                classes = list(classes) + ["top-comparison-choices-hidden"]
                kwargs["elem_classes"] = classes
                kwargs["elem_id"] = kwargs.get("elem_id") or "original-comparison-choices-hidden"
              return _medibuddy_original_gr_button(*args, **kwargs)
            gr.Button = _medibuddy_hidden_comparison_button
            try:
              build_full_comparison_dashboard_ui()
            finally:
              gr.Button = _medibuddy_original_gr_button
          else:
            gr.HTML("""
            <div class="disclaimer-card">
              <b>Comparison dashboard could not load.</b><br>
              Please make sure <code>src/comparison_dashboard.py</code> contains
              <code>build_full_comparison_dashboard_ui()</code>.
            </div>
            """)

        with gr.Row(elem_classes=["bottom-action-row", "comparison-bottom-actions"]):
          comparison_back_btn = gr.Button(" Back to Options", elem_id="comparison-back-btn", elem_classes=["back-btn"])
          comparison_choices_bottom_btn = gr.Button(" Comparison Choices", elem_id="comparison-choices-bottom-btn", elem_classes=["back-btn", "comparison-middle-btn"])
          comparison_finish_btn = gr.Button("Finish", elem_id="comparison-finish-btn", elem_classes=["finish-btn"])

    gr.HTML("""
    <div class="app-footer">
      <div class="footer-subtext">MediBuddy AI Smart Medical Report Assistant Educational Use Only Building Smart Solutions for a Better Digital Future Pinky</div>
    </div>
    """)

  nav_loading_js = """() => {
    if (window.medibuddyShowLoader) window.medibuddyShowLoader();
    setTimeout(() => {
      const app = document.querySelector('.app-shell') || document.querySelector('.gradio-container');
      if (app) app.scrollIntoView({behavior: 'smooth', block: 'start'});
    }, 120);
    return [];
  }"""

  # Welcome navigation button: switch smoothly from Page 1 to Page 2 in Colab/Gradio.
  # The visible Gradio page tabs are hidden with CSS, so users move forward using buttons only.
  def go_to_choose_page():
    return gr.update(selected="choose")

  get_started_btn.click(
    fn=go_to_choose_page,
    inputs=[],
    outputs=[main_tabs],
    js=nav_loading_js,
  )

  # Lab report analysis buttons
  lab_outputs = [
    lab_status_out,
    lab_dashboard_out,
    lab_table_out,
    report_type_out,
    patient_info_out,
    raw_text_out,
    formatted_text_out,
    radiology_sections_out,
    ai_explanation_out,
    summary_out,
    raw_json_out,
    pdf_file_out,
    json_file_out,
  ]

  def go_to_lab_upload_page():
    return gr.update(selected="lab_result")

  def go_to_xray_upload_page():
    return gr.update(selected="xray_result")

  def go_to_comparison_page():
    return gr.update(selected="comparison")

  def go_back_to_welcome_page():
    return gr.update(selected="welcome")

  back_to_welcome_btn.click(
    fn=go_back_to_welcome_page,
    inputs=[],
    outputs=[main_tabs],
    js=nav_loading_js,
  )

  # Finish buttons: exit the current workflow and return directly to Page 1 Welcome.
  for finish_btn in [lab_finish_btn, xray_finish_btn, chat_finish_btn, comparison_finish_btn]:
    finish_btn.click(
      fn=go_back_to_welcome_page,
      inputs=[],
      outputs=[main_tabs],
      js=nav_loading_js,
    )

  lab_back_btn.click(
    fn=go_to_choose_page,
    inputs=[],
    outputs=[main_tabs],
    js=nav_loading_js,
  )

  xray_back_btn.click(
    fn=go_to_choose_page,
    inputs=[],
    outputs=[main_tabs],
    js=nav_loading_js,
  )

  xray_chat_btn.click(
    fn=lambda: gr.update(selected="ai_chat"),
    inputs=[],
    outputs=[main_tabs],
    js=nav_loading_js,
  )

  chat_back_btn.click(
    fn=go_to_choose_page,
    inputs=[],
    outputs=[main_tabs],
    js=nav_loading_js,
  )

  comparison_back_btn.click(
    fn=go_to_choose_page,
    inputs=[],
    outputs=[main_tabs],
    js=nav_loading_js,
  )

  comparison_choices_js = """() => {
    if (window.medibuddyClickOriginalComparisonChoices) {
      window.medibuddyClickOriginalComparisonChoices();
    } else {
      const norm = (s) => (s || '').replace(/\s+/g, ' ').trim().toLowerCase();
      const buttons = Array.from(document.querySelectorAll('#comparison-dashboard-inner button, button'));
      const original = buttons.find((btn) => norm(btn.textContent).includes('back to comparison choices') && !btn.closest('.comparison-bottom-actions'));
      if (original) original.click();
    }
    return [];
  }"""

  comparison_choices_bottom_btn.click(
    fn=lambda: None,
    inputs=[],
    outputs=[],
    js=comparison_choices_js,
  )


  choice_lab_btn.click(
    fn=go_to_lab_upload_page,
    inputs=[],
    outputs=[main_tabs],
    js=nav_loading_js,
  )

  lab_analyze_btn.click(
    fn=lambda uploaded_file: analyze_lab_dashboard_ui(uploaded_file, "English"),
    inputs=[lab_file_input],
    outputs=lab_outputs,
  )

  lab_file_input.change(
    fn=lambda uploaded_file: (
      gr.update(visible=uploaded_file is not None),
      build_uploaded_file_info(uploaded_file),
      gr.update(value=None, visible=False),
    ),
    inputs=[lab_file_input],
    outputs=[lab_clear_file_btn, lab_uploaded_file_info, pdf_file_out],
  )



  lab_clear_file_btn.click(
    fn=lambda: (
      gr.update(value=None),
      gr.update(value=None, visible=False),
      "",
      build_placeholder_card(),
      gr.update(value=None),
      gr.update(visible=False),
      gr.update(value="", visible=False),
    ),
    inputs=[],
    outputs=[lab_file_input, pdf_file_out, lab_status_out, lab_dashboard_out, json_file_out, lab_clear_file_btn, lab_uploaded_file_info],
  )

  # X-ray analysis buttons
  xray_outputs = [
    xray_status_out,
    xray_preview,
    xray_dashboard_out,
    xray_text_hidden,
    xray_pdf_file,
  ]

  choice_xray_btn.click(
    fn=go_to_xray_upload_page,
    inputs=[],
    outputs=[main_tabs],
    js=nav_loading_js,
  )

  choice_compare_btn.click(
    fn=go_to_comparison_page,
    inputs=[],
    outputs=[main_tabs],
    js=nav_loading_js,
  )





  xray_analyze_btn.click(
    fn=lambda uploaded_xray_file, selected_xray_region: analyze_xray_dashboard_ui(uploaded_xray_file, "English", selected_xray_region),
    inputs=[xray_file_input, xray_region_input],
    outputs=xray_outputs,
  )

  xray_file_input.change(
    fn=lambda uploaded_file: (
      gr.update(visible=uploaded_file is not None),
      build_uploaded_file_info(uploaded_file),
      gr.update(value=None, visible=False),
    ),
    inputs=[xray_file_input],
    outputs=[xray_clear_file_btn, xray_uploaded_file_info, xray_pdf_file],
  )



  xray_clear_file_btn.click(
    fn=lambda: (
      gr.update(value=None),
      "",
      gr.update(value=None),
      build_placeholder_card("No X-ray result yet", "Upload an X-ray image and click analyze to see the visual summary."),
      gr.update(value=None, visible=False),
      "",
      gr.update(visible=False),
      gr.update(value="", visible=False),
    ),
    inputs=[],
    outputs=[xray_file_input, xray_status_out, xray_preview, xray_dashboard_out, xray_pdf_file, xray_text_hidden, xray_clear_file_btn, xray_uploaded_file_info],
  )

  # Lab page chatbot
  lab_ask_btn.click(
    fn=ask_question_about_report,
    inputs=[lab_question_input, lab_chatbot],
    outputs=[lab_chatbot, lab_question_input],
  )

  lab_question_input.submit(
    fn=ask_question_about_report,
    inputs=[lab_question_input, lab_chatbot],
    outputs=[lab_chatbot, lab_question_input],
  )

  # X-ray page chatbot (open layout)
  xray_ask_btn.click(
    fn=ask_question_about_report,
    inputs=[xray_question_input, xray_chatbot],
    outputs=[xray_chatbot, xray_question_input],
  )

  xray_question_input.submit(
    fn=ask_question_about_report,
    inputs=[xray_question_input, xray_chatbot],
    outputs=[xray_chatbot, xray_question_input],
  )

  # AI assistant for both lab report and X-ray context
  ask_btn.click(
    fn=ask_question_about_report,
    inputs=[question_input, chatbot],
    outputs=[chatbot, question_input],
  )

  question_input.submit(
    fn=ask_question_about_report,
    inputs=[question_input, chatbot],
    outputs=[chatbot, question_input],
  )

  sample_q1.click(fn=lambda: "Is my report serious?", inputs=[], outputs=[question_input])
  sample_q2.click(fn=lambda: "What foods should I avoid?", inputs=[], outputs=[question_input])
  sample_q3.click(fn=lambda: "How can I improve my health?", inputs=[], outputs=[question_input])

# launch is handled by app.py



