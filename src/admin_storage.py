import json
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = DATA_DIR / "logs"
LOG_FILE = LOG_DIR / "analysis_history.jsonl"

LOG_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "uploads").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "reports").mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Safe conversion helpers
# -----------------------------------------------------------------------------


def _safe_str(value: Any, default: str = "") -> str:
  if value is None:
    return default
  return str(value).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
  try:
    if value in [None, "", "None"]:
      return default
    return float(value)
  except Exception:
    return default


def _safe_int(value: Any, default: int = 0) -> int:
  try:
    if value in [None, "", "None"]:
      return default
    return int(float(value))
  except Exception:
    return default


def _normalize_event(payload: Dict[str, Any]) -> Dict[str, Any]:
  """Return a consistent event shape for admin analytics.

  The user app may log slightly different payloads for lab reports and X-ray
  reviews. This function keeps the dashboard stable by making sure all common
  fields exist and numeric fields are safe.
  """
  event = dict(payload or {})

  event.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
  event["analysis_type"] = _safe_str(event.get("analysis_type"), "Unknown") or "Unknown"
  event["file_name"] = _safe_str(event.get("file_name"), "Unknown") or "Unknown"
  event["report_category"] = _safe_str(event.get("report_category"), "Unknown") or "Unknown"
  event["report_subtype"] = _safe_str(
    event.get("report_subtype") or event.get("report_category"), "Unknown"
  ) or "Unknown"
  event["risk_level"] = _safe_str(event.get("risk_level"), "Unknown") or "Unknown"
  event["ocr_quality"] = _safe_str(event.get("ocr_quality"), "Unknown") or "Unknown"
  event["status"] = _safe_str(event.get("status"), "success") or "success"

  event["risk_score"] = _safe_float(event.get("risk_score"), 0)
  event["ocr_score"] = _safe_float(event.get("ocr_score"), 0)
  event["total_tests"] = _safe_int(event.get("total_tests"), 0)
  event["abnormal_count"] = _safe_int(event.get("abnormal_count"), 0)

  return event


# -----------------------------------------------------------------------------
# Logging API used by the user app
# -----------------------------------------------------------------------------


def log_analysis_event(payload: Dict[str, Any]) -> bool:
  """Append one report-analysis event for the admin dashboard/history."""
  try:
    event = _normalize_event(payload or {})
    event["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
      f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return True
  except Exception:
    return False


def read_history(limit: int = 200) -> List[Dict[str, Any]]:
  """Read latest report-analysis history.

  Bad/corrupt JSONL lines are ignored so one broken log entry does not break
  the admin dashboard.
  """
  try:
    limit = max(1, int(limit))
  except Exception:
    limit = 200

  if not LOG_FILE.exists():
    return []

  rows: List[Dict[str, Any]] = []
  try:
    with LOG_FILE.open("r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          parsed = json.loads(line)
          if isinstance(parsed, dict):
            rows.append(_normalize_event(parsed))
        except Exception:
          continue
  except Exception:
    return []

  return rows[-limit:][::-1]


# Kept for backwards compatibility with older code, but the admin UI no longer
# exposes a Clear Logs button.
def clear_history() -> bool:
  try:
    LOG_FILE.write_text("", encoding="utf-8")
    return True
  except Exception:
    return False


# -----------------------------------------------------------------------------
# Analytics helpers for admin_panel.py
# -----------------------------------------------------------------------------


def _is_lab_event(row: Dict[str, Any]) -> bool:
  text = f"{row.get('analysis_type', '')} {row.get('report_category', '')} {row.get('report_subtype', '')}".lower()
  return any(term in text for term in ["lab", "laboratory", "cbc", "hematology", "biochemistry"])


def _is_xray_event(row: Dict[str, Any]) -> bool:
  text = f"{row.get('analysis_type', '')} {row.get('report_category', '')} {row.get('report_subtype', '')}".lower()
  return any(term in text for term in ["x-ray", "xray", "x ray", "radiology"])


def _is_attention_risk(row: Dict[str, Any]) -> bool:
  risk = _safe_str(row.get("risk_level")).lower()
  return (
    risk in ["high", "critical", "needs attention", "high risk", "critical risk"]
    or "attention" in risk
    or "critical" in risk
  )


def _average(values: List[float]) -> float:
  values = [v for v in values if v > 0]
  return round(sum(values) / len(values), 1) if values else 0


def _daily_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
  counts: Dict[str, int] = defaultdict(int)
  for row in rows:
    stamp = _safe_str(row.get("timestamp"))
    day = stamp[:10] if len(stamp) >= 10 else "Unknown"
    counts[day] += 1
  return dict(sorted(counts.items())[-14:])


def dashboard_stats() -> Dict[str, Any]:
  """Return safe dashboard statistics for the admin UI."""
  rows = read_history(10000)
  total = len(rows)

  lab = sum(1 for row in rows if _is_lab_event(row))
  xray = sum(1 for row in rows if _is_xray_event(row))
  high = sum(1 for row in rows if _is_attention_risk(row))

  ocr_scores = [_safe_float(row.get("ocr_score")) for row in rows]
  risk_scores = [_safe_float(row.get("risk_score")) for row in rows]
  abnormal_counts = [_safe_int(row.get("abnormal_count")) for row in rows]

  by_type = Counter(
    _safe_str(row.get("report_subtype") or row.get("report_category"), "Unknown") or "Unknown"
    for row in rows
  )
  by_risk = Counter(_safe_str(row.get("risk_level"), "Unknown") or "Unknown" for row in rows)
  by_analysis_type = Counter(_safe_str(row.get("analysis_type"), "Unknown") or "Unknown" for row in rows)
  by_status = Counter(_safe_str(row.get("status"), "success") or "success" for row in rows)

  success_count = sum(
    1
    for row in rows
    if _safe_str(row.get("status"), "success").lower() in ["success", "", "none"]
  )

  return {
    "total": total,
    "lab": lab,
    "xray": xray,
    "high": high,
    "avg_ocr": _average(ocr_scores),
    "avg_risk_score": _average(risk_scores),
    "avg_abnormal_count": _average([float(x) for x in abnormal_counts]),
    "success_rate": round((success_count / max(total, 1)) * 100, 1) if total else 0,
    "by_type": dict(by_type.most_common(8)),
    "by_risk": dict(by_risk.most_common(8)),
    "by_analysis_type": dict(by_analysis_type.most_common(8)),
    "by_status": dict(by_status.most_common(8)),
    "daily_counts": _daily_counts(list(reversed(rows))),
    "recent": rows[:50],
  }

