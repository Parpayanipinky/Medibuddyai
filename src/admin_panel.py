import os
import html
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import gradio as gr

from src.admin_storage import dashboard_stats, read_history


# -----------------------------------------------------------------------------
# Admin authentication
# -----------------------------------------------------------------------------


def _esc(value: Any) -> str:
  return html.escape(str(value if value is not None else ""))


def _check_admin_password(password: str) -> bool:
  required = os.getenv("MEDIBUDDY_ADMIN_PASSWORD", "admin123").strip()
  return str(password or "") == required


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------


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


def _history_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
  desired_columns = [
    "timestamp",
    "analysis_type",
    "file_name",
    "report_subtype",
    "risk_level",
    "risk_score",
    "ocr_quality",
    "ocr_score",
    "total_tests",
    "abnormal_count",
    "status",
  ]
  display_columns = {
    "timestamp": "Time",
    "analysis_type": "Type",
    "file_name": "File",
    "report_subtype": "Report",
    "risk_level": "Risk",
    "risk_score": "Risk Score",
    "ocr_quality": "OCR Quality",
    "ocr_score": "OCR Score",
    "total_tests": "Tests",
    "abnormal_count": "Abnormal",
    "status": "Status",
  }
  df = pd.DataFrame(rows)
  if df.empty:
    return pd.DataFrame(columns=[display_columns[c] for c in desired_columns])
  keep = [c for c in desired_columns if c in df.columns]
  return df[keep].rename(columns=display_columns)


def _counter_from_rows(rows: Iterable[Dict[str, Any]], key: str, fallback: str = "Unknown") -> Dict[str, int]:
  counter = Counter(str(row.get(key) or fallback) for row in rows)
  return dict(counter.most_common(8))


def _status_health(rows: List[Dict[str, Any]]) -> Tuple[str, str, str]:
  if not rows:
    return "No Activity", "No analysis events have been logged yet.", "idle"

  failures = sum(1 for row in rows if str(row.get("status", "")).lower() not in ["success", "", "none"])
  attention = sum(
    1
    for row in rows
    if str(row.get("risk_level", "")).lower() in ["high", "critical", "needs attention"]
    or "attention" in str(row.get("risk_level", "")).lower()
  )

  if failures:
    return "Needs Review", f"{failures} event(s) have a non-success status.", "danger"
  if attention:
    return "Attention Cases", f"{attention} high-risk or attention case(s) found.", "warn"
  return "Healthy", "All recent events are being logged successfully.", "good"


# -----------------------------------------------------------------------------
# Chart helpers
# -----------------------------------------------------------------------------


def _empty_chart(title: str, icon: str = "", message: str = "No data available yet.") -> str:
  return f"""
  <div class="chart-card">
   <div class="chart-header">
    <span class="chart-icon">{icon}</span>
    <span class="chart-title">{_esc(title)}</span>
   </div>
   <div class="empty-chart">
    <div class="empty-icon"></div>
    <div>{_esc(message)}</div>
   </div>
  </div>
  """


def _bar_chart_html(title: str, values: Dict[str, int], icon: str = "", note: str = "") -> str:
  if not values:
    return _empty_chart(title, icon)

  max_value = max(values.values()) or 1
  color_stops = [
    "var(--accent-blue)", "var(--accent-teal)", "var(--accent-indigo)",
    "var(--accent-cyan)", "var(--accent-emerald)", "var(--accent-violet)",
    "var(--accent-sky)", "var(--accent-green)",
  ]
  rows_html = []
  for i, (label, value) in enumerate(values.items()):
    width = max(6, int((value / max_value) * 100))
    color = color_stops[i % len(color_stops)]
    pct = int((value / max_value) * 100)
    rows_html.append(f"""
      <div class="bar-row" style="animation-delay:{i * 0.07}s">
       <div class="bar-label-wrap">
        <span class="bar-label" title="{_esc(label)}">{_esc(label)}</span>
        <span class="bar-pct">{pct}%</span>
       </div>
       <div class="bar-track">
        <div class="bar-fill" style="width:{width}%; background:{color};"></div>
       </div>
       <div class="bar-value">{_esc(value)}</div>
      </div>
    """)

  return f"""
  <div class="chart-card">
   <div class="chart-header">
    <span class="chart-icon">{icon}</span>
    <span class="chart-title">{_esc(title)}</span>
    {f'<span class="chart-note-inline">{_esc(note)}</span>' if note else ''}
   </div>
   <div class="bar-chart">{''.join(rows_html)}</div>
  </div>
  """


def _line_chart_svg(title: str, values: List[float], suffix: str = "", max_y: float | None = None, icon: str = "") -> str:
  if not values:
    return _empty_chart(title, icon)

  values = values[-24:]
  max_y = max_y or max(max(values), 1)
  min_y = min(min(values) * 0.9, 0)
  width, height = 580, 200
  left, right, top, bottom = 44, 12, 16, 36
  plot_w = width - left - right
  plot_h = height - top - bottom

  def point(idx: int, value: float) -> Tuple[float, float]:
    x = left + (plot_w * idx / max(len(values) - 1, 1))
    y = top + plot_h - ((value - min_y) / max(max_y - min_y, 1)) * plot_h
    return x, y

  points = [point(i, v) for i, v in enumerate(values)]
  polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
  area_pts = (
    f"{left},{height - bottom} "
    + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    + f" {points[-1][0]:.1f},{height - bottom}"
  )

  circles_html = "".join(
    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" class="data-circle"><title>{round(values[i], 2)}{suffix}</title></circle>'
    for i, (x, y) in enumerate(points)
  )

  latest = values[-1]
  avg_val = round(sum(values) / len(values), 1)

  # Y-axis labels
  y_labels = [
    f'<text x="{left - 5}" y="{top + 6}" class="axis-label" text-anchor="end">{round(max_y, 0):.0f}</text>',
    f'<text x="{left - 5}" y="{top + plot_h // 2 + 4}" class="axis-label" text-anchor="end">{round((max_y + min_y) / 2, 0):.0f}</text>',
    f'<text x="{left - 5}" y="{top + plot_h + 4}" class="axis-label" text-anchor="end">{round(min_y, 0):.0f}</text>',
  ]

  return f"""
  <div class="chart-card">
   <div class="chart-header">
    <span class="chart-icon">{icon}</span>
    <span class="chart-title">{_esc(title)}</span>
    <span class="chart-badge">Latest: <b>{round(latest, 1)}{_esc(suffix)}</b></span>
   </div>
   <div class="svg-wrap">
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">
     <defs>
      <linearGradient id="lg_{title[:6].replace(' ', '')}" x1="0" y1="0" x2="0" y2="1">
       <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.28" />
       <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.02" />
      </linearGradient>
     </defs>
     <!-- grid lines -->
     <line x1="{left}" y1="{top}" x2="{width - right}" y2="{top}" class="grid-line" />
     <line x1="{left}" y1="{top + plot_h // 2}" x2="{width - right}" y2="{top + plot_h // 2}" class="grid-line" />
     <line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" class="grid-line" />
     <!-- axes -->
     <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis" />
     <!-- area fill -->
     <polygon points="{area_pts}" fill="url(#lg_{title[:6].replace(' ', '')})" />
     <!-- trend line -->
     <polyline points="{polyline}" class="trend-line" />
     <!-- circles -->
     {circles_html}
     <!-- labels -->
     {''.join(y_labels)}
     <text x="{left}" y="{height - 4}" class="axis-label">oldest</text>
     <text x="{width - right}" y="{height - 4}" class="axis-label" text-anchor="end">latest</text>
    </svg>
   </div>
   <div class="chart-footer">
    <span>Avg: <b>{avg_val}{_esc(suffix)}</b></span>
    <span>Points: <b>{len(values)}</b></span>
    <span>Peak: <b>{round(max(values), 1)}{_esc(suffix)}</b></span>
   </div>
  </div>
  """


def _donut_chart_html(title: str, good: int, warning: int, danger: int) -> str:
  total = max(good + warning + danger, 1)
  good_pct = round((good / total) * 100)
  warning_pct = round((warning / total) * 100)
  danger_pct = 100 - good_pct - warning_pct

  segments = []
  offset = 0
  r = 54
  circ = 2 * 3.14159 * r
  cx, cy = 70, 70

  def arc_segment(pct, color, dash_offset):
    dash = (pct / 100) * circ
    gap = circ - dash
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="18" stroke-dasharray="{dash:.1f} {gap:.1f}" stroke-dashoffset="{-dash_offset:.1f}" style="transform:rotate(-90deg);transform-origin:{cx}px {cy}px;" />'

  used = 0
  if good_pct > 0:
    segments.append(arc_segment(good_pct, "var(--accent-emerald)", used * circ / 100))
    used += good_pct
  if warning_pct > 0:
    segments.append(arc_segment(warning_pct, "var(--accent-amber)", used * circ / 100))
    used += warning_pct
  if danger_pct > 0:
    segments.append(arc_segment(danger_pct, "var(--accent-red)", used * circ / 100))

  dominant_label = "Low Risk" if good >= warning and good >= danger else ("Moderate" if warning >= danger else "High Risk")
  dominant_pct = good_pct if good >= warning and good >= danger else (warning_pct if warning >= danger else danger_pct)

  return f"""
  <div class="chart-card">
   <div class="chart-header">
    <span class="chart-icon"></span>
    <span class="chart-title">{_esc(title)}</span>
   </div>
   <div class="donut-wrap">
    <div class="donut-svg-wrap">
     <svg viewBox="0 0 140 140" width="160" height="160">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="rgba(148,163,184,0.12)" stroke-width="18" />
      {''.join(segments)}
      <text x="{cx}" y="{cy - 6}" text-anchor="middle" class="donut-center-num">{total}</text>
      <text x="{cx}" y="{cy + 14}" text-anchor="middle" class="donut-center-lbl">events</text>
     </svg>
    </div>
    <div class="donut-legend">
     <div class="legend-row">
      <span class="legend-swatch" style="background:var(--accent-emerald)"></span>
      <span class="legend-label">Low / Normal</span>
      <span class="legend-count">{good}</span>
      <span class="legend-pct">{good_pct}%</span>
     </div>
     <div class="legend-row">
      <span class="legend-swatch" style="background:var(--accent-amber)"></span>
      <span class="legend-label">Moderate</span>
      <span class="legend-count">{warning}</span>
      <span class="legend-pct">{warning_pct}%</span>
     </div>
     <div class="legend-row">
      <span class="legend-swatch" style="background:var(--accent-red)"></span>
      <span class="legend-label">High / Critical</span>
      <span class="legend-count">{danger}</span>
      <span class="legend-pct">{danger_pct}%</span>
     </div>
     <div class="donut-dominant">
      Dominant: <b>{dominant_label}</b> ({dominant_pct}%)
     </div>
    </div>
   </div>
  </div>
  """


def _sparkline_svg(values: List[float], color: str = "#38bdf8") -> str:
  if not values or len(values) < 2:
    return ""
  vmin, vmax = min(values), max(values)
  rng = max(vmax - vmin, 1)
  w, h = 80, 28
  pts = []
  for i, v in enumerate(values[-12:]):
    x = i / max(len(values[-12:]) - 1, 1) * w
    y = h - ((v - vmin) / rng) * h
    pts.append(f"{x:.1f},{y:.1f}")
  return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>'


def _build_charts_html(rows: List[Dict[str, Any]]) -> str:
  chronological = list(reversed(rows))

  type_counts = _counter_from_rows(rows, "analysis_type")
  subtype_counts = _counter_from_rows(rows, "report_subtype")
  risk_counts = _counter_from_rows(rows, "risk_level")

  ocr_values = [_safe_float(row.get("ocr_score")) for row in chronological if _safe_float(row.get("ocr_score")) > 0]
  risk_scores = [_safe_float(row.get("risk_score")) for row in chronological if row.get("risk_score") not in [None, ""]]
  abnormal_values = [_safe_float(row.get("abnormal_count")) for row in chronological if row.get("abnormal_count") not in [None, ""]]

  good_risk = sum(1 for row in rows if str(row.get("risk_level", "")).lower() in ["low", "normal", "low risk"])
  danger_risk = sum(
    1 for row in rows
    if str(row.get("risk_level", "")).lower() in ["high", "critical", "needs attention", "high risk"]
    or "attention" in str(row.get("risk_level", "")).lower()
  )
  warning_risk = max(len(rows) - good_risk - danger_risk, 0)

  return f"""
  <div class="section-block">
   <div class="section-header">
    <div>
     <div class="section-eyebrow"> Analytics</div>
     <div class="section-title">Monitoring Signals</div>
    </div>
    <div class="section-subtitle">Charts update from local JSONL analysis logs</div>
   </div>

   <div class="chart-grid-2">
    {_bar_chart_html("Analysis Volume by Type", type_counts, "", "Lab and X-ray activity split.")}
    {_donut_chart_html("Risk Distribution", good_risk, warning_risk, danger_risk)}
   </div>

   <div class="chart-grid-3">
    {_line_chart_svg("OCR Quality Trend", ocr_values, "%", 100, "")}
    {_line_chart_svg("Risk Score Trend", risk_scores, "/100", 100, "")}
    {_line_chart_svg("Abnormal Findings Trend", abnormal_values, "", max(max(abnormal_values), 5) if abnormal_values else None, "")}
   </div>

   <div class="chart-grid-2">
    {_bar_chart_html("Top Report Subtypes", subtype_counts, "", "Most frequently analyzed report categories.")}
    {_bar_chart_html("Risk Level Breakdown", risk_counts, "", "Counts for each logged risk label.")}
   </div>
  </div>
  """


# -----------------------------------------------------------------------------
# Dashboard rendering
# -----------------------------------------------------------------------------


def _metric_card(title: str, value: Any, caption: str, icon: str = "", accent: str = "", sparkline_html: str = "") -> str:
  return f"""
  <div class="kpi-card {accent}">
   <div class="kpi-top">
    <div class="kpi-icon">{icon}</div>
    {sparkline_html}
   </div>
   <div class="kpi-label">{_esc(title)}</div>
   <div class="kpi-value">{_esc(value)}</div>
   <div class="kpi-caption">{_esc(caption)}</div>
  </div>
  """


def _build_dashboard_html() -> Tuple[str, pd.DataFrame]:
  stats = dashboard_stats()
  rows = read_history(10000)
  health_label, health_note, health_class = _status_health(rows)

  total = _safe_int(stats.get("total"))
  lab = _safe_int(stats.get("lab"))
  xray = _safe_int(stats.get("xray"))
  high = _safe_int(stats.get("high"))
  avg_ocr = _safe_float(stats.get("avg_ocr"))
  avg_risk_score = _safe_float(stats.get("avg_risk_score"))
  success_rate = _safe_float(stats.get("success_rate"))

  chronological = list(reversed(rows))
  ocr_spark = [_safe_float(r.get("ocr_score")) for r in chronological if _safe_float(r.get("ocr_score")) > 0]
  risk_spark = [_safe_float(r.get("risk_score")) for r in chronological if r.get("risk_score") not in [None, ""]]

  latest_time = rows[0].get("timestamp", "No activity yet") if rows else "No activity yet"
  recent_df = _history_dataframe(stats.get("recent", []))
  charts_html = _build_charts_html(rows)

  # Status icon
  status_icon = {"good": "", "warn": "", "danger": "", "idle": ""}.get(health_class, "")
  status_color = {"good": "var(--accent-emerald)", "warn": "var(--accent-amber)", "danger": "var(--accent-red)", "idle": "var(--muted)"}.get(health_class, "var(--muted)")

  kpi_cards = "".join([
    _metric_card("Total Analyses", f"{total:,}", "All logged events", ""),
    _metric_card("Lab Reports", f"{lab:,}", "Text / OCR report analyses", ""),
    _metric_card("X-Ray Reviews", f"{xray:,}", "Image-based analyses", ""),
    _metric_card("High / Attention", f"{high:,}", "Cases requiring review", "", "kpi-danger"),
    _metric_card("Avg OCR Quality", f"{avg_ocr:.1f}%", "Mean extraction quality", "",
           sparkline_html=_sparkline_svg(ocr_spark, "#38bdf8")),
    _metric_card("Avg Risk Score", f"{avg_risk_score:.1f}/100", "Rule-based risk estimate", "",
           sparkline_html=_sparkline_svg(risk_spark, "#f472b6")),
  ])

  html_card = f"""
  <div class="dash-root">

   <!-- HERO BANNER -->
   <div class="hero-banner">
    <div class="hero-left">
     <div class="hero-eyebrow"> MediBuddy AI Admin Dashboard</div>
     <h1 class="hero-title">End-to-End Monitoring</h1>
     <p class="hero-sub">Track report usage, risk cases, OCR quality, abnormal findings, and real-time analysis activity.</p>
     <div class="hero-pills">
      <span class="pill">Last activity: <b>{_esc(latest_time)}</b></span>
      <span class="pill pill-status" style="border-color:{status_color}; color:{status_color};">{status_icon} {_esc(health_label)}</span>
      <span class="pill">Success rate: <b>{_esc(success_rate)}%</b></span>
     </div>
    </div>
    <div class="hero-right">
     <div class="hero-stat-label">TRACKED EVENTS</div>
     <div class="hero-stat-num">{total:,}</div>
     <div class="hero-stat-sub">across all analysis sessions</div>
     <div class="hero-ring"></div>
    </div>
   </div>

   <!-- KPI GRID -->
   <div class="section-block">
    <div class="section-header">
     <div>
      <div class="section-eyebrow"> Key Metrics</div>
      <div class="section-title">Performance Overview</div>
     </div>
    </div>
    <div class="kpi-grid">{kpi_cards}</div>
   </div>

   <!-- SYSTEM STATUS -->
   <div class="status-banner status-{_esc(health_class)}">
    <div class="status-left">
     <div class="status-dot"></div>
     <div>
      <div class="status-label">Operational Status</div>
      <div class="status-title">{_esc(health_label)}</div>
      <div class="status-note">{_esc(health_note)}</div>
     </div>
    </div>
    <div class="status-right">
     <div class="status-icon">{status_icon}</div>
    </div>
   </div>

   <!-- CHARTS -->
   {charts_html}

  </div>
  """
  return html_card, recent_df


def handle_admin_login(password: str):
  if not _check_admin_password(password):
    return (
      False,
      gr.update(visible=True),
      gr.update(visible=False),
      "",
      _history_dataframe([]),
      "<div class='login-error'> Incorrect password. Please try again.</div>",
    )
  dashboard_html, history_df = _build_dashboard_html()
  return (
    True,
    gr.update(visible=False),
    gr.update(visible=True),
    dashboard_html,
    history_df,
    "",
  )


def refresh_admin_dashboard(is_logged_in: bool):
  if not is_logged_in:
    return "", _history_dataframe([]), "<div class='login-error'>Please log in first.</div>"
  dashboard_html, history_df = _build_dashboard_html()
  return dashboard_html, history_df, "<div class='status-ok'> Dashboard refreshed successfully.</div>"


def logout_admin():
  return (
    False,
    gr.update(visible=True),
    gr.update(visible=False),
    "",
    _history_dataframe([]),
    "<div class='status-ok'> Logged out successfully.</div>",
  )


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------


ADMIN_CSS = """
/* Design Tokens MediBuddy Light Theme */
:root {
 --bg-base:    #e8eef0;
 --bg-surface:  #dde5e8;
 --bg-card:    #ffffff;
 --bg-card-alt:  rgba(255,255,255,0.70);
 --stroke:    rgba(26, 90, 70, 0.10);
 --stroke-bright: rgba(26, 90, 70, 0.18);
 --text:     #0f2620;
 --text-dim:   #3d6058;
 --muted:     #7a9e95;
 --brand:     #1a5a46;
 --brand-mid:   #1e7a5e;
 --brand-light:  #2dab85;
 --brand-pale:  #d0ede6;
 --accent-blue:  #2563eb;
 --accent-sky:  #0ea5e9;
 --accent-teal:  #0d9488;
 --accent-cyan:  #06b6d4;
 --accent-indigo: #4f46e5;
 --accent-violet: #7c3aed;
 --accent-emerald:#059669;
 --accent-green: #16a34a;
 --accent-amber: #d97706;
 --accent-red:  #dc2626;
 --accent-rose:  #e11d48;
 --font: 'DM Sans', 'Segoe UI', system-ui, sans-serif;
 --radius-lg: 20px;
 --radius-xl: 28px;
 --shadow-card: 0 4px 20px rgba(26,90,70,0.09), 0 1px 0 var(--stroke-bright) inset;
}

/* Base */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,700;0,9..40,800;0,9..40,900&family=DM+Mono:wght@500&display=swap');

.gradio-container {
 background: var(--bg-base) !important;
 min-height: 100vh;
 font-family: var(--font) !important;
}
.main {
 max-width: 1400px !important;
 margin: 0 auto !important;
 padding: 24px 20px 60px !important;
}

/* Login */
#login-shell {
 max-width: 520px;
 margin: 80px auto 0;
}
.login-card {
 background: var(--bg-card);
 border: 1px solid var(--stroke-bright);
 border-radius: var(--radius-xl);
 padding: 44px 40px;
 box-shadow: var(--shadow-card), 0 0 60px rgba(26,90,70,.07);
 position: relative;
 overflow: hidden;
}
.login-card::before {
 content: '';
 position: absolute;
 top: -60px; right: -60px;
 width: 200px; height: 200px;
 border-radius: 50%;
 background: radial-gradient(circle, rgba(45,171,133,.12), transparent 70%);
 pointer-events: none;
}
.login-logo {
 width: 60px; height: 60px;
 border-radius: 16px;
 background: linear-gradient(135deg, var(--brand), var(--brand-light));
 display: grid; place-items: center;
 font-size: 26px;
 box-shadow: 0 12px 32px rgba(26,90,70,.22);
 margin-bottom: 24px;
}
.login-card h1 {
 margin: 0 0 10px;
 font-size: 30px;
 font-weight: 800;
 color: var(--text);
 line-height: 1.1;
}
.login-card p {
 margin: 0;
 color: var(--text-dim);
 font-size: 14px;
 line-height: 1.7;
}
.eyebrow {
 font-size: 11px;
 font-weight: 700;
 text-transform: uppercase;
 letter-spacing: .14em;
 color: var(--brand-mid);
 margin-bottom: 10px;
}
.login-error, .status-ok {
 margin-top: 12px;
 padding: 12px 16px;
 border-radius: 12px;
 font-weight: 600;
 font-size: 14px;
}
.login-error { background: rgba(220,38,38,.08); border: 1px solid rgba(220,38,38,.22); color: #b91c1c; }
.status-ok  { background: rgba(5,150,105,.08); border: 1px solid rgba(5,150,105,.22); color: #065f46; }

/* Buttons */
#login-btn button, #refresh-btn button {
 background: linear-gradient(135deg, var(--brand), var(--brand-light)) !important;
 color: #fff !important;
 border: none !important;
 border-radius: 14px !important;
 min-height: 48px !important;
 font-weight: 700 !important;
 font-size: 15px !important;
 letter-spacing: .02em !important;
 box-shadow: 0 6px 20px rgba(26,90,70,.22) !important;
 transition: opacity .2s !important;
}
#login-btn button:hover, #refresh-btn button:hover { opacity: .88 !important; }
#logout-btn button {
 background: #fff !important;
 color: var(--brand) !important;
 border: 1px solid var(--stroke-bright) !important;
 border-radius: 14px !important;
 min-height: 48px !important;
 font-weight: 700 !important;
 font-size: 15px !important;
}
#login-input input, #login-input textarea {
 background: var(--bg-base) !important;
 border: 1px solid var(--stroke-bright) !important;
 border-radius: 12px !important;
 color: var(--text) !important;
}
#top-controls { gap: 12px !important; margin-bottom: 18px !important; }

/* Dashboard Shell */
#dashboard-shell, #dashboard-group { background: transparent !important; }
.dash-root { font-family: var(--font); color: var(--text); }

/* Hero Banner */
.hero-banner {
 display: grid;
 grid-template-columns: 1fr auto;
 gap: 24px;
 background: var(--bg-card);
 border: 1px solid var(--stroke-bright);
 border-radius: var(--radius-xl);
 padding: 36px 40px;
 margin-bottom: 20px;
 box-shadow: var(--shadow-card);
 position: relative;
 overflow: hidden;
}
.hero-banner::after {
 content: '';
 position: absolute;
 inset: 0;
 background: linear-gradient(135deg, rgba(45,171,133,.05) 0%, rgba(26,90,70,.04) 100%);
 pointer-events: none;
}
.hero-eyebrow {
 font-size: 12px;
 font-weight: 700;
 text-transform: uppercase;
 letter-spacing: .14em;
 color: var(--brand-mid);
 margin-bottom: 10px;
}
.hero-title {
 margin: 0 0 12px;
 font-size: 40px;
 font-weight: 900;
 line-height: 1.05;
 color: var(--brand);
 letter-spacing: -.02em;
}
.hero-sub {
 margin: 0 0 20px;
 color: var(--text-dim);
 font-size: 15px;
 line-height: 1.7;
 max-width: 580px;
}
.hero-pills { display: flex; flex-wrap: wrap; gap: 10px; }
.pill {
 background: var(--brand-pale);
 border: 1px solid rgba(26,90,70,.15);
 border-radius: 999px;
 padding: 7px 14px;
 font-size: 13px;
 color: var(--brand);
}
.pill b { color: var(--brand); }
.pill-status { font-weight: 700; }
.hero-right {
 background: linear-gradient(145deg, var(--brand), var(--brand-light));
 border: none;
 border-radius: var(--radius-lg);
 padding: 28px 32px;
 text-align: center;
 min-width: 180px;
 position: relative;
 display: flex;
 flex-direction: column;
 justify-content: center;
 box-shadow: 0 8px 32px rgba(26,90,70,.22);
}
.hero-stat-label {
 font-size: 10px;
 font-weight: 800;
 text-transform: uppercase;
 letter-spacing: .16em;
 color: rgba(255,255,255,.75);
 margin-bottom: 8px;
}
.hero-stat-num {
 font-size: 56px;
 font-weight: 900;
 color: #fff;
 line-height: 1;
 letter-spacing: -.03em;
 font-family: 'DM Mono', monospace;
}
.hero-stat-sub {
 font-size: 12px;
 color: rgba(255,255,255,.65);
 margin-top: 8px;
}
.hero-ring {
 position: absolute;
 top: -40px; right: -40px;
 width: 130px; height: 130px;
 border-radius: 50%;
 border: 22px solid rgba(255,255,255,.10);
}

/* Section Blocks */
.section-block { margin-bottom: 24px; }
.section-header {
 display: flex;
 align-items: flex-end;
 justify-content: space-between;
 margin-bottom: 16px;
 gap: 16px;
}
.section-eyebrow {
 font-size: 11px;
 font-weight: 700;
 text-transform: uppercase;
 letter-spacing: .14em;
 color: var(--brand-mid);
 margin-bottom: 4px;
}
.section-title {
 font-size: 22px;
 font-weight: 800;
 color: var(--text);
}
.section-subtitle {
 font-size: 13px;
 color: var(--text-dim);
}

/* KPI Cards */
.kpi-grid {
 display: grid;
 grid-template-columns: repeat(3, 1fr);
 gap: 14px;
}
.kpi-card {
 background: var(--bg-card);
 border: 1px solid var(--stroke);
 border-radius: var(--radius-lg);
 padding: 20px 22px;
 box-shadow: var(--shadow-card);
 transition: transform .2s, box-shadow .2s;
 position: relative;
 overflow: hidden;
}
.kpi-card::before {
 content: '';
 position: absolute;
 top: 0; left: 0; right: 0;
 height: 3px;
 background: linear-gradient(90deg, var(--brand), var(--brand-light));
 opacity: 0;
 transition: opacity .2s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 12px 36px rgba(26,90,70,.14); }
.kpi-card:hover::before { opacity: 1; }
.kpi-card.kpi-danger {
 border-color: rgba(220,38,38,.18);
 background: linear-gradient(160deg, rgba(220,38,38,.04), var(--bg-card));
}
.kpi-card.kpi-danger::before { background: linear-gradient(90deg, var(--accent-red), var(--accent-amber)); opacity: 1; }
.kpi-top {
 display: flex;
 align-items: center;
 justify-content: space-between;
 margin-bottom: 12px;
}
.kpi-icon { font-size: 22px; }
.kpi-label {
 font-size: 11px;
 font-weight: 700;
 text-transform: uppercase;
 letter-spacing: .12em;
 color: var(--text-dim);
 margin-bottom: 6px;
}
.kpi-value {
 font-size: 32px;
 font-weight: 900;
 color: var(--brand);
 line-height: 1;
 letter-spacing: -.02em;
 font-family: 'DM Mono', monospace;
}
.kpi-caption {
 font-size: 12px;
 color: var(--text-dim);
 margin-top: 8px;
}

/* Status Banner */
.status-banner {
 display: flex;
 align-items: center;
 justify-content: space-between;
 padding: 20px 28px;
 border-radius: var(--radius-lg);
 border: 1px solid var(--stroke);
 background: var(--bg-card);
 margin-bottom: 24px;
 box-shadow: var(--shadow-card);
}
.status-good { border-left: 4px solid var(--accent-emerald); }
.status-warn { border-left: 4px solid var(--accent-amber); }
.status-danger{ border-left: 4px solid var(--accent-red); }
.status-idle { border-left: 4px solid var(--muted); }
.status-left { display: flex; align-items: center; gap: 18px; }
.status-dot {
 width: 10px; height: 10px;
 border-radius: 50%;
 flex-shrink: 0;
 animation: pulse-dot 2s ease-in-out infinite;
}
.status-good .status-dot { background: var(--accent-emerald); box-shadow: 0 0 0 6px rgba(5,150,105,.18); }
.status-warn .status-dot { background: var(--accent-amber);  box-shadow: 0 0 0 6px rgba(217,119,6,.18); }
.status-danger .status-dot{ background: var(--accent-red);   box-shadow: 0 0 0 6px rgba(220,38,38,.18); }
.status-idle .status-dot { background: var(--muted); }
@keyframes pulse-dot {
 0%, 100% { opacity: 1; }
 50%    { opacity: .5; }
}
.status-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .12em; color: var(--text-dim); }
.status-title { font-size: 22px; font-weight: 800; color: var(--text); margin-top: 2px; }
.status-note { font-size: 13px; color: var(--text-dim); margin-top: 2px; }
.status-icon { font-size: 36px; }

/* Chart Grid Layouts */
.chart-grid-2 {
 display: grid;
 grid-template-columns: repeat(2, 1fr);
 gap: 16px;
 margin-bottom: 16px;
}
.chart-grid-3 {
 display: grid;
 grid-template-columns: repeat(3, 1fr);
 gap: 16px;
 margin-bottom: 16px;
}

/* Chart Cards */
.chart-card {
 background: var(--bg-card);
 border: 1px solid var(--stroke);
 border-radius: var(--radius-lg);
 padding: 20px 22px;
 box-shadow: var(--shadow-card);
 min-height: 240px;
 display: flex;
 flex-direction: column;
 gap: 14px;
}
.chart-header {
 display: flex;
 align-items: center;
 gap: 8px;
 flex-wrap: wrap;
}
.chart-icon { font-size: 16px; }
.chart-title {
 font-size: 14px;
 font-weight: 700;
 color: var(--text);
 flex: 1;
}
.chart-note-inline {
 font-size: 11px;
 color: var(--text-dim);
 margin-left: auto;
}
.chart-badge {
 font-size: 12px;
 color: var(--brand);
 background: var(--brand-pale);
 border: 1px solid rgba(26,90,70,.12);
 border-radius: 999px;
 padding: 3px 10px;
 margin-left: auto;
}
.chart-footer {
 display: flex;
 gap: 16px;
 font-size: 12px;
 color: var(--text-dim);
 margin-top: auto;
 padding-top: 8px;
 border-top: 1px solid var(--stroke);
}
.chart-footer b { color: var(--brand); }
.empty-chart {
 flex: 1;
 display: flex;
 flex-direction: column;
 align-items: center;
 justify-content: center;
 gap: 10px;
 color: var(--text-dim);
 font-size: 13px;
 border: 1px dashed var(--stroke-bright);
 border-radius: 14px;
 text-align: center;
 padding: 32px;
 background: var(--bg-base);
}
.empty-icon { font-size: 32px; opacity: .4; }

/* Bar Chart */
.bar-chart {
 display: flex;
 flex-direction: column;
 gap: 11px;
 flex: 1;
}
.bar-row {
 display: grid;
 grid-template-columns: 1fr auto;
 grid-template-rows: auto auto;
 gap: 4px 8px;
 animation: fade-up .4s ease both;
}
@keyframes fade-up {
 from { opacity: 0; transform: translateY(6px); }
 to  { opacity: 1; transform: none; }
}
.bar-label-wrap {
 display: flex;
 justify-content: space-between;
 align-items: center;
 grid-column: 1;
}
.bar-label {
 font-size: 12px;
 color: var(--text-dim);
 white-space: nowrap;
 overflow: hidden;
 text-overflow: ellipsis;
 max-width: 170px;
}
.bar-pct {
 font-size: 11px;
 color: var(--muted);
}
.bar-track {
 height: 8px;
 border-radius: 999px;
 background: var(--brand-pale);
 overflow: hidden;
 grid-column: 1;
 align-self: center;
}
.bar-fill {
 height: 100%;
 border-radius: 999px;
 transition: width .6s cubic-bezier(.16,1,.3,1);
}
.bar-value {
 font-size: 13px;
 font-weight: 700;
 color: var(--brand);
 text-align: right;
 grid-column: 2;
 grid-row: 1 / 3;
 align-self: center;
 font-family: 'DM Mono', monospace;
}

/* SVG Line Chart */
.svg-wrap {
 width: 100%;
 border-radius: 12px;
 overflow: hidden;
 background: var(--bg-base);
 border: 1px solid var(--stroke);
 flex: 1;
}
svg { width: 100%; height: auto; display: block; }
.axis { stroke: var(--muted); stroke-width: 1; }
.grid-line { stroke: rgba(26,90,70,.07); stroke-width: 1; stroke-dasharray: 3 3; }
.trend-line {
 fill: none;
 stroke: var(--brand-mid);
 stroke-width: 2.8;
 stroke-linecap: round;
 stroke-linejoin: round;
}
.data-circle {
 fill: var(--brand-light);
 stroke: #fff;
 stroke-width: 2;
}
.axis-label { fill: var(--muted); font-size: 10px; font-weight: 500; }
.donut-center-num { fill: var(--brand); font-size: 22px; font-weight: 800; }
.donut-center-lbl { fill: var(--text-dim); font-size: 11px; }

/* Donut Chart */
.donut-wrap {
 display: flex;
 align-items: center;
 gap: 24px;
 flex: 1;
}
.donut-svg-wrap { flex-shrink: 0; }
.donut-legend {
 display: flex;
 flex-direction: column;
 gap: 12px;
 flex: 1;
}
.legend-row {
 display: flex;
 align-items: center;
 gap: 8px;
 font-size: 13px;
}
.legend-swatch {
 width: 10px; height: 10px;
 border-radius: 3px;
 flex-shrink: 0;
}
.legend-label { flex: 1; color: var(--text-dim); }
.legend-count { font-weight: 700; color: var(--text); font-family: 'DM Mono', monospace; }
.legend-pct  { font-size: 11px; color: var(--muted); width: 36px; text-align: right; }
.donut-dominant {
 margin-top: 6px;
 font-size: 12px;
 color: var(--brand);
 padding: 8px 12px;
 background: var(--brand-pale);
 border: 1px solid rgba(26,90,70,.12);
 border-radius: 10px;
}
.donut-dominant b { color: var(--brand); }

/* History Table */
#history-table {
 background: var(--bg-card) !important;
 border: 1px solid var(--stroke-bright) !important;
 border-radius: var(--radius-lg) !important;
 overflow: hidden !important;
}
#history-table label { color: var(--text) !important; font-weight: 700 !important; }
#history-table .wrap.svelte-1ipelgc,
#history-table .table-wrap,
#history-table .table-container {
 background: transparent !important;
 overflow-x: auto !important;
}
#history-table table {
 min-width: 1100px !important;
 table-layout: auto !important;
}
#history-table th,
#history-table td {
 white-space: nowrap !important;
 word-break: normal !important;
 overflow-wrap: normal !important;
 font-size: 13px !important;
}
#history-table th {
 font-weight: 800 !important;
 color: var(--text) !important;
}

/* Status indicators */
#dashboard-status, #login-status { min-height: 10px; }

/* Responsive */
@media (max-width: 1200px) {
 .kpi-grid    { grid-template-columns: repeat(2, 1fr); }
 .chart-grid-3  { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 900px) {
 .hero-banner  { grid-template-columns: 1fr; }
 .hero-right   { display: none; }
 .chart-grid-2,
 .chart-grid-3  { grid-template-columns: 1fr; }
}
@media (max-width: 600px) {
 .main      { padding: 16px 10px 40px !important; }
 .kpi-grid    { grid-template-columns: 1fr; }
 .hero-title   { font-size: 28px; }
 .bar-row    { grid-template-columns: 1fr; }
 .bar-value   { grid-column: 1; text-align: left; }
}
"""


# -----------------------------------------------------------------------------
# Gradio UI
# -----------------------------------------------------------------------------


MEDIBUDDY_THEME = gr.themes.Base(
  primary_hue="emerald",
  secondary_hue="teal",
  neutral_hue="slate",
  font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"],
  font_mono=[gr.themes.GoogleFont("DM Mono"), "monospace"],
).set(
  body_background_fill="#e8eef0",
  background_fill_primary="#ffffff",
  background_fill_secondary="#f0f5f3",
  border_color_primary="rgba(26,90,70,0.18)",
  color_accent_soft="rgba(45,171,133,0.10)",
  button_primary_background_fill="linear-gradient(135deg,#1a5a46,#2dab85)",
  button_primary_text_color="#ffffff",
  button_secondary_background_fill="#ffffff",
  button_secondary_border_color="rgba(26,90,70,0.20)",
  button_secondary_text_color="#1a5a46",
  input_background_fill="#e8eef0",
)

# NOTE: In Gradio 4.x, `theme` and `css` MUST be passed to gr.Blocks(), not to .launch().
with gr.Blocks(title="MediBuddy AI Admin", theme=MEDIBUDDY_THEME, css=ADMIN_CSS) as admin_demo:
  is_logged_in = gr.State(False)

  with gr.Group(visible=True, elem_id="login-shell") as login_view:
    gr.HTML("""
      <div class="login-card">
       <div class="login-logo"></div>
       <div class="eyebrow">Secure Admin Access</div>
       <h1>MediBuddy AI Admin</h1>
       <p>Sign in to access the analytics dashboard, analysis history, risk monitoring signals, and system-wide activity trends.</p>
      </div>
    """)
    admin_password = gr.Textbox(
      label="Admin password",
      type="password",
      placeholder="Enter admin password",
      container=True,
      elem_id="login-input",
    )
    login_btn = gr.Button("Login to Dashboard ", variant="primary", elem_id="login-btn")
    login_status = gr.HTML(elem_id="login-status")

  with gr.Group(visible=False, elem_id="dashboard-group") as dashboard_view:
    with gr.Row(elem_id="top-controls"):
      refresh_btn = gr.Button(" Refresh Dashboard", variant="primary", elem_id="refresh-btn")
      logout_btn = gr.Button("Logout", variant="secondary", elem_id="logout-btn")
    dashboard_status = gr.HTML(elem_id="dashboard-status")

    with gr.Tabs(elem_id="dash-tabs") as tabs:
      with gr.TabItem(" Overview", id="overview"):
        admin_html = gr.HTML(elem_id="dashboard-shell")

      with gr.TabItem(" Analysis History", id="history"):
        history_table = gr.Dataframe(
          label="Recent Analysis History",
          interactive=False,
          wrap=False,
          elem_id="history-table",
        )

  # Wire up events
  login_btn.click(
    handle_admin_login,
    inputs=[admin_password],
    outputs=[is_logged_in, login_view, dashboard_view, admin_html, history_table, login_status],
  )
  admin_password.submit(
    handle_admin_login,
    inputs=[admin_password],
    outputs=[is_logged_in, login_view, dashboard_view, admin_html, history_table, login_status],
  )
  refresh_btn.click(
    refresh_admin_dashboard,
    inputs=[is_logged_in],
    outputs=[admin_html, history_table, dashboard_status],
  )
  logout_btn.click(
    logout_admin,
    inputs=None,
    outputs=[is_logged_in, login_view, dashboard_view, admin_html, history_table, login_status],
  )

