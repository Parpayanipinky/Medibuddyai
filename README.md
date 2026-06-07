---
title: Medibuddyai
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# MediBuddy AI End-to-End Application

Final version includes:

- User-side lab report analysis
- User-side X-ray analysis
- AI explanation and report-aware chatbot
- OCR quality scoring
- Risk level and simple health suggestions
- PDF/JSON downloads
- User-side Health Progress Comparison Dashboard
- Separate Admin Dashboard for monitoring analytics/history
- Local JSONL logging for admin dashboard
- Hugging Face-ready project structure

## Hugging Face deployment

This project is configured for a Hugging Face Gradio Space.

Required secret:

- GROQ_API_KEY

Optional secret:

- MEDIBUDDY_ADMIN_PASSWORD

## Medical safety note

This tool is for educational support only and does not replace a doctor or radiologist.
