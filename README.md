# MediBuddy AI End-to-End Application

Final version includes:

- User-side lab report analysis
- User-side X-ray analysis
- AI explanation and report-aware chatbot
- OCR quality scoring
- Risk level and simple health suggestions
- PDF/JSON downloads
- User-side Health Progress Comparison Dashboard
  - Select previous saved lab report + upload current report
  - Upload both lab reports manually
  - Basic side-by-side X-ray history comparison
- Separate Admin Dashboard for monitoring analytics/history
- Local JSONL logging for admin dashboard
- Hugging Face-ready project structure

## Run user app locally

```bash
cd /d "D:\BNU-TOPUP\Zip -file\medibuddy_ai_end_to_end_admin"
venv\Scripts\activate
set GROQ_API_KEY=your_key_here
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

## Run admin panel separately

Open a second terminal:

```bash
cd /d "D:\BNU-TOPUP\Zip -file\medibuddy_ai_end_to_end_admin"
venv\Scripts\activate
set MEDIBUDDY_ADMIN_PASSWORD=admin123
python admin_app.py
```

Open:

```text
http://127.0.0.1:7861
```

## Comparison dashboard flow

On Page 2, select **Compare Reports**.

You can compare in three ways:

1. Select a previously saved lab report and upload the current/new lab report.
2. Upload both previous and current lab reports manually.
3. Upload previous and current X-ray images for a basic visual/history comparison.

Lab comparison shows improved values, needs-attention values, current risk level, a graph, detailed table, explanation, motivational quote, and medical disclaimer.

## Hugging Face deployment

Upload all project files to a Gradio Space and add these secrets:

- `GROQ_API_KEY`
- `MEDIBUDDY_ADMIN_PASSWORD` optional

For Hugging Face, `app.py` launches the user app. The admin panel is separated in `admin_app.py` to keep the user interface stable.

## Medical safety note

This tool is for educational support only and does not replace a doctor or radiologist.
