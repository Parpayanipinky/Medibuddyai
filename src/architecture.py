"""Professional architecture layer for teacher review.

This file documents the object-oriented design direction of the project.
The current Gradio UI uses the legacy notebook workflow, while these classes
provide clean extension points for future AI inference, storage, and testing.
"""

class ReportAnalyzer:
    """Future home for OCR, parsing, classification, risk scoring, and PDF export."""
    def analyze(self, file_path):
        raise NotImplementedError("The current implementation lives in src.user_app for compatibility.")

class UserInterface:
    """Future home for user-facing Gradio components."""
    pass

class AdminAnalytics:
    """Future home for admin analytics, usage dashboards, and audit views."""
    pass

