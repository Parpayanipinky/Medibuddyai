"""Run the MediBuddy AI user app.

Admin panel is intentionally separated into admin_app.py so the polished user UI
is not wrapped in a parent TabbedInterface that can break layout/CSS.
"""

import os
from src.userapp import demo

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.queue()
    demo.launch(server_name=server_name, server_port=server_port, share=False, show_error=True)

