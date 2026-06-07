"""Run the MediBuddy AI admin dashboard separately."""

import os
import gradio as gr
from src.admin_panel import ADMIN_CSS, admin_demo

# NOTE: In Gradio 4.x, `theme` and `css` belong in gr.Blocks(), NOT in .launch().
# They are applied at module level inside admin_panel.py via the THEME / ADMIN_CSS constants.

if __name__ == "__main__":
    server_name = os.getenv("ADMIN_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("ADMIN_SERVER_PORT", "7861"))

    admin_demo.queue()
    admin_demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_error=True,
    )

