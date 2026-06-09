"""Run the MediBuddy AI admin dashboard separately."""

import os
from src.admin_panel import ADMIN_CSS, MEDIBUDDY_THEME, admin_demo

if __name__ == "__main__":
    server_name = os.getenv("ADMIN_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("ADMIN_SERVER_PORT", "7861"))

    admin_demo.queue()
    admin_demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_error=True,
        theme=MEDIBUDDY_THEME,
        css=ADMIN_CSS,
    )
