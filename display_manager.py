# voice_assistant/display_manager.py
from __future__ import annotations
import textwrap
import time
from typing import List
from PIL import ImageFont

from .config import config

def _log(msg: str):  # very chatty on purpose while we debug
    print(f"[display] {msg}")

try:
    from luma.core.interface.serial import i2c
    from luma.oled.device import sh1107
    from luma.core.render import canvas
    _HAVE_LUMA = True
except Exception as e:
    _HAVE_LUMA = False
    _log(f"luma import failed: {e}")

class DisplayManager:
    """
    Simple SH1107 128x128 text renderer.
    - Assumes I2C address 0x3C on i2c-1 (can be changed via config)
    - Provides: splash(), show_message(title, body), show_user(), show_bot(), status(), idle()
    """
    def __init__(self):
        self.enabled = bool(getattr(config, "display_enabled", True))
        self.addr    = int(getattr(config, "oled_addr", 0x3C))
        self.port    = int(getattr(config, "oled_i2c_port", 1))
        self.width   = int(getattr(config, "oled_width", 128))
        self.height  = int(getattr(config, "oled_height", 128))
        self.device  = None
        self.font    = ImageFont.load_default()

        if not self.enabled:
            _log("disabled by config")
            return
        if not _HAVE_LUMA:
            _log("luma not available")
            return

        try:
            serial = i2c(port=self.port, address=self.addr)
            # SH1107 wants explicit width/height
            self.device = sh1107(serial, width=self.width, height=self.height)
            _log(f"OK: SH1107 {self.width}x{self.height} @0x{self.addr:02X} on i2c-{self.port}")
        except Exception as e:
            _log(f"init failed: {e}")
            self.device = None

    # --- helpers ---
    def _wrap(self, text: str, width_chars: int, max_lines: int) -> List[str]:
        import textwrap
        text = (text or "").replace("\n", " ").strip()
        return textwrap.wrap(text, width=width_chars)[:max_lines]

    def _render(self, title: str, body: str = ""):
        if not self.device:
            _log(f"[console] {title} :: {body}")  # fallback so you still see output
            return
        with canvas(self.device) as draw:
            # For the default bitmap font, ~21–22 chars/line fits across 128px.
            draw.text((2, 2), (title or "")[:22], font=self.font, fill=255)
            for i, ln in enumerate(self._wrap(body, width_chars=22, max_lines=7)):
                draw.text((2, 18 + i * 14), ln, font=self.font, fill=255)
        time.sleep(0.10)

    # --- public API ---
    def show_message(self, title: str, body: str = ""):
        self._render(title, body)

    def splash(self, title: str = "Optimus", body: str = "Voice system ready"):
        self._render(title, body)

    def show_user(self, text: str):
        self._render("You:", text)

    def show_bot(self, text: str):
        self._render("Bot:", text)

    def status(self, label: str):
        self._render("Status", label)

    def idle(self):
        self._render("Idle", "Say: Hej robot")
