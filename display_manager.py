# voice_assistant/display_manager.py
import re
import time
import threading

from PIL import ImageFont
from luma.core.render import canvas

from .config import config


class DisplayManager:
    _lock = threading.Lock()

    def __init__(self):
        self.device = None
        self.font = ImageFont.load_default()
        self.margin_x = 2
        self.margin_y = 2
        self.line_gap = 2  # extra pixels between lines

        self._initialize_device()

    def _initialize_device(self):
        """Set up the OLED device if the hardware stack is available."""
        if not getattr(config, "display_enabled", True):
            print("[display] disabled via config; using console output only.")
            return

        try:
            from luma.core.interface.serial import i2c
            from luma.oled.device import sh1107
        except ImportError:
            print("[display] luma oled libraries missing; using console output only.")
            return

        try:
            serial = i2c(port=1, address=config.oled_addr)
            self.device = sh1107(serial_interface=serial)
        except Exception as exc:
            print(f"[display] failed to initialize OLED ({exc}); using console output only.")
            self.device = None

    def _text_width(self, s: str) -> int:
        # Pillow-safe width measurement
        bbox = self.font.getbbox(s or "")
        return (bbox[2] - bbox[0]) if bbox else 0

    def _wrap_pixels(self, text: str, max_px: int, max_lines: int):
        """
        Wrap by pixel width. Handles long words/URLs by breaking them safely.
        Returns list of lines (max max_lines). Last line may be ellipsized.
        """
        if not text:
            return []
        # normalize whitespace & strip control chars
        t = re.sub(r"[\r\t]+", " ", text).replace("\n", " ").strip()
        words = t.split(" ")
        lines = []
        cur = ""

        def fits(s: str) -> bool:
            return self._text_width(s) <= max_px

        i = 0
        while i < len(words):
            w = words[i]
            candidate = w if not cur else cur + " " + w
            if fits(candidate):
                cur = candidate
                i += 1
                continue
            # current word doesn't fit appended; if nothing in cur, we must break the word
            if not cur:
                # break the long word at a character that fits
                chunk = ""
                for ch in w:
                    if fits(chunk + ch):
                        chunk += ch
                    else:
                        break
                if not chunk:  # extreme case: single char too wide; force 1 char
                    chunk = w[0]
                lines.append(chunk)
                # push the remainder back
                words[i] = w[len(chunk):]
                if len(lines) >= max_lines:
                    break
                continue
            # push cur as a line and retry the same word on a new line
            lines.append(cur)
            cur = ""
            if len(lines) >= max_lines:
                break

        if cur and len(lines) < max_lines:
            lines.append(cur)

        # Ellipsize last line if clipped
        if i < len(words) and len(lines) == max_lines:
            last = lines[-1]
            ell = "…"
            while last and not fits(last + ell):
                last = last[:-1]
            lines[-1] = (last + ell) if last else ell
        return lines

    def _render(self, title: str, body: str = ""):
        if not self.device:
            print(f"[display] (console) {title} :: {body}")
            return

        W = getattr(self.device, "width", 128)
        H = getattr(self.device, "height", 128)
        x = self.margin_x
        y = self.margin_y

        # layout: title (1 line) + wrapped body
        title = (title or "").strip()
        body  = (body or "").strip()

        max_body_px = W - 2 * self.margin_x
        title_h = (self.font.getbbox("A")[3] - self.font.getbbox("A")[1])
        line_h  = title_h + self.line_gap

        # Leave space for title on first line; body starts below
        with canvas(self.device) as draw:
            if title:
                draw.text((x, y), title, font=self.font, fill=255)
                y += line_h

            # compute how many lines fit in remaining height
            remaining_px = max(0, H - y - self.margin_y)
            max_lines = max(1, remaining_px // line_h)

            for ln in self._wrap_pixels(body, max_px=max_body_px, max_lines=max_lines):
                draw.text((x, y), ln, font=self.font, fill=255)
                y += line_h

        # small post-flush delay so immediate redraws don’t wipe it before you see it
        time.sleep(0.08)

    # public API
    def show_message(self, title: str, body: str = ""):
        with self._lock:
            self._render(title, body)

    def splash(self, title: str = "Optimus", body: str = "Voice system ready"):
        self.show_message(title, body)

    def show_user(self, text: str):
        self.show_message("You:", text)

    def show_bot(self, text: str):
        self.show_message("Optimus", text)

    def status(self, label: str):
        self.show_message("Status", label)

    def idle(self):
        self.show_message("Idle", "Say: Hej robot")
