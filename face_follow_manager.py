from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from .display_manager import DisplayManager


class FaceFollowManager:
    """
    Lightweight manager that runs the existing follow_face.py script as a
    background subprocess and streams its stdout to update the display.

    We avoid refactoring the tracker; this wrapper provides start/stop and
    optional status updates.
    """

    def __init__(self, display: Optional[DisplayManager] = None, script_path: Optional[Path] = None):
        self.display = display
        # Resolve default script path: only support co-located file
        # work/voice_assistant/follow_face.py
        if script_path is None:
            script_path = Path(__file__).resolve().parent / "follow_face.py"
        self.script_path = Path(script_path)

        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    def start(self) -> bool:
        if self._proc and self._proc.poll() is None:
            return True
        if not self.script_path.exists():
            if self.display:
                self.display.status("face: script missing")
            return False

        try:
            self._stop_evt.clear()
            cmd = [sys.executable, "-u", str(self.script_path)]
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if self.display:
                self.display.status("face: starting…")

            # Detect immediate failures (missing deps, import errors, etc.)
            time.sleep(0.4)
            if self._proc.poll() is not None:
                output = ""
                if self._proc.stdout:
                    try:
                        output = self._proc.stdout.read()
                    except Exception:
                        output = ""
                msg = f"[face-follow] process exited early (code={self._proc.returncode})"
                print(msg)
                if output:
                    print(output.strip())
                if self.display:
                    self.display.status("face: start failed")
                self.stop()
                return False

            # Reader thread to consume output and update display
            self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
            self._reader_thread.start()
            return True
        except Exception as e:
            if self.display:
                self.display.status(f"face: start failed")
            print(f"[face-follow] start failed: {e}")
            return False

    def _read_stdout(self):
        last_update = 0.0
        try:
            if not self._proc or not self._proc.stdout:
                return
            for line in self._proc.stdout:
                if self._stop_evt.is_set():
                    break
                line = (line or "").strip()
                if not line:
                    continue
                print(f"[face-follow] {line}")
                # Throttle display updates to avoid OLED spam
                now = time.time()
                if now - last_update < 0.25:
                    continue
                last_update = now

                # Heuristics based on script output
                if line.startswith("Face dx="):
                    # Example: Face dx= +12 dy=  -3 AZ:+120.0 Hz ALT:-80.0 Hz
                    msg = "face: tracking"
                    if self.display:
                        self.display.status(msg)
                elif line.startswith("No face"):
                    if self.display:
                        self.display.status("face: none")
                elif line.lower().startswith("drivers will"):
                    if self.display:
                        self.display.status("face: init")
                elif line.lower().startswith("stopping") or line.lower().startswith("drivers disabled"):
                    if self.display:
                        self.display.status("face: stopped")
        except Exception as e:
            print(f"[face-follow] reader error: {e}")

    def stop(self, kill_after: float = 2.0):
        self._stop_evt.set()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=kill_after)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            except Exception as e:
                print(f"[face-follow] stop error: {e}")
        self._proc = None
        self._reader_thread = None

    def is_running(self) -> bool:
        return bool(self._proc and self._proc.poll() is None)
