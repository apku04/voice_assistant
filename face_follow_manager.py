from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
        self._state_lock = threading.Lock()
        self._latest_behavior: Dict[str, Any] = {
            "mood": None,
            "left_hand": None,
            "right_hand": None,
            "last_event": None,
            "mood_features": None,
        }

    def start(self) -> bool:
        if self._proc and self._proc.poll() is None:
            return True
        if not self.script_path.exists():
            if self.display:
                self.display.status("face: script missing")
            return False

        try:
            self._stop_evt.clear()
            # Use unbuffered python to get timely stdout
            cmd = [sys.executable, "-u", str(self.script_path)]
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            # Reader thread to consume output and update display
            self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
            self._reader_thread.start()
            if self.display:
                self.display.status("face: starting…")
            return True
        except Exception as e:
            if self.display:
                self.display.status(f"face: start failed")
            print(f"[face-follow] start failed: {e}")
            return False

    def _read_stdout(self):
        last_update = 0.0
        behavior_re = re.compile(r"^(Mood|Left hand|Right hand):\s+([\w\s]+)\s*\((\d+\.\d+)\)")
        mood_features_re = re.compile(r"^Mood features:\s+(.+)$")
        try:
            if not self._proc or not self._proc.stdout:
                return
            for line in self._proc.stdout:
                if self._stop_evt.is_set():
                    break
                line = (line or "").strip()
                if not line:
                    continue

                # Capture behavior lines regardless of throttling
                match = behavior_re.match(line)
                if match:
                    label, state, conf = match.groups()
                    key = {
                        "Mood": "mood",
                        "Left hand": "left_hand",
                        "Right hand": "right_hand",
                    }.get(label)
                    if key:
                        with self._state_lock:
                            self._latest_behavior[key] = (state.strip(), float(conf))
                            self._latest_behavior["last_event"] = (key, time.time())
                    continue

                feature_match = mood_features_re.match(line)
                if feature_match:
                    payload: Dict[str, float | bool] = {}
                    for token in feature_match.group(1).split():
                        if "=" not in token:
                            continue
                        name, raw_val = token.split("=", 1)
                        value = raw_val.rstrip(",")
                        if name == "face_visible":
                            payload[name] = value.lower() in {"true", "1", "yes", "on"}
                        else:
                            try:
                                payload[name] = float(value)
                            except ValueError:
                                continue
                    if payload:
                        with self._state_lock:
                            self._latest_behavior["mood_features"] = dict(payload)
                            self._latest_behavior["last_event"] = ("mood_features", time.time())
                    continue

                # Throttle display updates for non-behavior messages
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
                    with self._state_lock:
                        self._latest_behavior["last_event"] = ("face", time.time())
                elif line.lower().startswith("drivers will"):
                    if self.display:
                        self.display.status("face: init")
                elif line.lower().startswith("stopping") or line.lower().startswith("drivers disabled"):
                    if self.display:
                        self.display.status("face: stopped")
                # Also mirror to console for debugging
                #print(f"[face-follow] {line}")
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

    def get_behavior_snapshot(self) -> Dict[str, Any]:
        with self._state_lock:
            return dict(self._latest_behavior)
