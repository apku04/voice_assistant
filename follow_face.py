#!/usr/bin/env python3
"""
Face-follow tracker for Raspberry Pi 5 (headless).
USB webcam (OpenCV/V4L2) + two TMC2209 steppers (Azimuth+Altitude) via gpiozero+lgpio.
Includes MJPEG web streaming for live debugging.

Detectors (auto):
1) YuNet (ONNX, cv2.FaceDetectorYN)  -> best
2) DNN (res10 SSD Caffe)             -> good
3) Haar cascade                      -> basic

Pins:
- Motor A (Azimuth / horizontal): DIR=4,  STEP=25, EN=24
- Motor B (Altitude / vertical):   DIR=12, STEP=6,  EN=5
"""

import math
import os
import time
import threading
from typing import Optional

import cv2
import numpy as np
from gpiozero import DigitalOutputDevice, Device
from gpiozero.pins.lgpio import LGPIOFactory

# ---------------- Camera config ----------------
CAP_INDEX = 0
FRAME_W, FRAME_H = 640, 480
FPS = 30

# Model files (override via env if desired)
YUNET_ONNX = os.getenv("YUNET_ONNX", "/usr/share/opencv4/face_detection_yunet_2023mar.onnx")
DNN_PROTO = os.getenv("DNN_PROTO", "/usr/share/opencv4/deploy.prototxt")
DNN_MODEL = os.getenv("DNN_MODEL", "/usr/share/opencv4/res10_300x300_ssd_iter_140000.caffemodel")
HAAR_XML = os.getenv("HAAR_XML", "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")

# ---------------- Control tuning ----------------
DEADBAND_PX = 16            # no movement if |error| <= deadband
K_P = 3.0                   # pixel error -> step rate (Hz)
MIN_RATE_HZ = 40            # minimal step rate when outside deadband
MAX_RATE_HZ = 1200          # clamp for safety/smoothness
STEP_PULSE_US = 250         # STEP high/low microseconds (>= driver min pulse)
SMOOTHING = 0.35            # EWMA; higher = smoother/slower
MIN_FACE_AREA_RATIO = 0.004 # ignore blobs smaller than 0.4% of frame


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# Optional behavioural perception (mood + hands)
ENABLE_BEHAVIOR = _env_bool("FF_BEHAVIOR", True)
BEHAVIOR_INTERVAL = max(0.2, _env_float("FF_BEHAVIOR_INTERVAL", 0.6))

# Direction polarity (flip if motion is reversed for your rig)
AZ_DIR_POS_RIGHT = False    # True => dx>0 drives DIR=ON as "right"
ALT_DIR_POS_DOWN = False    # True => dy>0 (face below) drives DIR=ON as "down"

# Auto-disable behaviour
FACE_LOSS_TIMEOUT = 1.5     # seconds without face → disable drivers
IDLE_TIMEOUT_SEC = 2.0      # seconds with 0 rate (even if face centered) → disable

# ---------------- HTTP MJPEG streamer (Flask) ----------------
import io
from flask import Flask, Response

STREAM_HOST = "0.0.0.0"  # visit http://<pi-ip>:5000/
STREAM_PORT = 5000
STREAM_QUALITY = 70       # JPEG quality (lower = faster)
STREAM_WIDTH = 640        # resize width for stream; 0 = keep original

app = Flask(__name__)
_latest_jpeg = None
_latest_lock = threading.Lock()


def _set_stream_frame(bgr_frame):
    """Convert BGR frame to JPEG and store for streamer."""
    global _latest_jpeg
    if STREAM_WIDTH and bgr_frame.shape[1] != STREAM_WIDTH:
        scale = STREAM_WIDTH / float(bgr_frame.shape[1])
        bgr_frame = cv2.resize(
            bgr_frame,
            (STREAM_WIDTH, int(bgr_frame.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_QUALITY])
    if ok:
        with _latest_lock:
            _latest_jpeg = buf.tobytes()


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        with _latest_lock:
            jpg = _latest_jpeg
        if jpg is None:
            time.sleep(0.03)
            continue
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.03)


@app.route("/")
def index():
    return (
        "<html><head><title>Face Debug Stream</title></head>"
        "<body style='margin:0;background:#111;color:#ddd;font-family:sans-serif'>"
        "<div style='padding:8px'>MJPEG stream — <code>/video</code></div>"
        "<img src='/video' style='width:100%;height:auto'/>"
        "</body></html>"
    )


@app.route("/video")
def video():
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------- GPIO setup ----------------
Device.pin_factory = LGPIOFactory()

A_DIR, A_STEP, A_EN = 4, 25, 24   # Azimuth
B_DIR, B_STEP, B_EN = 12, 6, 5    # Altitude

A_dir = DigitalOutputDevice(A_DIR)
A_en = DigitalOutputDevice(A_EN, active_high=False, initial_value=True)  # EN low = enabled
A_step = DigitalOutputDevice(A_STEP)

B_dir = DigitalOutputDevice(B_DIR)
B_en = DigitalOutputDevice(B_EN, active_high=False, initial_value=True)
B_step = DigitalOutputDevice(B_STEP)


def enable_drivers(enable: bool):
    # active_high=False ⇒ on() pulls EN LOW (enable), off() pulls EN HIGH (disable)
    if enable:
        A_en.on()
        B_en.on()
    else:
        A_en.off()
        B_en.off()


# ---------------- Stepper worker ----------------
class StepperWorker(threading.Thread):
    def __init__(self, name, dir_pin: DigitalOutputDevice, step_pin: DigitalOutputDevice, rate_hz_getter, direction_getter):
        super().__init__(name=name, daemon=True)
        self.dir_pin = dir_pin
        self.step_pin = step_pin
        self.rate_hz_getter = rate_hz_getter
        self.direction_getter = direction_getter
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        next_toggle = time.monotonic()
        phase_high = False
        while not self._stop_event.is_set():
            rate = self.rate_hz_getter()
            if rate <= 0.0:
                time.sleep(0.002)
                continue

            # set direction each cycle (cheap and avoids races)
            self.dir_pin.on() if self.direction_getter() else self.dir_pin.off()

            # toggle STEP with min pulse width honored
            half_period = 0.5 / rate
            now = time.monotonic()
            if now < next_toggle:
                time.sleep(min(next_toggle - now, 0.002))
                continue

            if phase_high:
                self.step_pin.off()
                phase_high = False
            else:
                self.step_pin.on()
                phase_high = True

            # ensure pulse width isn't shorter than STEP_PULSE_US
            next_toggle = now + max(half_period, STEP_PULSE_US / 1_000_000.0)


# ---------------- Shared state (vision → motor threads) ----------------
class AxisControl:
    def __init__(self):
        self._rate_hz = 0.0
        self._dir_positive = True
        self._lock = threading.Lock()

    def set(self, rate_hz: float, dir_positive: bool):
        with self._lock:
            self._rate_hz = max(0.0, min(rate_hz, MAX_RATE_HZ))
            self._dir_positive = dir_positive

    def get_rate(self) -> float:
        with self._lock:
            return self._rate_hz

    def get_dir(self) -> bool:
        with self._lock:
            return self._dir_positive


class BehaviorAnalyzer:
    """Optional MediaPipe-powered mood + hand cue inference."""

    def __init__(self, interval_sec: float):
        self.enabled = ENABLE_BEHAVIOR
        self.interval_sec = max(0.2, float(interval_sec))
        self._last_run = 0.0
        self._holistic = None
        self._mp = None
        self._drawing = None
        self._drawing_styles = None
        self._state_lock = threading.Lock()
        self._state: dict[str, Optional[tuple[str, float]]] = {
            "mood": None,
            "left": None,
            "right": None,
        }
        self._last_face_landmarks = None
        self._last_left_hand_landmarks = None
        self._last_right_hand_landmarks = None

        if not self.enabled:
            print("[behavior] Disabled (FF_BEHAVIOR=0)")
            return

        try:
            import mediapipe as mp  # type: ignore

            self._mp = mp
            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                refine_face_landmarks=True,
            )
            self._drawing = mp.solutions.drawing_utils
            self._drawing_styles = mp.solutions.drawing_styles
            print("[behavior] MediaPipe holistic enabled")
        except Exception as exc:  # noqa: BLE001
            print(f"[behavior] MediaPipe unavailable: {exc}")
            self.enabled = False

    def close(self) -> None:
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None

    def update(self, frame: np.ndarray, face_box: Optional[tuple[int, int, int, int]]) -> None:
        if not self.enabled or self._holistic is None:
            return

        now = time.time()
        if now - self._last_run < self.interval_sec:
            return
        self._last_run = now

        _ = face_box  # reserved for future region-of-interest heuristics

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._holistic.process(rgb)

        self._last_face_landmarks = results.face_landmarks
        self._last_left_hand_landmarks = results.left_hand_landmarks
        self._last_right_hand_landmarks = results.right_hand_landmarks

        summary = {
            "mood": self._normalize_state(
                self._classify_mood(results.face_landmarks, frame.shape) if results.face_landmarks else None
            ),
            "left": self._normalize_state(
                self._classify_hand(results.left_hand_landmarks, frame.shape, "left")
                if results.left_hand_landmarks
                else None
            ),
            "right": self._normalize_state(
                self._classify_hand(results.right_hand_landmarks, frame.shape, "right")
                if results.right_hand_landmarks
                else None
            ),
        }

        with self._state_lock:
            prev = self._state.copy()
            self._state.update(summary)

        if prev.get("mood") != summary["mood"]:
            self._log_state_change("Mood", summary["mood"])
        if prev.get("left") != summary["left"]:
            self._log_state_change("Left hand", summary["left"])
        if prev.get("right") != summary["right"]:
            self._log_state_change("Right hand", summary["right"])

    def annotate(self, frame: np.ndarray) -> None:
        if not self.enabled or self._holistic is None or self._drawing is None:
            return

        if self._last_face_landmarks is not None and self._mp is not None and self._drawing_styles is not None:
            try:
                self._drawing.draw_landmarks(
                    frame,
                    self._last_face_landmarks,
                    self._mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._drawing_styles.get_default_face_mesh_contours_style(),
                )
            except Exception:
                pass

        if self._last_left_hand_landmarks is not None and self._mp is not None and self._drawing_styles is not None:
            try:
                self._drawing.draw_landmarks(
                    frame,
                    self._last_left_hand_landmarks,
                    self._mp.solutions.hands.HAND_CONNECTIONS,
                    self._drawing_styles.get_default_hand_landmarks_style(),
                    self._drawing_styles.get_default_hand_connections_style(),
                )
            except Exception:
                pass

        if self._last_right_hand_landmarks is not None and self._mp is not None and self._drawing_styles is not None:
            try:
                self._drawing.draw_landmarks(
                    frame,
                    self._last_right_hand_landmarks,
                    self._mp.solutions.hands.HAND_CONNECTIONS,
                    self._drawing_styles.get_default_hand_landmarks_style(),
                    self._drawing_styles.get_default_hand_connections_style(),
                )
            except Exception:
                pass

        overlay_y = 24
        with self._state_lock:
            mood = self._state.get("mood")
            left = self._state.get("left")
            right = self._state.get("right")

        if mood is not None:
            label, conf = mood
            cv2.putText(
                frame,
                f"Mood: {label} ({conf:.2f})",
                (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 40),
                2,
            )
            overlay_y += 22

        if left is not None:
            label, conf = left
            cv2.putText(
                frame,
                f"Left hand: {label} ({conf:.2f})",
                (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (120, 200, 255),
                2,
            )
            overlay_y += 20

        if right is not None:
            label, conf = right
            cv2.putText(
                frame,
                f"Right hand: {label} ({conf:.2f})",
                (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 180, 120),
                2,
            )

    def _normalize_state(self, state: Optional[tuple[str, float]]) -> Optional[tuple[str, float]]:
        if not state:
            return None
        label, conf = state
        conf = max(0.0, min(1.0, float(conf)))
        return label, round(conf, 2)

    def _log_state_change(self, prefix: str, state: Optional[tuple[str, float]]) -> None:
        if state is None:
            print(f"{prefix}: unknown")
            return
        label, conf = state
        print(f"{prefix}: {label} ({conf:.2f})")

    def _classify_mood(
        self,
        face_landmarks,
        frame_shape: tuple[int, int, int],
    ) -> Optional[tuple[str, float]]:
        try:
            landmarks = face_landmarks.landmark
            needed = [61, 291, 13, 14]
            if any(idx >= len(landmarks) for idx in needed):
                return None

            mouth_left = self._landmark_to_xy(landmarks[61], frame_shape)
            mouth_right = self._landmark_to_xy(landmarks[291], frame_shape)
            mouth_top = self._landmark_to_xy(landmarks[13], frame_shape)
            mouth_bottom = self._landmark_to_xy(landmarks[14], frame_shape)

            mouth_width = self._distance(mouth_left, mouth_right)
            mouth_height = self._distance(mouth_top, mouth_bottom)
            if mouth_width < 1.0:
                return None

            ratio = mouth_height / mouth_width
            center_y = (mouth_top[1] + mouth_bottom[1]) * 0.5
            corners_y = (mouth_left[1] + mouth_right[1]) * 0.5
            curve = (center_y - corners_y) / mouth_width

            if ratio > 0.36:
                confidence = min(1.0, (ratio - 0.28) / 0.25)
                return "surprised", confidence
            if curve > 0.025:
                confidence = min(1.0, (curve - 0.015) / 0.05)
                return "happy", confidence
            if curve < -0.02:
                confidence = min(1.0, (-curve - 0.015) / 0.05)
                return "sad", confidence

            neutral_score = 1.0 - min(1.0, abs(ratio - 0.2) * 3.0 + abs(curve) * 10.0)
            return "neutral", neutral_score
        except Exception:
            return None

    def _classify_hand(
        self,
        hand_landmarks,
        frame_shape: tuple[int, int, int],
        handedness: str,
    ) -> Optional[tuple[str, float]]:
        try:
            pts = [self._landmark_to_xy(lm, frame_shape) for lm in hand_landmarks.landmark]
        except Exception:
            return None

        if len(pts) < 21:
            return None

        h, w = frame_shape[0], frame_shape[1]
        vert_thresh = max(6.0, 0.02 * h)
        horiz_thresh = max(6.0, 0.02 * w)

        def finger_extended(tip_idx: int, pip_idx: int) -> bool:
            return pts[tip_idx][1] < (pts[pip_idx][1] - vert_thresh)

        if handedness == "right":
            thumb_extended = pts[4][0] > (pts[2][0] + horiz_thresh)
        else:
            thumb_extended = pts[4][0] < (pts[2][0] - horiz_thresh)

        index_extended = finger_extended(8, 6)
        middle_extended = finger_extended(12, 10)
        ring_extended = finger_extended(16, 14)
        pinky_extended = finger_extended(20, 18)

        extended_flags = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
        extended_count = sum(1 for flag in extended_flags if flag)

        if extended_count == 0:
            return "fist", 1.0

        if index_extended and not any([middle_extended, ring_extended, pinky_extended, thumb_extended]):
            return "point", 0.9

        if extended_count >= 4:
            return "open", min(1.0, 0.6 + 0.1 * extended_count)

        if index_extended and middle_extended and not (ring_extended or pinky_extended or thumb_extended):
            return "peace", 0.85

        return "partial", min(1.0, 0.3 + 0.15 * extended_count)

    @staticmethod
    def _landmark_to_xy(landmark, frame_shape: tuple[int, int, int]) -> tuple[float, float]:
        h, w = frame_shape[0], frame_shape[1]
        return landmark.x * w, landmark.y * h

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])


az_ctrl = AxisControl()
alt_ctrl = AxisControl()


# ---------------- Detector abstraction ----------------
class Detector:
    def __init__(self):
        self.kind: Optional[str] = None
        self.detector = None
        self._order = ["yunet", "dnn", "haar"]
        if not self._init_detector():
            raise RuntimeError(
                "No face detector available. Provide YuNet ONNX, DNN prototxt+caffemodel, or Haar XML."
            )

    # ---------------- internal helpers ----------------
    def _init_detector(self, skip: set[str] | None = None) -> bool:
        skip = set(skip or set())
        self.detector = None

        for kind in self._order:
            if kind in skip:
                continue

            if kind == "yunet":
                if not ((hasattr(cv2, "FaceDetectorYN_create") or hasattr(cv2, "FaceDetectorYN")) and os.path.isfile(YUNET_ONNX)):
                    continue
                try:
                    if hasattr(cv2, "FaceDetectorYN_create"):
                        self.detector = cv2.FaceDetectorYN_create(
                            YUNET_ONNX, "", (FRAME_W, FRAME_H), score_threshold=0.5, nms_threshold=0.3, top_k=5000
                        )
                    else:
                        self.detector = cv2.FaceDetectorYN.create(
                            YUNET_ONNX, "", (FRAME_W, FRAME_H), 0.5, 0.3, 5000
                        )
                    self.kind = "yunet"
                    print("[detector] Using: yunet")
                    return True
                except Exception as exc:
                    print(f"[detector] YuNet init failed: {exc}")
                    self.detector = None
                    continue

            if kind == "dnn":
                if not (os.path.isfile(DNN_PROTO) and os.path.isfile(DNN_MODEL)):
                    continue
                try:
                    self.detector = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
                    self.kind = "dnn"
                    print("[detector] Using: dnn")
                    return True
                except Exception as exc:
                    print(f"[detector] DNN init failed: {exc}")
                    self.detector = None
                    continue

            if kind == "haar":
                if not os.path.isfile(HAAR_XML):
                    continue
                cascade = cv2.CascadeClassifier(HAAR_XML)
                if cascade.empty():
                    continue
                self.detector = cascade
                self.kind = "haar"
                print("[detector] Using: haar")
                return True

        self.kind = None
        return False

    def _fallback(self, failed_kind: str) -> bool:
        print(f"[detector] {failed_kind} failed; attempting fallback")
        return self._init_detector(skip={failed_kind})

    def detect(self, frame_bgr):
        """Return list of (x, y, w, h) int boxes."""
        h, w = frame_bgr.shape[:2]

        if self.kind == "yunet":
            try:
                try:
                    self.detector.setInputSize((w, h))
                except Exception:
                    pass
                _, faces = self.detector.detect(frame_bgr)
            except cv2.error as exc:
                print(f"[detector] YuNet runtime error: {exc}")
                if self._fallback("yunet"):
                    return self.detect(frame_bgr)
                return []

            boxes = []
            if faces is not None:
                for f in faces:
                    x, y, bw, bh = f[:4]
                    boxes.append((int(x), int(y), int(bw), int(bh)))
            return boxes

        if self.kind == "dnn":
            try:
                blob = cv2.dnn.blobFromImage(
                    frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
                )
                self.detector.setInput(blob)
                detections = self.detector.forward()
            except cv2.error as exc:
                print(f"[detector] DNN runtime error: {exc}")
                if self._fallback("dnn"):
                    return self.detect(frame_bgr)
                return []

            boxes = []
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf < 0.6:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h], dtype=np.float32)
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                boxes.append((x1, y1, max(0, x2 - x1), max(0, y2 - y1)))
            return boxes

        if self.kind == "haar":
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
                )
            except cv2.error as exc:
                print(f"[detector] Haar runtime error: {exc}")
                return []
            return [(int(x), int(y), int(bw), int(bh)) for (x, y, bw, bh) in faces]

        # If we reach here, detector is unavailable
        if self._init_detector():
            return self.detect(frame_bgr)
        return []


# ---------------- Helpers ----------------
def error_to_rate_and_dir(dx, dy):
    # Azimuth (horizontal)
    if abs(dx) <= DEADBAND_PX:
        az_rate, az_dir = 0.0, True
    else:
        az_rate = max(MIN_RATE_HZ, min(MAX_RATE_HZ, K_P * abs(dx)))
        az_dir = (dx > 0) == AZ_DIR_POS_RIGHT  # map sign to DIR polarity

    # Altitude (vertical)
    if abs(dy) <= DEADBAND_PX:
        alt_rate, alt_dir = 0.0, True
    else:
        alt_rate = max(MIN_RATE_HZ, min(MAX_RATE_HZ, K_P * abs(dy)))
        alt_dir = (dy > 0) == ALT_DIR_POS_DOWN

    return az_rate, az_dir, alt_rate, alt_dir


def main():
    print("Drivers will auto-enable/disable")
    enable_drivers(False)  # start disabled
    time.sleep(0.05)

    # Start motor threads
    a_worker = StepperWorker("AZ", A_dir, A_step, az_ctrl.get_rate, az_ctrl.get_dir)
    b_worker = StepperWorker("ALT", B_dir, B_step, alt_ctrl.get_rate, alt_ctrl.get_dir)
    a_worker.start()
    b_worker.start()

    # Open USB camera
    cap = cv2.VideoCapture(CAP_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        a_worker.stop()
        b_worker.stop()
        a_worker.join(timeout=1.0)
        b_worker.join(timeout=1.0)
        enable_drivers(False)
        raise RuntimeError(f"Could not open /dev/video{CAP_INDEX}")

    # Init face detector (YuNet -> DNN -> Haar)
    detector = Detector()
    behavior = BehaviorAnalyzer(BEHAVIOR_INTERVAL)
    print("Face-follow running. Ctrl+C to stop.")

    # Start HTTP stream (daemon thread)
    threading.Thread(
        target=lambda: app.run(host=STREAM_HOST, port=STREAM_PORT, debug=False, use_reloader=False),
        daemon=True,
    ).start()
    print(f"Stream at http://{STREAM_HOST}:{STREAM_PORT} (use the Pi's IP if 0.0.0.0)")

    # Throttled "No face" logging
    had_face_prev = False
    last_no_face_log = 0.0
    NO_FACE_LOG_INTERVAL = 2.0
    last_face_time = 0.0
    last_motion_time = 0.0
    ewma_dx = 0.0
    ewma_dy = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            # Detect faces
            boxes = detector.detect(frame)
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            min_face_area = int(w * h * MIN_FACE_AREA_RATIO)

            # choose largest face
            best = None
            best_area = 0
            for (x, y, bw, bh) in boxes:
                area = bw * bh
                if area > best_area:
                    best_area = area
                    best = (x, y, bw, bh)

            if best is not None and best_area < min_face_area:
                # Treat tiny blobs as noise
                best = None

            behavior.update(frame, best)

            if best is not None:
                x, y, bw, bh = best
                fx, fy = x + bw // 2, y + bh // 2
                dx, dy = fx - cx, fy - cy

                # smooth the error to avoid jitter
                ewma_dx = (1 - SMOOTHING) * ewma_dx + SMOOTHING * dx
                ewma_dy = (1 - SMOOTHING) * ewma_dy + SMOOTHING * dy

                az_rate, az_dir, alt_rate, alt_dir = error_to_rate_and_dir(ewma_dx, ewma_dy)
                az_ctrl.set(az_rate, az_dir)
                alt_ctrl.set(alt_rate, alt_dir)

                # Any motion requested?
                if (az_rate > 0.0) or (alt_rate > 0.0):
                    last_motion_time = time.time()
                    enable_drivers(True)  # ensure powered when moving

                last_face_time = time.time()
                had_face_prev = True

                # annotate
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.circle(frame, (fx, fy), 4, (255, 0, 0), -1)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                print(
                    f"Face dx={dx:+4d} dy={dy:+4d} "
                    f"AZ:{'+' if az_dir else '-'}{az_rate:5.1f} Hz "
                    f"ALT:{'+' if alt_dir else '-'}{alt_rate:5.1f} Hz"
                )

            else:
                # no face: stop gently
                az_ctrl.set(0.0, True)
                alt_ctrl.set(0.0, True)

                # If we've been without a face long enough, disable drivers
                if (time.time() - last_face_time) > FACE_LOSS_TIMEOUT:
                    enable_drivers(False)

                now = time.time()
                if had_face_prev or (now - last_no_face_log) >= NO_FACE_LOG_INTERVAL:
                    print("No face")
                    last_no_face_log = now
                had_face_prev = False

            behavior.annotate(frame)
            _set_stream_frame(frame)

            # If face is centered (both rates 0) for a while, also disable
            if (
                az_ctrl.get_rate() == 0.0
                and alt_ctrl.get_rate() == 0.0
                and (time.time() - last_motion_time) > IDLE_TIMEOUT_SEC
            ):
                enable_drivers(False)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        # stop motors
        az_ctrl.set(0.0, True)
        alt_ctrl.set(0.0, True)
        a_worker.stop()
        b_worker.stop()
        a_worker.join(timeout=1.0)
        b_worker.join(timeout=1.0)
        enable_drivers(False)
        behavior.close()
        cap.release()
        print("Drivers disabled. Bye.")


if __name__ == "__main__":
    main()
