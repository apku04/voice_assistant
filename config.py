"""Structured configuration utilities for the voice assistant."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

try:  # Optional dependency – YAML is nice to have but not required
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without PyYAML
    yaml = None


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "")) if os.getenv(name) else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "")) if os.getenv(name) else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


@dataclass(slots=True)
class AppConfig:
    """Typed configuration with environment + file overrides."""

    # Ollama / LLM
    pc_ollama_url: str = field(default_factory=lambda: _env_str("OLLAMA_PC_URL", "http://192.168.1.6:11434/api/chat"))
    pi_ollama_url: str = field(default_factory=lambda: _env_str("OLLAMA_PI_URL", "http://localhost:11434/api/chat"))
    default_model: str = field(default_factory=lambda: _env_str("OLLAMA_MODEL", "llama3.2"))

    # Piper / ALSA (playback)
    piper_model: str = field(default_factory=lambda: _env_str("PIPER_MODEL", "/home/pi/piper-tts/piper/voices/en_US-joe-medium.onnx"))
    alsa_dev: str = field(default_factory=lambda: _env_str("ALSA_DEV", "plughw:0,0"))
    robot_preset: str = field(default_factory=lambda: _env_str("ROBOT_PRESET", "metallic"))
    piper_noise: str = field(default_factory=lambda: _env_str("PIPER_NOISE", "0.04"))
    piper_len: str = field(default_factory=lambda: _env_str("PIPER_LEN", "1.20"))
    piper_noisew: str = field(default_factory=lambda: _env_str("PIPER_NOISEW", "1.0"))

    # Mic ALSA path
    mic_alsa: str = field(default_factory=lambda: _env_str("MIC_DEV", "plughw:CARD=U0x46d0x821,DEV=0"))
    mic_name_hint: str = field(default_factory=lambda: _env_str("MIC_NAME_HINT", "logitech"))

    # STT defaults
    stt_backend: str = field(default_factory=lambda: _env_str("STT_BACKEND", "vosk"))
    lang_default: str = field(default_factory=lambda: _env_str("LANG_DEFAULT", "en"))
    whisper_model_name: str = field(default_factory=lambda: _env_str("WHISPER_MODEL", "small"))
    whisper_threads: int = field(default_factory=lambda: _env_int("WHISPER_THREADS", 4))
    whisper_secs: int = field(default_factory=lambda: _env_int("WHISPER_SECS", 6))
    vosk_model_dir: str = field(default_factory=lambda: _env_str("VOSK_MODEL", ""))
    vosk_rate: int = 16000
    vosk_block: int = field(default_factory=lambda: _env_int("VOSK_BLOCK", 8000))
    silence_rms: float = field(default_factory=lambda: _env_float("VOSK_SILENCE_RMS", 0.003))
    silence_hold_ms: int = field(default_factory=lambda: _env_int("VOSK_SIL_MS", 900))
    max_listen_ms: int = field(default_factory=lambda: _env_int("VOSK_MAX_MS", 12000))

    # RGB LED Config
    red_pin: int = 22
    green_pin: int = 17
    blue_pin: int = 27
    led_active_high: bool = True
    idle_color: tuple[float, float, float] = (0.0, 1.0, 0.0)
    think_color: tuple[float, float, float] = (0.0, 0.0, 1.0)
    speak_color: tuple[float, float, float] = (0.0, 1.0, 1.0)
    error_color: tuple[float, float, float] = (1.0, 0.0, 0.0)

    # Display defaults (SH1107 OLED)
    display_enabled: bool = True
    oled_addr: int = 0x3C
    oled_i2c_port: int = 1
    oled_width: int = 128
    oled_height: int = 128

    # Diagnostics / Logs Config
    max_log_chars: int = 3000
    default_log_lines: int = 150
    status_patterns: list[str] = field(default_factory=lambda: [
        r"\bstatus\b", r"\bhealth\b", r"\btemp\w*", r"\bcpu\b", r"\bram\b", r"\bmemory\b",
        r"\bdisk\b", r"\bstorage\b", r"\buptime\b", r"\bthrottl(?:e|ed)\b", r"\boverheat\w*",
        r"\bjournal\b", r"\berrors?\b", r"\bdiag(?:nostic)?\b"
    ])

    # Conversation settings
    max_history_length: int = 20
    max_chunks: int = 6
    max_chars: int = 240
    pause_s: float = 0.08

    # Features / integration toggles
    face_follow_enabled: bool = field(default_factory=lambda: _env_bool("FACE_FOLLOW_ENABLED", True))

    # Internal cached metadata
    _config_path: Optional[Path] = field(default=None, init=False, repr=False)

    # ------------------------- factory helpers -------------------------
    @classmethod
    def from_mapping(cls, overrides: Mapping[str, Any] | None = None) -> "AppConfig":
        """Build configuration applying overrides from a mapping."""
        overrides = overrides or {}
        flat = _flatten(overrides)
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in flat.items() if k in valid_keys}
        cfg = cls(**filtered)
        return cfg

    # ----------------------------- methods -----------------------------
    def get_vosk_model_path(self) -> str:
        """Return configured or fallback Vosk model directory."""
        if self.vosk_model_dir and os.path.isdir(self.vosk_model_dir):
            return self.vosk_model_dir
        home = Path.home()
        for candidate in ("vosk-model-small-da-0.3", "vosk-model-small-en-us-0.15"):
            path = home / candidate
            if path.is_dir():
                return str(path)
        return self.vosk_model_dir

    def to_dict(self) -> Dict[str, Any]:
        """Expose configuration as a serialisable dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def update(self, mapping: Mapping[str, Any]) -> None:
        """Update configuration in-place using a mapping of overrides."""
        flat = _flatten(mapping)
        for key, value in flat.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # ---------------------------- reload -------------------------------
    def reload(self) -> None:
        """Reload configuration from the associated config path if available."""
        if not self._config_path:
            return
        data = _load_mapping(self._config_path)
        if data:
            self.update(data)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_SECTION_ALIASES: Dict[str, Dict[str, str]] = {
    "ollama": {
        "pc_url": "pc_ollama_url",
        "pi_url": "pi_ollama_url",
        "default_model": "default_model",
    },
    "piper": {
        "model": "piper_model",
        "alsa_dev": "alsa_dev",
        "preset": "robot_preset",
        "robot_preset": "robot_preset",
        "noise": "piper_noise",
        "len": "piper_len",
        "length": "piper_len",
        "length_scale": "piper_len",
        "noise_w": "piper_noisew",
    },
    "microphone": {
        "alsa": "mic_alsa",
        "alsa_device": "mic_alsa",
        "device": "mic_alsa",
        "name_hint": "mic_name_hint",
    },
    "stt": {
        "backend": "stt_backend",
        "lang_default": "lang_default",
        "language": "lang_default",
        "whisper_model": "whisper_model_name",
        "whisper_model_name": "whisper_model_name",
        "whisper_threads": "whisper_threads",
        "whisper_seconds": "whisper_secs",
        "whisper_secs": "whisper_secs",
        "vosk_model": "vosk_model_dir",
        "vosk_model_dir": "vosk_model_dir",
        "vosk_rate": "vosk_rate",
        "vosk_block": "vosk_block",
        "silence_rms": "silence_rms",
        "silence_hold_ms": "silence_hold_ms",
        "max_listen_ms": "max_listen_ms",
    },
    "led": {
        "red_pin": "red_pin",
        "green_pin": "green_pin",
        "blue_pin": "blue_pin",
        "active_high": "led_active_high",
        "idle_color": "idle_color",
        "think_color": "think_color",
        "speak_color": "speak_color",
        "error_color": "error_color",
    },
    "display": {
        "enabled": "display_enabled",
        "oled_addr": "oled_addr",
        "address": "oled_addr",
        "i2c_port": "oled_i2c_port",
        "oled_i2c_port": "oled_i2c_port",
        "width": "oled_width",
        "height": "oled_height",
    },
    "diagnostics": {
        "max_log_chars": "max_log_chars",
        "default_log_lines": "default_log_lines",
        "status_patterns": "status_patterns",
    },
    "conversation": {
        "max_history_length": "max_history_length",
        "max_chunks": "max_chunks",
        "max_chars": "max_chars",
        "pause_s": "pause_s",
    },
    "features": {
        "face_follow": "face_follow_enabled",
        "face_follow_enabled": "face_follow_enabled",
    },
}


def _flatten(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in mapping.items():
        if key in _SECTION_ALIASES and isinstance(value, Mapping):
            section = _SECTION_ALIASES[key]
            for sub_key, sub_value in value.items():
                alias = section.get(sub_key)
                if alias:
                    flat[alias] = sub_value
        else:
            flat[key] = value
    return flat


def _load_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if yaml is not None and path.suffix.lower() in {".yml", ".yaml"}:
        data = yaml.safe_load(text) or {}
        if isinstance(data, Mapping):
            return dict(data)
        raise ValueError(f"Expected mapping in config file {path}")
    try:
        data = json.loads(text)
        if isinstance(data, Mapping):
            return dict(data)
    except json.JSONDecodeError:
        raise ValueError(f"Unsupported config format for {path}. Install PyYAML for YAML support.")
    return {}


def _candidate_paths(explicit: Optional[Path]) -> Iterator[Path]:
    if explicit and explicit.exists():
        yield explicit
    env_path = os.getenv("VOICE_ASSISTANT_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.exists():
            yield p
    local_yaml = Path(__file__).resolve().with_name("config.yaml")
    if local_yaml.exists():
        yield local_yaml


def load_config(config_path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Load configuration from defaults, environment and optional file."""
    path_arg = Path(config_path) if config_path else None
    merged: Dict[str, Any] = {}
    for candidate in _candidate_paths(path_arg):
        merged.update(_load_mapping(candidate))
        path_arg = candidate  # remember last successful path
    cfg = AppConfig.from_mapping(merged)
    cfg._config_path = path_arg
    return cfg


def get_config() -> AppConfig:
    """Return process-global configuration instance."""
    return config


# Global configuration instance
config = load_config()

__all__ = ["AppConfig", "config", "load_config", "get_config"]
