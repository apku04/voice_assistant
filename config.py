# voice_assistant/config.py
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple


def _env_flag(name: str, default: bool = False) -> bool:
    """Interpret common truthy strings from the environment."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}

class AppConfig:
    """Application configuration management"""
    
    def __init__(self):
        # Ollama endpoints
        self.pc_ollama_url = "http://192.168.1.6:11434/api/chat"
        self.pi_ollama_url = "http://localhost:11434/api/chat"
        self.default_model = "llama3.2"
        
        # Piper / ALSA (playback)
        self.piper_model = os.getenv("PIPER_MODEL", "/home/pi/piper-tts/piper/voices/en_US-joe-medium.onnx")
        self.alsa_dev = os.getenv("ALSA_DEV", "plughw:0,0")
        self.robot_preset = os.getenv("ROBOT_PRESET", "metallic")
        self.piper_noise = os.getenv("PIPER_NOISE", "0.04")
        self.piper_len = os.getenv("PIPER_LEN", "1.20")
        self.piper_noisew = os.getenv("PIPER_NOISEW", "1.0")
        
        # Mic ALSA path
        self.mic_alsa = os.getenv("MIC_DEV", "plughw:CARD=U0x46d0x821,DEV=0")
        self.mic_name_hint = os.getenv("MIC_NAME_HINT", "logitech")
        
        # STT defaults - match original 2-file behavior
        self.stt_backend = os.getenv("STT_BACKEND", "vosk")
        self.lang_default = "en"
        self.whisper_model_name = os.getenv("WHISPER_MODEL", "small")
        self.whisper_threads = int(os.getenv("WHISPER_THREADS", "4"))
        self.whisper_secs = int(os.getenv("WHISPER_SECS", "6"))
        self.vosk_model_dir = os.getenv("VOSK_MODEL", "")
        self.vosk_rate = 16000
        self.vosk_block = int(os.getenv("VOSK_BLOCK", "8000"))
        # Match original: 0.003 inside listen_once() even if module constant is 0.008
        self.silence_rms = float(os.getenv("VOSK_SILENCE_RMS", "0.003"))  # Changed to 0.003
        self.silence_hold_ms = int(os.getenv("VOSK_SIL_MS", "900"))
        self.max_listen_ms = int(os.getenv("VOSK_MAX_MS", "12000"))
        
        # RGB LED Config
        self.red_pin, self.green_pin, self.blue_pin = 22, 17, 27
        self.led_active_high = True
        self.idle_color, self.think_color, self.speak_color, self.error_color = (
            (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 0.0)
        )
        
        # Diagnostics / Logs Config
        self.max_log_chars = 3000
        self.default_log_lines = 150
        self.status_patterns = [
            r"\bstatus\b", r"\bhealth\b", r"\btemp\w*", r"\bcpu\b", r"\bram\b", r"\bmemory\b",
            r"\bdisk\b", r"\bstorage\b", r"\buptime\b", r"\bthrottl(?:e|ed)\b", r"\boverheat\w*",
            r"\bjournal\b", r"\berrors?\b", r"\bdiag(?:nostic)?\b"
        ]
        
        # Conversation settings
        self.max_history_length = 20
        self.max_chunks = 6
        self.max_chars = 240
        self.pause_s = 0.08

        # Face follow integration
        self.face_follow_enabled = True  # start tracker on boot

        # Perception LLM bridge (disabled by default)
        self.perception_llm_enabled = _env_flag("PERCEPTION_LLM", True)
        self.perception_llm_model = os.getenv("PERCEPTION_LLM_MODEL", self.default_model)
        try:
            self.perception_llm_cooldown = float(os.getenv("PERCEPTION_LLM_COOLDOWN", "3.0"))
        except ValueError:
            self.perception_llm_cooldown = 3.0
        
    def get_vosk_model_path(self) -> str:
        """Get Vosk model path with fallback logic"""
        if self.vosk_model_dir and os.path.isdir(self.vosk_model_dir):
            return self.vosk_model_dir
            
        # Fall back to common defaults
        home = str(Path.home())
        for candidate in ("vosk-model-small-da-0.3", "vosk-model-small-en-us-0.15"):
            path = os.path.join(home, candidate)
            if os.path.isdir(path):
                return path
                
        return self.vosk_model_dir  # Return original even if invalid

# Global configuration instance
config = AppConfig()
