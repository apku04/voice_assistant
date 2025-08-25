# voice_assistant/stt_backends.py
import os
import json
import array
import tempfile
import subprocess
import time
from typing import Optional
import contextlib


try:
    from vosk import Model as VoskModel, KaldiRecognizer
except ImportError:
    VoskModel = None
    KaldiRecognizer = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from .config import config

class STTBase:
    """Base class for Speech-to-Text backends"""
    
    def __init__(self):
        self.device_index = None
        
    def listen_once(self) -> str:
        """Listen and transcribe audio"""
        raise NotImplementedError("Subclasses must implement listen_once")


class STTWhisper(STTBase):
    """Whisper-based speech recognition"""
    
    def __init__(self, lang: str = "en", device_index: Optional[int] = None):
        super().__init__()
        if WhisperModel is None:
            raise RuntimeError("faster-whisper not installed. pip3 install faster-whisper")
        
        self.lang = lang
        self.device_index = device_index
        self.model = WhisperModel(
            config.whisper_model_name, 
            device="cpu", 
            compute_type="int8", 
            cpu_threads=config.whisper_threads
        )

    def listen_once(self, secs: int = None) -> str:
        """Record and transcribe using Whisper"""
        if secs is None:
            secs = config.whisper_secs
            
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            wav = wf.name
            
        try:
            # Use device index if specified, otherwise use config
            mic_device = f"plughw:{self.device_index},0" if self.device_index is not None else config.mic_alsa
            
            print(f"[rec] {secs}s @ 16000 Hz from {mic_device} …")
            cmd = [
                "arecord", "-q", "-D", mic_device, 
                "-f", "S16_LE", "-r", "16000", "-c", "1", 
                "-d", str(secs), wav
            ]
            subprocess.run(cmd, check=True)
            
            print(f"[stt] whisper model={config.whisper_model_name} lang={self.lang} (INT8)")
            segments, info = self.model.transcribe(wav, language=self.lang)
            txt_parts = [s.text for s in segments if getattr(s, 'text', '').strip()]
            return " ".join(txt_parts).strip()
        finally:
            try:
                os.remove(wav)
            except Exception:
                pass


class STTVosk(STTBase):
    """Vosk-based speech recognition"""
    
    def __init__(self, model_dir: str = None, device_index: Optional[int] = None):
        super().__init__()
        if VoskModel is None or KaldiRecognizer is None:
            raise RuntimeError("vosk not installed. pip3 install vosk")
            
        if model_dir is None:
            model_dir = config.get_vosk_model_path()
            
        if not model_dir or not os.path.isdir(model_dir):
            raise RuntimeError("VOSK_MODEL path invalid. Set env VOSK_MODEL or download a model.")
            
        self.model = VoskModel(model_dir)
        self.device_index = device_index

    def listen_once(self) -> str:
        """Stream audio and transcribe using Vosk - matches original 2-file behavior exactly"""
        rate = config.vosk_rate
        # Match original: use 0.003 inside listen_once() even if module constant is different
        sil = 0.003  # Hardcoded to match original behavior
        holdms = config.silence_hold_ms
        maxms = config.max_listen_ms

        # 150 ms chunk @16kHz, int16 => 2 bytes/sample
        CHUNK = 2400 * 2

        # Use device index if specified, otherwise use config
        mic_device = f"plughw:{self.device_index},0" if self.device_index is not None else config.mic_alsa
        
        cmd = [
            "arecord", "-q", "-D", mic_device, 
            "-f", "S16_LE", "-r", str(rate), 
            "-c", "1", "-t", "raw"
        ]
        
        print(f"[stt] vosk via arecord… dev={mic_device} rate={rate} sil={sil} hold={holdms}ms max={maxms}ms")
        rec = KaldiRecognizer(self.model, rate)

        started = False
        last_voice_ms = 0
        t0 = time.time()

        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0) as p:
            try:
                while True:
                    buf = p.stdout.read(CHUNK)
                    if not buf:
                        break

                    a = array.array('h', buf)
                    rms = ((sum(x*x for x in a)/len(a))**0.5 / 32768.0) if a else 0.0

                    now_ms = int((time.time() - t0) * 1000)
                    if rms >= sil:
                        started = True
                        last_voice_ms = now_ms

                    rec.AcceptWaveform(buf)

                    if now_ms > maxms:
                        break
                    if started and (now_ms - last_voice_ms) >= holdms:
                        break
            finally:
                with contextlib.suppress(Exception):
                    p.terminate()
                    p.wait(timeout=0.2)

        try:
            return json.loads(rec.FinalResult()).get("text", "").strip()
        except Exception:
            return ""


class STTManager:
    """Manager for STT backends"""
    
    def __init__(self):
        self.current_backend = None
        self.current_lang = config.lang_default
        self.current_device_index = None
        
    def initialize_backend(self, backend_name: str, lang: str = None, device_index: Optional[int] = None) -> STTBase:
        """Initialize the specified STT backend"""
        if lang is None:
            lang = self.current_lang
            
        if device_index is None:
            device_index = self.current_device_index
            
        if backend_name == "whisper":
            obj = STTWhisper(lang=lang, device_index=device_index)
        elif backend_name == "vosk":
            obj = STTVosk(model_dir=config.get_vosk_model_path(), device_index=device_index)
        else:
            raise ValueError(f"Unknown STT backend: {backend_name}")
            
        self.current_backend = obj
        return obj
            
    def set_language(self, lang: str):
        """Set language for STT backends"""
        self.current_lang = lang
        if isinstance(self.current_backend, STTWhisper):
            # Reinitialize whisper with new language
            self.current_backend = STTWhisper(lang=lang, device_index=self.current_device_index)
            
    def set_device_index(self, device_index: int):
        """Set audio device index for STT backends"""
        self.current_device_index = device_index
        if self.current_backend:
            self.current_backend.device_index = device_index