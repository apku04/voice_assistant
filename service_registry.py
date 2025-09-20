"""Centralised service registry for the voice assistant."""
from __future__ import annotations

from typing import Any, Dict

from .config import AppConfig, get_config
from .led_manager import LEDManager
from .display_manager import DisplayManager
from .piper_tts import PiperTTS
from .stt_backends import STTManager
from .memory_manager import ConversationMemory
from .face_follow_manager import FaceFollowManager
from .script_manager import ScriptManager


class ServiceRegistry:
    """Lazy, injectable factory for core services."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or get_config()
        self._cache: Dict[str, Any] = {}

    # ---------------------------- helpers -----------------------------
    def _get(self, key: str, factory):
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    # ----------------------------- services ---------------------------
    def get_led_manager(self) -> LEDManager:
        return self._get("led", lambda: LEDManager(self.config))

    def get_display_manager(self) -> DisplayManager:
        return self._get("display", lambda: DisplayManager(self.config))

    def get_tts(self) -> PiperTTS:
        def _factory() -> PiperTTS:
            return PiperTTS(
                led_manager=self.get_led_manager(),
                max_chunks=self.config.max_chunks,
                max_chars=self.config.max_chars,
                pause_s=self.config.pause_s,
                config=self.config,
            )
        return self._get("tts", _factory)

    def get_stt_manager(self) -> STTManager:
        return self._get("stt_manager", lambda: STTManager(self.config))

    def get_memory(self) -> ConversationMemory:
        return self._get("memory", ConversationMemory)

    def get_script_manager(self) -> ScriptManager:
        return self._get("scripts", ScriptManager)

    def get_face_follow(self) -> FaceFollowManager:
        def _factory() -> FaceFollowManager:
            return FaceFollowManager(display=self.get_display_manager())
        return self._get("face_follow", _factory)

    # --------------------------- lifecycle ----------------------------
    def reset(self) -> None:
        """Dispose cached instances."""
        for key in list(self._cache.keys()):
            inst = self._cache.pop(key, None)
            stop = getattr(inst, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass


__all__ = ["ServiceRegistry"]
