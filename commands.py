"""Command routing for the voice assistant."""
from __future__ import annotations

import datetime
import re
import shlex
from typing import Callable, Optional, TYPE_CHECKING

from .config import AppConfig
from .diagnostics import maybe_offer_diag
from .system_utils import get_service_log, tail_file

if TYPE_CHECKING:  # pragma: no cover
    from .main import VoiceAssistant


class CommandRouter:
    """Route slash-style commands to the appropriate handler."""

    def __init__(self, assistant: "VoiceAssistant", config: AppConfig,
                 run_diag: Callable[..., str]) -> None:
        self.assistant = assistant
        self.config = config
        self._run_diag = run_diag

    # ------------------------------------------------------------------
    def handle(self, user_input: str) -> bool:
        """Return True if a command was handled."""
        low = user_input.strip().lower()
        if not low:
            return True

        # Quit ----------------------------------------------------------
        if low in {"q", "/q", "/quit", "exit"}:
            print("Bye!")
            return True

        # History -------------------------------------------------------
        if low in {"/clear", "/reset"}:
            self.assistant.history.clear()
            print("History cleared.")
            return True

        # Model ---------------------------------------------------------
        if low.startswith("/model "):
            self.assistant.current_model = user_input.split(" ", 1)[1].strip() or self.assistant.current_model
            self.assistant.history.clear()
            print(f"Model set to: {self.assistant.current_model}. History cleared.")
            return True

        # Face follow ---------------------------------------------------
        if low.startswith("/follow"):
            parts = user_input.split()
            sub = parts[1].lower() if len(parts) > 1 else "status"
            face = self.assistant.face_follow
            if sub in {"on", "start"}:
                ok = face.start()
                print("Face follow:", "started" if ok else "failed")
            elif sub in {"off", "stop"}:
                face.stop()
                print("Face follow: stopped")
            else:
                print("Face follow:", "running" if face.is_running() else "stopped")
                print("Usage: /follow on|off|status")
            return True

        # Language ------------------------------------------------------
        if low.startswith("/lang "):
            arg = user_input.split(" ", 1)[1].strip().lower()
            if arg in {"en", "da"}:
                self.assistant.set_language(arg)
                print(f"Language set to {self.assistant.current_lang.upper()}. History cleared.")
                self.assistant.switch_whisper_language(arg)
            else:
                print("Usage: /lang en | da")
            return True

        # Whisper seconds -----------------------------------------------
        if low.startswith("/secs "):
            try:
                val = int(user_input.split(" ", 1)[1].strip())
                assert 1 <= val <= 20
                self.config.whisper_secs = val
                print(f"Whisper record seconds set to {self.config.whisper_secs}.")
            except Exception:
                print("Usage: /secs 3..20")
            return True

        # STT mode ------------------------------------------------------
        if low.startswith("/stt "):
            mode = user_input.split(" ", 1)[1].strip().lower()
            if mode in {"vosk", "whisper", "off"}:
                self.assistant.set_stt_mode(mode)
            else:
                print("Usage: /stt vosk | whisper | off")
            return True

        # Quick facts ---------------------------------------------------
        if low.strip() in {"/temp", "/temperature"} or re.search(r"\b(cpu|soc|board)?\s*temp\w*\b", low, re.I):
            msg = f"CPU temperature: {self.assistant.pi_temperature()}"
            print(msg)
            self.assistant.tts.speak_chunks([msg])
            return True

        if low.strip() == "/time" or re.search(r"\b(system\s+)?time\b", low):
            now = datetime.datetime.now()
            msg = f"System time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
            print(msg)
            self.assistant.tts.speak_chunks([msg])
            return True

        # Diagnostics ---------------------------------------------------
        if low.startswith("/diag"):
            self._handle_diag(user_input)
            return True

        if low.startswith("/logs "):
            self._handle_logs(user_input)
            return True

        if low.startswith("/tail "):
            self._handle_tail(user_input)
            return True

        # Audio devices -------------------------------------------------
        if low.strip() == "/audio":
            self.assistant.list_audio_devices()
            return True

        # Scripts -------------------------------------------------------
        if low.startswith("/scripts"):
            self._handle_scripts(user_input)
            return True

        # Mic -----------------------------------------------------------
        if low.startswith("/mic "):
            self.assistant.set_microphone(user_input.split(" ", 1)[1].strip())
            return True

        # Memory --------------------------------------------------------
        if low.startswith("/remember "):
            self.assistant.manual_remember(user_input.split(" ", 1)[1].strip())
            return True

        if low.strip() == "/remember":
            print("Usage: /remember <text to save>")
            return True

        if low.startswith("/recall"):
            topic = user_input.split(" ", 1)[1].strip() if " " in low else ""
            self.assistant.manual_recall(topic)
            return True

        if low.startswith("/memory"):
            self.assistant.memory_command(user_input)
            return True

        return False

    # ------------------------------------------------------------------
    def _handle_diag(self, user_input: str) -> None:
        parts = user_input.split()
        flt: Optional[str] = None
        verbose = False
        for tok in parts[1:]:
            if tok == "-v":
                verbose = True
            elif not tok.startswith("-"):
                flt = tok

        self.assistant.busy.start("analyzing")
        try:
            reply = self._run_diag(
                filter_text=flt,
                verbose=verbose,
                current_model=self.assistant.current_model,
                current_lang=self.assistant.current_lang,
            )
            print(f"\nBOT ({self.assistant.current_model}): {reply}\n")
            self.assistant.display.show_message(f"\nOptimus: {reply}\n")
            self.assistant.tts.speak(reply)

            sug = maybe_offer_diag(reply, allow=True)
            if sug:
                q = f"Run diagnostics{(' for ' + sug.get('filter')) if sug.get('filter') else ''}{' (verbose)' if sug.get('verbose') else ''} now?"
                print(f"\n[Prompt] {q} (yes/no)")
                self.assistant.tts.speak_chunks([q])
        finally:
            self.assistant.busy.stop()

    def _handle_logs(self, user_input: str) -> None:
        parts = user_input.split()
        svc = parts[1] if len(parts) >= 2 else ""
        n = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else self.config.default_log_lines
        if not svc:
            print("Usage: /logs <service> [lines]")
            return
        data = get_service_log(svc, n)
        print(data)
        if data:
            self.assistant.tts.speak_chunks([data[:240]])

    def _handle_tail(self, user_input: str) -> None:
        parts = shlex.split(user_input)
        if len(parts) < 2:
            print("Usage: /tail <path> [lines]")
            return
        path = parts[1]
        n = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else self.config.default_log_lines
        data = tail_file(path, n)
        print(data)
        if data:
            self.assistant.tts.speak_chunks([data[:240]])

    def _handle_scripts(self, user_input: str) -> None:
        parts = user_input.split()
        sub = parts[1] if len(parts) > 1 else "list"
        scripts = self.assistant.scripts
        if sub == "list":
            ents = scripts.list_entries()
            print("Scripts:")
            for e in ents:
                print(f"- {e.key}: {', '.join(e.phrases)} -> {e.script} {e.args}")
        elif sub == "reload":
            scripts.reload()
            print("Scripts reloaded.")
        elif sub == "run" and len(parts) > 2:
            ok, out = scripts.run(parts[2])
            print(out)
            if ok:
                self.assistant.tts.speak_chunks([out])
        else:
            print("Usage: /scripts list | reload | run <key>")
