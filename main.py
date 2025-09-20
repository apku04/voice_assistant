# voice_assistant/main.py
from __future__ import annotations

import json
import logging
import re
import shlex
import sys
import time
import threading
from typing import Dict, List, Optional

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional dependency
    sd = None

from .config import AppConfig, get_config
from .stt_backends import STTManager, STTWhisper, STTBase
from .system_utils import (
    collect_system_snapshot,
    ai_status_injection_needed,
    _pi_temp,
    run,
    redact,
    clamp,
)
from .led_manager import LEDManager
from .ollama_client import chat_once
from .diagnostics import make_system_prompt, diag_system_prompt
from .memory_manager import ConversationMemory
from .commands import CommandRouter
from .service_registry import ServiceRegistry
import atexit

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

PRONOUN_REF_RE = re.compile(r"^(this|that|it|the above|the last one)\b", re.I)
WHAT_MY_RE = re.compile(r"^\s*(what\s+is\s+my|what's\s+my)\s+(.+?)\s*\??$", re.I)
ABOUT_RE = re.compile(r"\b(what\s+(do|did)\s+you\s+(know|remember|save)\s+(about|regarding)\s+)(.+)$", re.I)
RECALL_RE = re.compile(r"^\s*(recall|remember)\s+(.+)$", re.I)


class BusyIndicator:
    def __init__(self, led_manager: LEDManager, period: float = 0.12):
        self.period = period
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._fi = 0
        self.led_manager = led_manager

    def start(self, label: str = "thinking"):
        if self._thr and self._thr.is_alive():
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, args=(label,), daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=1.0)
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()
        if self.led_manager:
            self.led_manager.set_color("idle")

    def _run(self, label: str):
        while not self._stop.is_set():
            frame = self._frames[self._fi % len(self._frames)]
            self._fi += 1
            sys.stdout.write(f"\r{frame} {label}… ")
            sys.stdout.flush()
            if self.led_manager:
                self.led_manager.pulse("think")
            time.sleep(self.period)


def run_diag_command(*, filter_text: str | None, verbose: bool,
                     current_model: str, current_lang: str) -> str:
    """Run diagnostics and get LLM analysis."""
    snap = collect_system_snapshot()
    payload = "SYSTEM_SNAPSHOT:\n```json\n" + json.dumps(snap, ensure_ascii=False, indent=2) + "\n```\n"

    if filter_text:
        lines = 300 if verbose else 120
        flt = re.sub(r"[^\w@:/\.\-\+]", "", filter_text)
        journal = clamp(
            redact(run(f"journalctl -xb --no-pager -n {lines} -g {shlex.quote(flt)} 2>/dev/null")), 1600
        )
        payload += f"LOG_FILTER: {flt}\nLOG_DATA:\n```\n{journal}\n```\n"
        task = f"Task: Analyze filtered journal for '{flt}'. Summarize root cause and 1–3 concrete actions."
    else:
        task = "Task: Summarize health, top issues, and one concrete action per issue."

    result = chat_once(
        history=[],
        sys_prompt=diag_system_prompt(current_lang),
        user_text=payload + task,
        current_model=current_model,
        deterministic=True,
    )
    return result["reply"]


class VoiceAssistant:
    REMEMBER_RE = re.compile(r"^\s*(remember|save|note|store)\b[:\- ]?(.*)$", re.I)
    REMEMBER_TRAIL_RE = re.compile(r"(.*)\b(remember|save|store|note)\s+(this|that|it)\s*\.?\s*$", re.I)

    def __init__(self, config: AppConfig | None = None, services: ServiceRegistry | None = None):
        self.config = config or get_config()
        self.services = services or ServiceRegistry(self.config)

        self.current_lang = self.config.lang_default
        self.current_model = self.config.default_model
        self.sys_prompt = make_system_prompt(self.current_lang)
        self.history: List[Dict[str, str]] = []

        self.led_manager = self.services.get_led_manager()
        self.busy = BusyIndicator(self.led_manager, period=0.12)
        self.stt_manager: STTManager = self.services.get_stt_manager()
        self.memory: ConversationMemory = self.services.get_memory()
        self.tts = self.services.get_tts()
        self.display = self.services.get_display_manager()
        self.face_follow = self.services.get_face_follow()
        self.scripts = self.services.get_script_manager()
        self.command_router = CommandRouter(self, self.config, run_diag_command)

        self.stt_mode = (self.config.stt_backend or "vosk").lower()
        self.stt_obj: Optional[STTBase] = None
        self.init_stt()

        self._load_memory_primer()
        self.display.splash("Optimus", "Voice system ready")

        if self.config.face_follow_enabled:
            self.face_follow.start()

        atexit.register(self.shutdown)

    # ------------------------------------------------------------------
    def init_stt(self) -> None:
        try:
            if self.stt_mode == "whisper":
                self.stt_obj = self.stt_manager.initialize_backend(
                    "whisper", self.current_lang, self.stt_manager.current_device_index
                )
            elif self.stt_mode == "vosk":
                self.stt_obj = self.stt_manager.initialize_backend(
                    "vosk", self.current_lang, self.stt_manager.current_device_index
                )
            else:
                self.stt_obj = None
                logger.info("STT disabled (mode='%s')", self.stt_mode)
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.warning("STT initialization failed: %s", exc)
            self.stt_obj = None
            self.stt_mode = "off"

    def switch_whisper_language(self, lang: str) -> bool:
        if isinstance(self.stt_obj, STTWhisper):
            self.stt_obj = STTWhisper(
                lang=lang,
                device_index=self.stt_manager.current_device_index,
                config=self.config,
            )
            self.stt_manager.current_backend = self.stt_obj
            return True
        return False

    def set_stt_mode(self, mode: str) -> None:
        if mode not in {"vosk", "whisper", "off"}:
            print("Usage: /stt vosk | whisper | off")
            return
        self.stt_mode = mode
        if self.stt_mode == "off":
            self.stt_obj = None
            print("STT disabled. Type your prompts.")
            return
        try:
            self.init_stt()
            print(f"STT set to {self.stt_mode}.")
        except Exception as exc:
            print(f"[stt-error] {exc}")
            self.stt_mode = "off"
            self.stt_obj = None

    def make_system_prompt(self, lang: str) -> str:
        return make_system_prompt(lang)

    # ------------------------------------------------------------------
    def _load_memory_primer(self) -> None:
        primer = self.memory.load_or_build_primer()
        if primer:
            self.sys_prompt = f"{self.sys_prompt}\n\n## Previous Context:\n{primer}"
            logger.info("Loaded memory primer into system prompt")

    def _enhance_prompt_with_memory(self, user_input: str) -> str:
        memory_context = self.memory.get_memory_context(user_input)
        if not memory_context:
            return self.sys_prompt
        return (
            f"{self.sys_prompt}\n\n"
            f"{memory_context}\n\n"
            "Current Instruction: When appropriate, you may choose to remember important information from this conversation. "
            "Use phrases like \"I'll remember that...\" or \"Noted: ...\" when storing important information. "
            "Respond to the user's current query while being aware of our previous conversation context."
        )

    def _save_to_memory_if_important(self, user_input: str, ai_response: str) -> None:
        try:
            if self.memory.durable_trigger(user_input, ai_response):
                bullets = self.memory.extract_notes(user_input, "", self.history[-6:])
                if bullets:
                    self.memory.add_bullets(bullets)
                    self.memory.append_notes_md([text for text, _ in bullets])
                    self.memory.maybe_rebuild_primer()
                    logger.info("Saved %s notes to memory", len(bullets))
                    primer = self.memory.load_or_build_primer()
                    if primer:
                        self.sys_prompt = f"{make_system_prompt(self.current_lang)}\n\n## Previous Context:\n{primer}"
        except Exception as exc:
            logger.warning("Memory save failed: %s", exc)

    def _maybe_remember_inline(self, user_text: str) -> bool:
        match = self.REMEMBER_RE.match(user_text or "")
        if not match:
            return False
        payload = (match.group(2) or "").strip()
        if not payload or PRONOUN_REF_RE.match(payload):
            prev_user = ""
            for msg in reversed(self.history):
                if msg.get("role") == "user":
                    prev_user = msg.get("content", "").strip()
                    if prev_user:
                        break
            payload = prev_user or payload or "(no previous user text to store)"
        try:
            self.memory.add_bullets([f"Note: {payload}"], confidence=0.7)
            bullets = self.memory.extract_notes(payload, "(remember request)")
            if bullets:
                self.memory.add_bullets(bullets, confidence=0.8)
            self.memory.maybe_rebuild_primer()
        except Exception as exc:
            logger.warning("Inline remember failed: %s", exc)
        msg = "Noted. I’ll remember that."
        print(f"\nOptimus ({self.current_model}): {msg}\n")
        self.display.splash(f"\nOptimus ({self.current_model}): {msg}\n")
        self.tts.speak(msg)
        return True

    def _maybe_remember_trailing(self, user_text: str) -> bool:
        match = self.REMEMBER_TRAIL_RE.match(user_text or "")
        if not match:
            return False
        payload = (match.group(1) or "").strip().strip(",; ")
        if not payload:
            for msg in reversed(self.history):
                if msg.get("role") == "user":
                    payload = msg.get("content", "").strip()
                    if payload:
                        break
        if not payload:
            payload = "(no previous user text to store)"
        try:
            self.memory.add_bullets([f"Note: {payload}"], confidence=0.7)
            bullets = self.memory.extract_notes(payload, "(remember trailing)")
            if bullets:
                self.memory.add_bullets(bullets, confidence=0.8)
            self.memory.maybe_rebuild_primer()
        except Exception as exc:
            logger.warning("Trailing remember failed: %s", exc)
        msg = "Noted. I’ll remember that."
        print(f"\nBOT ({self.current_model}): {msg}\n")
        self.display.splash(f"\nOptimus: {msg}\n")
        self.tts.speak(msg)
        return True

    # ------------------------------------------------------------------
    def process_text(self, user_text: str) -> str:
        lowq = user_text.strip().lower()

        hit = self.scripts.match(user_text)
        if hit:
            ok, out = self.scripts.run(hit.key)
            msg = out.strip() or ("ok" if ok else "error")
            print(f"\nBOT ({self.current_model}): {msg}\n")
            self.display.show_message(f"\nOptimus: {msg}\n")
            self.tts.speak(msg)
            return msg

        if self._maybe_remember_inline(user_text) or self._maybe_remember_trailing(user_text):
            return "OK"

        topic = self._extract_topic(user_text)
        if topic is not None or re.search(r"\b(what\s+did\s+you\s+(remember|save)|what\s+do\s+you\s+know)\b", lowq):
            msg = self._llm_answer_from_memory(topic)
            print(f"\nBOT ({self.current_model}): {msg}\n")
            self.display.show_message(f"\nOptimus: {msg}\n")
            self.tts.speak(msg)
            return msg

        user_for_llm = user_text
        if ai_status_injection_needed(user_text):
            snap = collect_system_snapshot()
            user_for_llm += "\n\nSYSTEM_SNAPSHOT:\n```json\n" + json.dumps(
                snap, ensure_ascii=False, indent=2
            ) + "\n```"

        enhanced_prompt = self._enhance_prompt_with_memory(user_text)

        self.busy.start("thinking")
        try:
            result = chat_once(
                self.history,
                enhanced_prompt,
                user_for_llm,
                self.current_model,
                deterministic=False,
            )
            reply = result["reply"]
            self.history = result["history"][-self.config.max_history_length:]

            print(f"\nBOT ({self.current_model}): {reply}\n")
            self.display.show_message(f"\nOptimus: {reply}\n")
            self.tts.speak(reply)

            self._save_to_memory_if_important(user_text, reply)
            return reply

        except Exception as exc:
            logger.error("Chat failed: %s", exc, exc_info=True)
            msg = f"Error: {exc}"
            print(msg)
            return msg
        finally:
            self.busy.stop()

    def _llm_answer_from_memory(self, query: Optional[str]) -> str:
        if query:
            match = re.match(r"^\s*my\s+([a-z0-9 _\-]{1,40})\s*$", query, flags=re.I)
            if match:
                key = match.group(1).strip().lower()
                key = {"fullname": "name", "user": "name", "handle": "name"}.get(key, key)
                val = self.memory.lookup_kv(key)
                if val:
                    return val
        items = self.memory.recall(query or None, top_k=6)
        if not items:
            return ("I haven’t saved anything yet." if not query
                    else f"I don’t have anything saved about {query}.")
        mem_lines = [f"- {i['text'].replace('Note:', '').strip()}" for i in items]
        mem_block = "\n".join(mem_lines)
        sys_inst = (
            "You are a precise assistant. You have access to a small MEMORY section "
            "containing durable notes persisted across sessions. Answer the user's query "
            "succinctly based ONLY on MEMORY. If MEMORY doesn't contain the answer, say "
            "'Unknown'. Prefer a single clear sentence; don't parrot bullets unless asked."
        )
        user_q = (query or "What did you remember?")
        user_payload = (
            f"MEMORY:\n{mem_block}\n\n"
            f"QUESTION: {user_q}\n\n"
            "RULES:\n"
            "- If MEMORY includes the answer, state it plainly.\n"
            "- If MEMORY has multiple partial hints, synthesize a concise answer.\n"
            "- If answer is not present, reply exactly: Unknown."
        )
        result = chat_once(
            history=[],
            sys_prompt=sys_inst,
            user_text=user_payload,
            current_model=self.current_model,
            deterministic=True,
        )
        reply = (result.get("reply") or "").strip()
        return reply or "Unknown"

    def _extract_topic(self, text: str) -> Optional[str]:
        s = text.strip()
        match = WHAT_MY_RE.match(s)
        if match:
            return "my " + match.group(2).strip()
        match = ABOUT_RE.search(s)
        if match:
            return match.group(5).strip()
        match = RECALL_RE.match(s)
        if match:
            return match.group(2).strip()
        return None

    # ------------------------------------------------------------------
    def handle_command(self, user_input: str) -> bool:
        return self.command_router.handle(user_input)

    def set_language(self, lang: str) -> None:
        self.current_lang = lang
        self.sys_prompt = make_system_prompt(self.current_lang)
        self.history.clear()

    def pi_temperature(self) -> str:
        return _pi_temp()

    def list_audio_devices(self) -> None:
        if sd is None:
            print("sounddevice not installed.")
            return
        try:
            devs = sd.query_devices()
            for idx, device in enumerate(devs):
                print(f"[{idx}] {device['name']}  in={device['max_input_channels']} out={device['max_output_channels']}")
        except Exception as exc:  # pragma: no cover - hardware dependent
            print(f"[audio-error] {exc}")

    def set_microphone(self, arg: str) -> None:
        if sd is None:
            print("sounddevice not installed.")
            return
        idx: Optional[int] = None
        try:
            idx = int(arg)
        except ValueError:
            needle = arg.lower()
            try:
                for i, device in enumerate(sd.query_devices()):
                    if device.get("max_input_channels", 0) > 0 and needle in device.get("name", "").lower():
                        idx = i
                        break
            except Exception as exc:
                print(f"[audio-error] {exc}")
                return
        if idx is None:
            print("Usage: /mic <index|name-substring>")
            return
        self.stt_manager.set_device_index(idx)
        if self.stt_obj:
            self.stt_obj.device_index = idx
        print(f"[mic] Input device set to index {idx}.")

    def manual_remember(self, payload: str) -> None:
        payload = payload.strip()
        if not payload:
            print("Usage: /remember <text to save>")
            return
        try:
            self.memory.add_bullets([f"Note: {payload}"], confidence=0.7)
            bullets = self.memory.extract_notes(payload, "", self.history[-6:])
            if bullets:
                self.memory.add_bullets(bullets)
                self.memory.append_notes_md([text for text, _ in bullets])
                self.memory.maybe_rebuild_primer()
        except Exception as exc:
            logger.warning("/remember failed: %s", exc)
        print("Noted. I’ll remember that.")
        self.tts.speak_chunks(["Noted. I’ll remember that."])

    def manual_recall(self, topic: str) -> None:
        items = self.memory.recall(topic or None, top_k=5)
        if not items:
            print("No saved notes." if not topic else f"No notes about: {topic}")
            return
        print("Saved notes:" if not topic else f"Notes about {topic}:")
        for item in items:
            print(f"- {item['text']}")

    def memory_command(self, user_input: str) -> None:
        parts = user_input.split()
        if len(parts) > 1 and parts[1] == "clear":
            self.memory.clear_memory()
            self.sys_prompt = make_system_prompt(self.current_lang)
            print("Conversation memory cleared.")
        elif len(parts) > 1 and parts[1] == "rebuild":
            primer = self.memory.maybe_rebuild_primer(force=True)
            if primer:
                self.sys_prompt = f"{make_system_prompt(self.current_lang)}\n\n## Previous Context:\n{primer}"
                print("Memory primer rebuilt and loaded.")
            else:
                print("No notes available to rebuild primer.")
        else:
            memory_count = len(self.memory.state.get("notes", []))
            primer = self.memory.state.get("primer", "")
            print(f"Conversation memory: {memory_count} entries")
            if primer:
                print("\nCurrent Primer:")
                print(primer[:600] + ("..." if len(primer) > 600 else ""))
            if memory_count > 0:
                print(f"\nLast {min(5, memory_count)} notes:")
                for i, note in enumerate(self.memory.state["notes"][-5:], 1):
                    print(f"  {i}. [{note.get('ts', '')[:19]}] {note.get('text', '')[:90]}")

    # ------------------------------------------------------------------
    def get_user_input(self) -> str:
        if self.stt_mode in {"vosk", "whisper"} and self.stt_obj is not None:
            print(f"\n[mic] Mode={self.stt_mode}. Speak now… (/stt off to type)")
            if self.led_manager:
                self.led_manager.set_color((0.2, 0.6, 1.0))
            try:
                user = (
                    self.stt_obj.listen_once(secs=self.config.whisper_secs)
                    if self.stt_mode == "whisper"
                    else self.stt_obj.listen_once()
                )
            finally:
                if self.led_manager:
                    self.led_manager.set_color("idle")
            user = (user or "").strip()
            if not user:
                print("[mic] No speech detected. Say again or /stt off to type.")
                return ""
            print(f"> {user}")
            return user
        return input("\n> ").strip()

    def shutdown(self) -> None:
        try:
            if getattr(self, "face_follow", None):
                self.face_follow.stop()
        except Exception:  # pragma: no cover - cleanup best effort
            pass

    def run(self) -> None:
        print("Commands:")
        print("  /stt vosk|whisper|off  -> choose input mode (default: vosk)")
        print("  /secs N                -> whisper record seconds (default:", self.config.whisper_secs, ")")
        print("  /model <name>          -> LLM model (e.g. qwen3:8b)")
        print("  /lang <en|da>          -> reply language")
        print("  /diag [filter] [-v]    -> live diagnostics")
        print("  /logs <svc> [n]        -> journal logs")
        print("  /tail <path> [n]       -> tail file")
        print("  /temp | /time          -> quick facts")
        print("  /clear | /quit         -> housekeeping")
        print("  /follow on|off|status  -> face follow control")
        print("  /scripts list|reload|run <key> -> run mapped script")
        print("  /audio                 -> list PortAudio devices")
        print("  /mic <idx|name>        -> set input mic by index or name substring")
        print("  /remember <text>       -> save a freeform note")
        print("  /recall [topic]        -> show saved notes (optionally filtered)")
        print("  /memory [show|clear|rebuild]")

        while True:
            try:
                user = self.get_user_input()
                if not user:
                    continue
                if user.startswith('/') and self.handle_command(user):
                    continue
                self.process_text(user)
            except KeyboardInterrupt:
                print("\nBye!")
                return
            except Exception as exc:
                if self.led_manager:
                    self.led_manager.set_color("error")
                    time.sleep(0.6)
                    self.led_manager.set_color("idle")
                logger.error("Error in main loop: %s", exc, exc_info=True)
                print("Error:", exc, file=sys.stderr)


def main() -> None:
    assistant = VoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
