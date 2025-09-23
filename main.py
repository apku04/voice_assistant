# voice_assistant/main.py
from __future__ import annotations

import datetime
import json
import logging
import re
import shlex
import sys
import time
import threading
from typing import Dict, List, Optional

from luma.core.interface.serial import i2c
from luma.oled.device import sh1107
from luma.core.render import canvas
from PIL import ImageFont

try:
    import sounddevice as sd
except ImportError:
    sd = None

# Import our modules
from .config import config
from .stt_backends import STTManager, STTWhisper, STTBase
from .system_utils import (
    collect_system_snapshot, get_service_log, tail_file,
    ai_status_injection_needed, _pi_temp, run, redact, clamp
)
from .led_manager import LEDManager
from .piper_tts import PiperTTS
from .ollama_client import chat_once
from .diagnostics import (
    make_system_prompt, diag_system_prompt,
    maybe_offer_diag
)
from .memory_manager import ConversationMemory
from .display_manager import DisplayManager
from .face_follow_manager import FaceFollowManager
from .perception import BehaviorMonitor, LLMMoodInterpreter
from .script_manager import ScriptManager
import atexit

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Put this near the top of the file with the other imports
PRONOUN_REF_RE = re.compile(r"^(this|that|it|the above|the last one)\b", re.I)

# Generic “what is my … / about …” patterns
WHAT_MY_RE   = re.compile(r"^\s*(what\s+is\s+my|what's\s+my)\s+(.+?)\s*\??$", re.I)
ABOUT_RE     = re.compile(r"\b(what\s+(do|did)\s+you\s+(know|remember|save)\s+(about|regarding)\s+)(.+)$", re.I)
RECALL_RE    = re.compile(r"^\s*(recall|remember)\s+(.+)$", re.I)



# ---------- Busy indicator (spinner + LED pulse) ----------
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


# ---------- Diagnostics helper ----------
def run_diag_command(*, filter_text: str | None, verbose: bool,
                     current_model: str, current_lang: str) -> str:
    """Run diagnostics and get LLM analysis"""
    snap = collect_system_snapshot()
    payload = "SYSTEM_SNAPSHOT:\n```json\n" + json.dumps(snap, ensure_ascii=False, indent=2) + "\n```\n"

    if filter_text:
        lines = 300 if verbose else 120
        flt = re.sub(r"[^\w@:/\.\-\+]", "", filter_text)
        j = clamp(redact(run(f"journalctl -xb --no-pager -n {lines} -g {shlex.quote(flt)} 2>/dev/null")), 1600)
        payload += f"LOG_FILTER: {flt}\nLOG_DATA:\n```\n{j}\n```\n"
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


# ---------- Main application ----------
class VoiceAssistant:
    """
    Voice assistant with:
      - Generic 'remember/save/note' capture (/remember and inline)
      - Cross-session recall (/recall or natural questions)
      - Memory-aware prompting (inject relevant notes)
      - Diagnostics (/diag, /logs, /tail)
      - STT control (/stt vosk|whisper|off, /secs N, /mic, /audio)
    """

    # Inline remember trigger (generic, language-agnostic basic)
    REMEMBER_RE = re.compile(r"^\s*(remember|save|note|store)\b[:\- ]?(.*)$", re.I)
    REMEMBER_TRAIL_RE = re.compile(r"(.*)\b(remember|save|store|note)\s+(this|that|it)\s*\.?\s*$", re.I)

    def __init__(self):
        self.current_lang = config.lang_default
        self.current_model = config.default_model
        self.sys_prompt = make_system_prompt(self.current_lang)
        self.history: List[Dict[str, str]] = []

        # Components
        self.led_manager = LEDManager()
        self.busy = BusyIndicator(self.led_manager, period=0.12)
        self.stt_manager = STTManager()

        # Memory
        self.memory = ConversationMemory()

        # TTS
        self.tts = PiperTTS(
            led_manager=self.led_manager,
            max_chunks=config.max_chunks,
            max_chars=config.max_chars,
            pause_s=config.pause_s,
            prefix_zwsp=True,
        )

        # STT backend
        self.stt_mode = (config.stt_backend or "vosk").lower()
        self.stt_obj: Optional[STTBase] = None
        self.init_stt()

        # Memory primer
        self._load_memory_primer()

        self.display = DisplayManager()
        self.display.splash("Optimus", "Voice system ready")

        # Face follow (USB cam + steppers) as background service
        self.face_follow = FaceFollowManager(display=self.display)
        if getattr(config, "face_follow_enabled", True):
            self.face_follow.start()
        self.mood_interpreter = (
            LLMMoodInterpreter()
            if getattr(config, "perception_llm_enabled", False)
            else None
        )
        self.behavior = BehaviorMonitor(
            self.face_follow,
            mood_interpreter=self.mood_interpreter,
            llm_passthrough=getattr(config, "perception_llm_enabled", False),
        )

        # Script manager (yaml-configured small actions)
        self.scripts = ScriptManager()

        # Ensure graceful shutdown
        atexit.register(self.shutdown)

    # ---------- init ----------
    def init_stt(self):
        """Initialize STT backend according to current mode and language."""
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
                logger.info(f"STT disabled (mode='{self.stt_mode}')")
        except Exception as e:
            logger.warning(f"STT initialization failed: {e}")
            self.stt_obj = None
            self.stt_mode = "off"

    def _llm_answer_from_memory(self, query: str | None) -> str:
        """Compose an answer from persisted notes using the LLM deterministically."""
        # 1) fetch relevant notes (or latest if no query)
        if query:
            m = re.match(r"^\s*my\s+([a-z0-9 _\-]{1,40})\s*$", query, flags=re.I)
            if m:
                key = m.group(1).strip().lower()
                key = {"fullname": "name", "user": "name", "handle": "name"}.get(key, key)
                val = self.memory.lookup_kv(key)
                if val:
                    return val

        items = self.memory.recall(query or None, top_k=6)
        if not items:
            return ("I haven’t saved anything yet." if not query
                    else f"I don’t have anything saved about {query}.")

        # 2) build context for the model
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

        # 3) deterministic call (no randomness)
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
        """Pull a topic/slot from natural questions like:
           'what is my wifi password', 'what do you know about telescope azimuth', 'recall ssh config'."""
        s = text.strip()
        m = WHAT_MY_RE.match(s)
        if m:
            # e.g. "wifi password" -> topic "my wifi password"
            return "my " + m.group(2).strip()
        m = ABOUT_RE.search(s)
        if m:
            return m.group(5).strip()
        m = RECALL_RE.match(s)
        if m:
            return m.group(2).strip()
        return None

    def _recall_answer(self, query: Optional[str]) -> str:
        """Query memory and render a concise answer."""
        items = self.memory.recall(query or None, top_k=5)
        if not items:
            return ("I haven’t saved anything yet." if not query
                    else f"I don’t have anything saved about {query}.")
        # If user asked “what is my X …” try to return a single best line
        if query and query.lower().startswith("my "):
            m = re.match(r"^\s*my\s+([a-z0-9 _\-]{1,40})\s*$", query, flags=re.I)
            if m:
                key = m.group(1).strip().lower()
                key = {"fullname": "name", "user": "name", "handle": "name"}.get(key, key)
                val = self.memory.lookup_kv(key)
                if val:
                    return val
        # Otherwise show top matches as bullets
        lines = [f"- {i['text'].replace('Note:', '').strip()}" for i in items]
        return ("Saved notes:" if not query else f"Notes about {query}:") + "\n" + "\n".join(lines)


    def _load_memory_primer(self):
        """Load memory primer and add it to system prompt."""
        primer = self.memory.load_or_build_primer()
        if primer:
            self.sys_prompt = f"{self.sys_prompt}\n\n## Previous Context:\n{primer}"
            logger.info("Loaded memory primer into system prompt")

    # ---------- memory helpers ----------
    def _enhance_prompt_with_memory(self, user_input: str) -> str:
        """Add relevant notes (top-k) into the system prompt for this turn."""
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
        """
        Persist durable details using generic extractor (notes + primer).
        IMPORTANT: Only consider the USER text as a source of truth.
        """
        try:
            if self.memory.durable_trigger(user_input, ai_response):
                # Extract from USER text only; reply is NOT a source of facts.
                bullets = self.memory.extract_notes(user_input, "", self.history[-6:])
                if bullets:
                    # bullets are tuples (text, confidence)
                    self.memory.add_bullets(bullets)
                    # notes.md expects strings
                    self.memory.append_notes_md([t for t, _ in bullets])
                    self.memory.maybe_rebuild_primer()
                    logger.info(f"Saved {len(bullets)} notes to memory")

                    # Refresh primer in system prompt for next turns
                    primer = self.memory.load_or_build_primer()
                    if primer:
                        self.sys_prompt = (
                            f"{make_system_prompt(self.current_lang)}\n\n## Previous Context:\n{primer}"
                        )
        except Exception as e:
            logger.warning(f"Memory save failed: {e}")



    def _maybe_remember_inline(self, user_text: str) -> bool:
        """
        If the user starts with 'remember/save/note/store', persist it.
        - If payload is empty or starts with 'this/that/it...', we capture the
          most recent **user** message from history (not the assistant).
        - We also run the extractor to turn freeform into durable bullets.
        """
        m = self.REMEMBER_RE.match(user_text or "")
        if not m:
            return False

        payload = (m.group(2) or "").strip()

        # If nothing explicit, or refers to "this/that/it", use last USER msg
        if not payload or PRONOUN_REF_RE.match(payload):
            prev_user = ""
            for msg in reversed(self.history):
                if msg.get("role") == "user":
                    prev_user = msg.get("content", "").strip()
                    if prev_user:
                        break
            payload = prev_user or payload or "(no previous user text to store)"

        # Persist: raw note + extracted bullets
        try:
            # Always keep a raw line as fallback
            self.memory.add_bullets([f"Note: {payload}"], confidence=0.7)
            # Ask extractor for durable bullets (names, prefs, paths, pins, etc.)
            bullets = self.memory.extract_notes(payload, "(remember request)")
            if bullets:
                self.memory.add_bullets(bullets, confidence=0.8)
            self.memory.maybe_rebuild_primer()
        except Exception as e:
            logger.warning(f"Inline remember failed: {e}")

        msg = "Noted. I’ll remember that."
        print(f"\nOptimus ({self.current_model}): {msg}\n")
        self.display.splash(f"\nOptimus ({self.current_model}): {msg}\n")
        self.tts.speak(msg)
        return True

    def _maybe_remember_trailing(self, user_text: str) -> bool:
        """
        If the user ends a sentence with '..., remember this/that/it', persist the
        preceding content. Returns True if handled.
        """
        m = self.REMEMBER_TRAIL_RE.match(user_text or "")
        if not m:
            return False

        payload = (m.group(1) or "").strip().strip(",; ")
        if not payload:
            # fallback: last user message from history
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
        except Exception as e:
            logger.warning(f"Trailing remember failed: {e}")

        msg = "Noted. I’ll remember that."
        print(f"\nBOT ({self.current_model}): {msg}\n")
        self.display.splash(f"\nOptimus: {msg}\n")
        self.tts.speak(msg)
        return True

    # ---------- main turn ----------
    def process_text(self, user_text: str) -> str:
        """
        One turn:
          1) Save requests (“remember/save/store/note …”, incl. trailing “… remember this”)
          2) Perception queries (face/hand status handled without the main LLM)
          3) Recall requests (“what is my X / recall Y / what did you save”)
          4) Normal LLM turn (+ optional diagnostics injection)
          5) Save durable user facts from this turn (user text only)
        """
        lowq = user_text.strip().lower()

        # 0) QUICK SCRIPTS (phrase-mapped actions)
        hit = self.scripts.match(user_text)
        if hit:
            ok, out = self.scripts.run(hit.key)
            msg = out.strip() or ("ok" if ok else "error")
            print(f"\nBOT ({self.current_model}): {msg}\n")
            self.display.show_message(f"\nOptimus: {msg}\n")
            self.tts.speak(msg)
            return msg

        # 1) SAVE FIRST (so "remember this" doesn't get misrouted to recall)
        if self._maybe_remember_inline(user_text) or self._maybe_remember_trailing(user_text):
            return "OK"

        # 2) PERCEPTION QUERIES (face/hand status without LLM)
        wants_mood = wants_hands = False
        snapshot = None
        perception_reply = None
        if self.behavior:
            wants_mood, wants_hands = self.behavior.requirements(user_text)
            if wants_mood or wants_hands:
                snapshot = self.behavior.snapshot()
                perception_reply = self.behavior.answer_query(user_text, snapshot)
                if perception_reply:
                    self._announce_perception(perception_reply)
                    return perception_reply

        # 3) RECALL (generic, model-synthesized from top-k notes)
        topic = self._extract_topic(user_text)
        if topic is not None or re.search(r"\b(what\s+did\s+you\s+(remember|save)|what\s+do\s+you\s+know)\b", lowq):
            msg = self._llm_answer_from_memory(topic)
            print(f"\nBOT ({self.current_model}): {msg}\n")
            self.display.show_message(f"\nOptimus: {msg}\n")
            self.tts.speak(msg)
            return msg

        # 4) NORMAL CHAT (with optional diagnostics injection)
        user_for_llm = user_text
        perception = ""
        if wants_mood or wants_hands:
            perception = self.behavior.build_context(
                snapshot,
                include_mood=wants_mood,
                include_hands=wants_hands,
            )
        if perception:
            user_for_llm += "\n\n[PERCEPTION]\n" + perception
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
            # Bound history
            self.history = result["history"][-config.max_history_length:]

            print(f"\nBOT ({self.current_model}): {reply}\n")
            self.display.show_message(f"\nOptimus: {reply}\n")
            self.tts.speak(reply)

            # 5) SAVE DURABLE FACTS (user text only; do not learn from the bot's own words)
            self._save_to_memory_if_important(user_text, reply)
            return reply

        except Exception as e:
            logger.error(f"Chat failed: {e}", exc_info=True)
            msg = f"Error: {e}"
            print(msg)
            return msg
        finally:
            self.busy.stop()

    def _announce_perception(self, text: str, *, speak: bool = True) -> None:
        print(f"\nBOT ({self.current_model}): {text}\n")
        if self.display:
            self.display.show_message(f"\nOptimus: {text}\n")
        if speak and self.tts:
            self.tts.speak(text)

    # ---------- commands ----------
    def handle_command(self, user_input: str) -> bool:
        """
        Return True if the command was handled, else False
        (so caller can process it as normal text).
        """
        low = user_input.strip().lower()

        # Quit
        if low in {"q", "/q", "/quit", "exit"}:
            print("Bye!")
            return True

        # History
        if low in {"/clear", "/reset"}:
            self.history.clear()
            print("History cleared.")
            return True

        # Model
        if low.startswith("/model "):
            self.current_model = user_input.split(" ", 1)[1].strip() or self.current_model
            self.history.clear()
            print(f"Model set to: {self.current_model}. History cleared.")
            return True

        # Face follow control
        if low.startswith("/follow"):
            parts = user_input.split()
            sub = parts[1].lower() if len(parts) > 1 else "status"
            if sub in {"on", "start"}:
                ok = self.face_follow.start()
                print("Face follow:", "started" if ok else "failed")
            elif sub in {"off", "stop"}:
                self.face_follow.stop()
                print("Face follow: stopped")
            else:
                print("Face follow:", "running" if self.face_follow.is_running() else "stopped")
                print("Usage: /follow on|off|status")
            return True

        # Language
        if low.startswith("/lang "):
            arg = user_input.split(" ", 1)[1].strip().lower()
            if arg in {"en", "da"}:
                self.current_lang = arg
                self.sys_prompt = make_system_prompt(self.current_lang)
                self.history.clear()
                print(f"Language set to {self.current_lang.upper()}. History cleared.")
                # Re-init Whisper if active
                if isinstance(self.stt_obj, STTWhisper):
                    self.stt_obj = STTWhisper(
                        lang=self.current_lang,
                        device_index=self.stt_manager.current_device_index
                    )
            else:
                print("Usage: /lang en | da")
            return True

        # Whisper seconds
        if low.startswith("/secs "):
            try:
                val = int(user_input.split(" ", 1)[1].strip())
                assert 1 <= val <= 20
                config.whisper_secs = val
                print(f"Whisper record seconds set to {config.whisper_secs}.")
            except Exception:
                print("Usage: /secs 3..20")
            return True

        # STT mode
        if low.startswith("/stt "):
            mode = user_input.split(" ", 1)[1].strip().lower()
            if mode in {"vosk", "whisper", "off"}:
                self.stt_mode = mode
                if self.stt_mode == "off":
                    self.stt_obj = None
                    print("STT disabled. Type your prompts.")
                else:
                    try:
                        self.init_stt()
                        print(f"STT set to {self.stt_mode}.")
                    except Exception as e:
                        print(f"[stt-error] {e}")
                        self.stt_mode = "off"
                        self.stt_obj = None
            else:
                print("Usage: /stt vosk | whisper | off")
            return True

        # Quick facts
        if low.strip() in {"/temp", "/temperature"} or re.search(r"\b(cpu|soc|board)?\s*temp\w*\b", low, re.I):
            msg = f"CPU temperature: {_pi_temp()}"
            print(msg)
            self.tts.speak_chunks([msg])
            return True

        if low.strip() == "/time" or re.search(r"\b(system\s+)?time\b", low):
            now = datetime.datetime.now()
            msg = f"System time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
            print(msg)
            self.tts.speak_chunks([msg])
            return True

        # Diagnostics
        if low.startswith("/diag"):
            parts = user_input.split()
            flt = None
            verbose = False
            for tok in parts[1:]:
                if tok == "-v":
                    verbose = True
                elif not tok.startswith("-"):
                    flt = tok

            self.busy.start("analyzing")
            try:
                reply = run_diag_command(
                    filter_text=flt,
                    verbose=verbose,
                    current_model=self.current_model,
                    current_lang=self.current_lang,
                )
                print(f"\nBOT ({self.current_model}): {reply}\n")
                self.display.show_message(f"\nOptimus: {reply}\n")
                self.tts.speak(reply)

                sug = maybe_offer_diag(reply, allow=True)
                if sug:
                    q = f"Run diagnostics{(' for ' + sug.get('filter')) if sug.get('filter') else ''}{' (verbose)' if sug.get('verbose') else ''} now?"
                    print(f"\n[Prompt] {q} (yes/no)")
                    self.tts.speak_chunks([q])
            finally:
                self.busy.stop()
            return True

        if low.startswith("/logs "):
            parts = user_input.split()
            svc = parts[1] if len(parts) >= 2 else ""
            n = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else config.default_log_lines
            if not svc:
                print("Usage: /logs <service> [lines]")
                return True

            log_txt = get_service_log(svc, n)
            snap = collect_system_snapshot()
            payload = (
                "SYSTEM_SNAPSHOT:\n```json\n"
                + json.dumps(snap, ensure_ascii=False, indent=2)
                + "\n```\n"
                "LOG_DATA:\n```\n"
                + log_txt
                + "\n```\n"
                "Task: Identify likely root cause and propose up to 3 precise actions."
            )

            self.busy.start("analyzing")
            try:
                result = chat_once(
                    history=[],
                    sys_prompt=diag_system_prompt(self.current_lang),
                    user_text=payload,
                    current_model=self.current_model,
                    deterministic=True,
                )
                reply = result["reply"]
                print(f"\nBOT ({self.current_model}): {reply}\n")
                self.display.show_message(f"\nOptimus: {reply}\n")
                self.tts.speak(reply)
            finally:
                self.busy.stop()
            return True

        if low.startswith("/tail "):
            parts = user_input.split()
            path = parts[1] if len(parts) >= 2 else ""
            n = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else config.default_log_lines
            if not path:
                print("Usage: /tail /var/log/<file> [lines]")
                return True

            log_txt = tail_file(path, n)
            payload = "LOG_DATA:\n```\n" + log_txt + "\n```\nTask: Summarize errors and the most likely root cause."

            self.busy.start("analyzing")
            try:
                result = chat_once(
                    history=[],
                    sys_prompt=diag_system_prompt(self.current_lang),
                    user_text=payload,
                    current_model=self.current_model,
                    deterministic=True,
                )
                reply = result["reply"]
                print(f"\nBOT ({self.current_model}): {reply}\n")
                self.display.show_message(f"\nOptimus: {reply}\n")
                self.tts.speak(reply)
            finally:
                self.busy.stop()
            return True

        # Audio devices (requires optional sounddevice import as 'sd')
        if low.strip() == "/audio":
            if sd is None:
                print("sounddevice not installed.")
                return True
            try:
                devs = sd.query_devices()
                for i, d in enumerate(devs):
                    print(f"[{i}] {d['name']}  in={d['max_input_channels']} out={d['max_output_channels']}")
            except Exception as e:
                print(f"[audio-error] {e}")
            return True

        # Scripts control
        if low.startswith("/scripts"):
            parts = user_input.split()
            sub = parts[1] if len(parts) > 1 else "list"
            if sub == "list":
                ents = self.scripts.list_entries()
                print("Scripts:")
                for e in ents:
                    print(f"- {e.key}: {', '.join(e.phrases)} -> {e.script} {e.args}")
            elif sub == "reload":
                self.scripts.reload()
                print("Scripts reloaded.")
            elif sub == "run" and len(parts) > 2:
                ok, out = self.scripts.run(parts[2])
                print(out)
                if ok:
                    self.tts.speak_chunks([out])
            else:
                print("Usage: /scripts list | reload | run <key>")
            return True

        if low.startswith("/mic "):
            if sd is None:
                print("sounddevice not installed.")
                return True
            arg = user_input.split(" ", 1)[1].strip()
            idx = None
            try:
                idx = int(arg)
            except ValueError:
                s = arg.lower()
                try:
                    for i, d in enumerate(sd.query_devices()):
                        if d.get("max_input_channels", 0) > 0 and s in d.get("name", "").lower():
                            idx = i
                            break
                except Exception as e:
                    print(f"[audio-error] {e}")

            if idx is None:
                print("Usage: /mic <index|name-substring>")
                return True

            self.stt_manager.set_device_index(idx)
            if self.stt_obj:
                self.stt_obj.device_index = idx
            print(f"[mic] Input device set to index {idx}.")
            return True

        # Memory commands (manual)
        if low.startswith("/remember "):
            payload = user_input.split(" ", 1)[1].strip()
            try:
                self.memory.add_bullets([f"Note: {payload}"], confidence=0.7)
                # try structured extraction too
                bullets = self.memory.extract_notes(user_input, "", self.history[-6:])
                if bullets:
                    self.memory.add_bullets(bullets)  # tuples ok
                    self.memory.append_notes_md([t for t, _ in bullets])  # md needs strings
                    self.memory.maybe_rebuild_primer()
            except Exception as e:
                logger.warning(f"/remember failed: {e}")
            print("Noted. I’ll remember that.")
            self.tts.speak_chunks(["Noted. I’ll remember that."])
            return True

        if low.strip() == "/remember":
            print("Usage: /remember <text to save>")
            return True

        if low.startswith("/recall"):
            topic = user_input.split(" ", 1)[1].strip() if " " in low else ""
            items = self.memory.recall(topic or None, top_k=5)
            if not items:
                print("No saved notes." if not topic else f"No notes about: {topic}")
            else:
                print("Saved notes:" if not topic else f"Notes about {topic}:")
                for i in items:
                    print(f"- {i['text']}")
            return True

        # Memory management
        if low.startswith("/memory"):
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
            return True

        return False

    # ---------- I/O ----------
    def get_user_input(self) -> str:
        """Get user input via STT or keyboard based on current mode."""
        if self.stt_mode in {"vosk", "whisper"} and self.stt_obj is not None:
            #print(f"\n[mic] Mode={self.stt_mode}. Speak now… (/stt off to type)")
            if self.led_manager:
                self.led_manager.set_color((0.2, 0.6, 1.0))
            try:
                user = (
                    self.stt_obj.listen_once(secs=config.whisper_secs)
                    if self.stt_mode == "whisper"
                    else self.stt_obj.listen_once()
                )
            finally:
                if self.led_manager:
                    self.led_manager.set_color("idle")
            user = (user or "").strip()
            if not user:
                #print("[mic] No speech detected. Say again or /stt off to type.")
                return ""
            print(f"> {user}")
            return user
        else:
            return input("\n> ").strip()

    # ---------- shutdown ----------
    def shutdown(self):
        try:
            if getattr(self, "face_follow", None):
                self.face_follow.stop()
        except Exception:
            pass

    # ---------- CLI loop ----------
    def run(self):
        print("Commands:")
        print("  /stt vosk|whisper|off  -> choose input mode (default: vosk)")
        print("  /secs N                -> whisper record seconds (default:", config.whisper_secs, ")")
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

                # Normal chat turn
                self.process_text(user)

            except KeyboardInterrupt:
                print("\nBye!")
                return
            except Exception as e:
                if self.led_manager:
                    self.led_manager.set_color("error")
                    time.sleep(0.6)
                    self.led_manager.set_color("idle")
                logger.error(f"Error in main loop: {e}", exc_info=True)
                print("Error:", e, file=sys.stderr)



def main():
    va = VoiceAssistant()
    va.run()


if __name__ == "__main__":
    main()

    
def _cleanup():
    pass
