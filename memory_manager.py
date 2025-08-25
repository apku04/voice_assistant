# voice_assistant/memory_manager.py
# Run (optional demo):  python -m voice_assistant.memory_manager
# Integrates with your existing Ollama client + config.
from __future__ import annotations
import os, json, datetime, pathlib, re
from typing import List, Dict, Any, Iterable, Tuple

from .config import config
from .ollama_client import chat_once

HOME      = pathlib.Path.home()
MEM_DIR   = pathlib.Path(os.environ.get("MEMORY_DIR", str(HOME / ".optimus")))
MEM_JSON  = pathlib.Path(os.environ.get("MEMORY_FILE", str(MEM_DIR / "memory.json")))
NOTES_MD  = pathlib.Path(os.environ.get("NOTES_FILE",  str(MEM_DIR / "notes.md")))
MEM_DIR.mkdir(parents=True, exist_ok=True)

MAX_NOTES            = 400
PRIMER_TARGET_CHARS  = 1600
PRIMER_MAX_AGE_DAYS  = 30
PRIMER_REBUILD_SECS  = 6 * 3600
RELEVANT_TOP_K       = 6
EXTRACT_MAX_BULLETS  = 5

def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")

def _tokenize(s: str) -> List[str]:
    s = s.casefold()  # unicode-friendly lowercase
    # keep word chars (unicode), numbers, a few useful symbols in paths/addresses
    toks = re.findall(r"[\w\-./:@]+", s, flags=re.UNICODE)
    stop = {
        "the","a","an","and","or","but","to","of","for","in","on","with","at","by","is","it","this","that",
        "det","og","at","en","et","til","af","på","i","er","som","der"
    }
    return [t for t in toks if t not in stop]


def _overlap_score(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb: return 0.0
    inter = len(sa & sb)
    return inter / (len(sa) ** 0.5 * len(sb) ** 0.5)

PRIMER_SYS = """You are a terse system summarizer.
Input: bullet notes captured across sessions for the same user and project.
Task:
- Produce a compact "Context Primer" (<= 120 words; bullets allowed).
- Keep enduring facts: hardware, wiring/pins, I2C/SPI addresses, models, file paths, configs, goals, TODOs.
- Omit chit-chat, stale dates, and private info not needed for function.
Output ONLY the primer text.
"""

EXTRACT_SYS = """Extract ONLY durable facts that were explicitly stated in the latest exchange.

Rules:
- If the fact is NOT explicitly in the text, DO NOT include it.
- Do NOT generalize or infer missing details.
- Do NOT include transient telemetry (uptime, CPU%, load, memory%, disk%).
- If there are no explicit durable facts, reply EXACTLY: NONE

Output format, strictly:
BULLETS:
- <fact text> | confidence=<0.0–1.0>
"""

# Pick a concise durable span from the provided text (no history).
SELECT_SYS = """You select durable USER facts to store.

Rules:
- Choose ONLY content explicitly present in INPUT_TEXT.
- Prefer a single concise fact that would be useful later (name, title, path, address, credential placeholder, config).
- Do NOT invent or generalize. If nothing durable, reply EXACTLY: NONE.

Output JSON:
{"text":"<verbatim or minimally cleaned user fact>","confidence":0..1}
"""

def _select_span_from_text(self, text: str) -> tuple[str, float] | None:
    """Return (text, confidence) chosen from `text`, or None."""
    if not text or len(text.strip()) < 6:
        return None
    try:
        reply = _ollama_chat(SELECT_SYS, f"INPUT_TEXT:\n{text}", model=config.default_model,
                             temperature=0.0, deterministic=True)
    except Exception:
        return None
    if re.fullmatch(r"NONE", (reply or "").strip(), re.I):
        return None
    m = re.search(r"\{.*\}", reply or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        span = (obj.get("text") or "").strip()
        conf = float(obj.get("confidence", 0.75))
        if not span:
            return None
        # Ground against the source text (must share tokens)
        grounded = self._filter_bullets_against_source([(span, conf)], source_text=text)
        return grounded[0] if grounded else None
    except Exception:
        return None


def _ollama_chat(system: str, user: str, model: str | None = None,
                 temperature: float = 0.2, deterministic: bool = True) -> str:
    """Thin wrapper over chat_once to run a single turn and return reply text."""
    res = chat_once(
        history=[],
        sys_prompt=system,
        user_text=user,
        current_model=model or config.default_model,
        deterministic=deterministic,
    )
    return (res.get("reply") or "").strip()

class ConversationMemory:
    _DURABLE_PATS = [
        r"\bGPIO\d+\b", r"\bI2C\b", r"\bSPI\b", r"\bTMC2209\b", r"\bSSD1306\b", r"\bHAT\b",
        r"/[A-Za-z0-9._\-/]+", r"\b\.py\b", r"\b\.ino\b", r"\bTODO\b", r"\bpin\b",
        r"\bmodel\b", r"\bvoice\b", r"\bALSA\b", r"\bdevice\b", r"\binstall\b",
        r"\bwiring\b", r"\baddress\b", r"\b0x[0-9A-Fa-f]+\b", r"\bPiper\b", r"\bVosk\b",
        r"\bfaster[- ]whisper\b", r"\bHailo\b", r"\bpan[- ]tilt\b"
    ]
    _REMEMBER_PATS = r"\b(remember|note|save this|persist|pinout|wiring|config|store)\b"

    def __init__(self,
                 memory_file: str | os.PathLike = MEM_JSON,
                 notes_file: str | os.PathLike = NOTES_MD,
                 max_notes: int = MAX_NOTES,
                 primer_target_chars: int = PRIMER_TARGET_CHARS,
                 primer_max_age_days: int = PRIMER_MAX_AGE_DAYS,
                 primer_rebuild_secs: int = PRIMER_REBUILD_SECS):
        self.memory_path = pathlib.Path(memory_file)
        self.notes_path  = pathlib.Path(notes_file)
        self.max_notes   = max_notes
        self.primer_target_chars = primer_target_chars
        self.primer_max_age_days = primer_max_age_days
        self.primer_rebuild_secs = primer_rebuild_secs
        self.state: Dict[str, Any] = self._load_state()

    # ---- persistence
    def _default_state(self) -> Dict[str, Any]:
        return {"notes": [], "primer": "", "primer_ts": ""}

    def _load_state(self) -> Dict[str, Any]:
        if self.memory_path.exists():
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "notes" in data and isinstance(data["notes"], list):
                        return data
            except Exception:
                pass
        return self._default_state()

    def _save_state(self) -> None:
        tmp = self.memory_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        tmp.replace(self.memory_path)

    def clear_memory(self) -> None:
        self.state = self._default_state()
        self._save_state()
        try:
            self.notes_path.write_text("", encoding="utf-8")
        except Exception:
            pass

    def append_notes_md(self, bullets: List[str]) -> None:
        stamp = _now_iso()
        lines = [f"### {stamp}", *[f"- {b}" for b in bullets], ""]
        with open(self.notes_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def add_bullets(self, bullets: Iterable[str | Tuple[str, float]], confidence: float = 0.75) -> None:
        notes: List[Dict[str, Any]] = self.state.get("notes", [])
        seen = set(n.get("text", "").strip().lower() for n in notes)
        for b in bullets:
            if isinstance(b, tuple):
                line, conf = b
            else:
                line, conf = b, confidence
            key = line.strip().lower()
            if key in seen:
                continue
            notes.append({"ts": _now_iso(), "text": line.strip(), "confidence": float(conf)})
            seen.add(key)
        if len(notes) > self.max_notes:
            notes[:] = notes[-self.max_notes:]
        self.state["notes"] = notes
        self._save_state()

    # ---- primer
    def load_or_build_primer(self) -> str:
        if self.state.get("primer"):
            return self.state["primer"]
        return self._rebuild_primer()

    def maybe_rebuild_primer(self, force: bool=False) -> str:
        if force:
            return self._rebuild_primer()
        last_s = self.state.get("primer_ts","")
        try:
            last = datetime.datetime.fromisoformat(last_s) if last_s else None
        except Exception:
            last = None
        need = not self.state.get("primer") or last is None
        if not need and (datetime.datetime.now() - last).total_seconds() > self.primer_rebuild_secs:
            need = True
        if self.state.get("notes") and len(self.state["notes"]) % 25 == 0:
            need = True
        return self._rebuild_primer() if need else self.state.get("primer","")

    def _rebuild_primer(self) -> str:
        notes = self.state.get("notes", [])
        cutoff = datetime.datetime.now() - datetime.timedelta(days=self.primer_max_age_days)
        keep: List[str] = []
        for n in notes[-self.max_notes:]:
            try:
                ts = datetime.datetime.fromisoformat(n.get("ts",""))
            except Exception:
                ts = datetime.datetime.now()
            if ts < cutoff: 
                continue
            if n.get("confidence", 0.0) < 0.6:
                continue
            txt = n.get("text","").strip()
            if txt:
                keep.append(f"- {txt}")
        if not keep:
            self.state["primer"] = ""
            self.state["primer_ts"] = _now_iso()
            self._save_state()
            return ""
        joined = "\n".join(keep)
        try:
            out = _ollama_chat(PRIMER_SYS, joined, model=config.default_model, temperature=0.2)
        except Exception:
            out = joined[:self.primer_target_chars]
        primer = (out or "").strip()[:self.primer_target_chars]
        self.state["primer"] = primer
        self.state["primer_ts"] = _now_iso()
        self._save_state()
        return primer

    # ---- relevance
    def get_memory_context(self, user_input: str) -> str:
        notes = self.state.get("notes", [])
        if not notes: return ""
        q = _tokenize(user_input)
        scored: List[Tuple[float, str]] = []
        for n in notes:
            t = n.get("text","")
            score = _overlap_score(q, _tokenize(t)) * (0.5 + 0.5 * n.get("confidence",0.5))
            scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [t for s,t in scored[:RELEVANT_TOP_K] if s > 0.0]
        return "Relevant notes:\n" + "\n".join(f"- {t}" for t in top) if top else ""

    # ---- triggers & extraction
    def durable_trigger(self, user_text: str, reply_text: str) -> bool:
        # learn from the USER unless they explicitly say remember in reply (rare)
        if re.search(self._REMEMBER_PATS, user_text, re.I):
            return True
        for pat in self._DURABLE_PATS:
            if re.search(pat, user_text, re.I):
                return True
        return False

    def extract_notes(
            self,
            user_text: str,
            reply_text: str,
            history_window: List[Dict[str, str]] | None = None
    ) -> list[tuple[str, float]]:
        ctx: List[str] = []
        if history_window:
            for m in history_window[-6:]:
                role = m.get("role", "").upper()[:9]
                txt = m.get("content", "")
                ctx.append(f"{role}: {txt}")
        block = "\n".join(ctx + [f"USER(last): {user_text}", f"ASSISTANT(last): {reply_text}"])
        try:
            out = _ollama_chat(EXTRACT_SYS, block, model=config.default_model, temperature=0.2, deterministic=True)
        except Exception:
            return self._heuristic_extract(user_text, reply_text)

        out = (out or "").strip()
        if re.search(r"\bNONE\b", out, re.I):
            return []

        notes: List[tuple[str, float]] = []
        for line in out.splitlines():
            if not line.strip().startswith(("-", "*")):
                continue
            txt = re.sub(r"^\s*[-*]\s*", "", line).strip()
            m = re.search(r"\|\s*confidence\s*=\s*([0-9.]+)", txt, re.I)
            if m:
                conf = float(m.group(1))
                txt = re.sub(r"\|\s*confidence\s*=[0-9.]+", "", txt).strip()
            else:
                conf = 0.7  # fallback
            if 4 <= len(txt) <= 240:
                notes.append((txt, conf))

        # Ground against actual source (user+reply) to avoid hallucinated bullets
        source = f"{user_text}\n{reply_text}"
        grounded = self._filter_bullets_against_source(notes[:EXTRACT_MAX_BULLETS], source_text=source)
        return grounded

    def _heuristic_extract(self, user_text: str, reply_text: str) -> list[tuple[str, float]]:
        lines: List[tuple[str, float]] = []
        text = user_text + "\n" + reply_text
        for m in re.findall(r"(/[A-Za-z0-9._\-/]+)", text):
            if len(m) <= 120 and (m, 0.7) not in lines:
                lines.append((f"Path: {m}", 0.7))
        for m in re.findall(r"\b0x[0-9A-Fa-f]+\b", text):
            tup = (f"I2C/SPI addr: {m}", 0.7)
            if tup not in lines:
                lines.append(tup)
        for m in re.findall(r"\bGPIO\d+\b", text):
            tup = (f"GPIO used: {m}", 0.7)
            if tup not in lines:
                lines.append(tup)
        for key in ["TMC2209", "SSD1306", "Piper", "Vosk", "Hailo", "pan-tilt"]:
            if re.search(fr"\b{key}\b", text, re.I):
                tup = (f"Uses: {key}", 0.7)
                if tup not in lines:
                    lines.append(tup)
        return lines[:EXTRACT_MAX_BULLETS]

    def _filter_bullets_against_source(self, bullets, source_text):
        src = source_text.lower()
        out = []
        for b in bullets:
            if isinstance(b, tuple):
                t, conf = b
            else:
                t, conf = b, 0.7
            txt = t.strip()
            low = txt.lower()
            if "optimus" in low or low.startswith("say:") or "i'll remember" in low:
                continue
            toks = re.findall(r"[A-Za-z0-9_/.\-:@]{3,}", txt)
            if any(tok.lower() in src for tok in toks):
                out.append((txt, conf))
        return out

    def add_freeform(self, text: str, *, source: str = "user") -> None:
        """
        Save freeform text using AI span selection + grounded extraction.
        - Picks a concise span from `text` (AI returns {text, confidence})
        - Grounds against `text`
        - Stores a raw Note (lower confidence) + structured bullets (tuples carry confidence)
        - Normalizes simple "my X is Y" facts
        """
        if not text or not text.strip():
            return
        raw = text.strip()

        # 1) Raw fallback (low confidence)
        self.add_bullets([("Note: " + raw, 0.65)])

        # 2) AI picks best span (grounded)
        picked = self._select_span_from_text(raw)

        # 3) Extract structured bullets from whichever text we decided to store
        chosen_text = picked[0] if picked else raw
        try:
            notes = self.extract_notes(chosen_text, "(remember request)",
                                       history_window=None)  # already grounded tuples
        except Exception:
            notes = []

        # 4) Store AI-picked span (if any) and structured bullets
        if picked:
            self.add_bullets([picked])  # (text, confidence)
        if notes:
            self.add_bullets(notes)

        # 5) Normalize "my X is Y" into key:value (nice for recall)
        for key, val in re.findall(r"\bmy\s+([a-z0-9 _\-]{1,40})\s+is\s+([^\.,;\n]+)", chosen_text, flags=re.I):
            kv = f"{key.strip().lower()}: {val.strip()}"
            self.add_bullets([(kv, max(0.75, picked[1] if picked else 0.8))])

        # 6) Keep primer policy updated
        self.maybe_rebuild_primer()

    def recall(self, query: str | None = None, top_k: int = 5) -> list[dict]:
        """
        Generic recall. If query is None -> newest notes. Else -> top-k relevant notes.
        Returns a list of dicts: {"ts": "...", "text": "...", "score": float}
        """
        notes = self.state.get("notes", [])
        if not notes:
            return []
        if not query:
            return [{"ts": n.get("ts", ""), "text": n.get("text", ""), "score": 1.0}
                    for n in notes[-top_k:]]

        q = _tokenize(query)
        out = []
        now = datetime.datetime.now()
        for n in notes:
            t = n.get("text", "")
            score = _overlap_score(q, _tokenize(t))
            # mild recency boost
            try:
                ts = datetime.datetime.fromisoformat(n.get("ts", ""))
                age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
                score *= 1.0 / (1.0 + 0.1 * age_days)
            except Exception:
                pass
            out.append({"ts": n.get("ts", ""), "text": t, "score": float(score)})
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top_k]

# --- Optional CLI demo (uses your endpoints)
def main():
    mem = ConversationMemory()
    primer = mem.load_or_build_primer()
    print("Primer:\n" + (primer or "(empty)"))
    hist: List[Dict[str,str]] = [{"role":"system","content":"Memory demo"}]
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user: continue
        if user in ("/q","/quit","exit"): break
        # echo via your chat_once equivalent is not used here to keep deps minimal
        reply = "Acknowledged. (Demo mode — no LLM turn here.)"
        print("\nBOT:", reply, "\n")
        hist.append({"role":"user","content":user}); hist.append({"role":"assistant","content":reply})
        if mem.durable_trigger(user, reply):
            bullets = mem.extract_notes(user, reply, hist)
            if bullets:
                mem.add_bullets(bullets); mem.maybe_rebuild_primer(); mem.append_notes_md(bullets)
                print(f"[memory] captured {len(bullets)} bullet(s).")

if __name__ == "__main__":
    main()
