# voice_assistant/memory_manager.py
# Run (optional demo):  python -m voice_assistant.memory_manager
# Integrates with your existing Ollama client + config.
from __future__ import annotations
import os, json, datetime, pathlib, re
from typing import List, Dict, Any, Iterable, Tuple

from .config import config
from .ollama_client import call_ollama  # reuse your dual-endpoint fallback  

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
    s = s.lower()
    s = re.sub(r"[^a-z0-9_\-./:]", " ", s)
    toks = [t for t in s.split() if t not in {"the","a","an","and","or","but","to","of","for","in","on","with","at","by","is","it","this","that","det","og","at","en","et","til","af","på","i","er","som","der"}]
    return toks

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

EXTRACT_SYS = """Extract durable notes from the latest exchange for persistent memory.

Rules:
- Keep only enduring facts: hardware (e.g., Raspberry Pi 5, HATs), wiring/pins, I2C/SPI addresses, file paths, configs, models/voices, TODOs/decisions.
- Ignore greetings, ephemeral status, or things likely obsolete soon.
- Be specific and concise. If nothing durable, reply ONLY with: NONE

Format:
BULLETS:
- item 1
- item 2
"""

def _ollama_chat(system: str, user: str, model: str | None = None, temperature: float = 0.2) -> str:
    payload = {
        "model": model or config.default_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 2048},
    }
    data = call_ollama(payload)  # your helper tries PC then Pi endpoints  :contentReference[oaicite:3]{index=3}
    return data.get("message", {}).get("content", "") or ""

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

    def add_bullets(self, bullets: List[str], confidence: float = 0.75) -> None:
        notes: List[Dict[str, Any]] = self.state.get("notes", [])
        seen = set(n.get("text","").strip().lower() for n in notes)
        for line in bullets[:EXTRACT_MAX_BULLETS]:
            key = line.strip().lower()
            if key in seen:
                continue
            notes.append({"ts": _now_iso(), "text": line.strip(), "confidence": float(confidence)})
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
        scan = f"{user_text}\n{reply_text}"
        for pat in self._DURABLE_PATS:
            if re.search(pat, scan, re.I): return True
        if re.search(self._REMEMBER_PATS, user_text, re.I): return True
        return False

    def extract_notes(self, user_text: str, reply_text: str, history_window: List[Dict[str,str]] | None = None) -> List[str]:
        ctx: List[str] = []
        if history_window:
            for m in history_window[-6:]:
                role = m.get("role","").upper()[:9]
                txt  = m.get("content","")
                ctx.append(f"{role}: {txt}")
        block = "\n".join(ctx + [f"USER(last): {user_text}", f"ASSISTANT(last): {reply_text}"])
        try:
            out = _ollama_chat(EXTRACT_SYS, block, model=config.default_model, temperature=0.2)
        except Exception:
            return self._heuristic_extract(user_text, reply_text)
        out = (out or "").strip()
        if re.search(r"\bNONE\b", out, re.I):
            return []
        bullets = [re.sub(r"^\s*[-*]\s*","",b).strip()
                   for b in out.splitlines()
                   if b.strip().startswith(("-", "*"))]
        clean = []
        for b in bullets:
            b = re.sub(r"\s+", " ", b)
            if 4 <= len(b) <= 240:
                clean.append(b)
        return clean[:EXTRACT_MAX_BULLETS]

    def _heuristic_extract(self, user_text: str, reply_text: str) -> List[str]:
        lines: List[str] = []
        text = user_text + "\n" + reply_text
        for m in re.findall(r"(/[A-Za-z0-9._\-/]+)", text):
            if len(m) <= 120 and m not in lines:
                lines.append(f"Path: {m}")
        for m in re.findall(r"\b0x[0-9A-Fa-f]+\b", text):
            if m not in lines:
                lines.append(f"I2C/SPI addr: {m}")
        for m in re.findall(r"\bGPIO\d+\b", text):
            if m not in lines:
                lines.append(f"GPIO used: {m}")
        for key in ["TMC2209","SSD1306","Piper","Vosk","Hailo","pan-tilt"]:
            if re.search(fr"\b{key}\b", text, re.I) and key not in [l.split(":")[0] for l in lines]:
                lines.append(f"Uses: {key}")
        return lines[:EXTRACT_MAX_BULLETS]

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
