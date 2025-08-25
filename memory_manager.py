# voice_assistant/memory_manager.py
from __future__ import annotations
import os, json, datetime, pathlib, re
from typing import List, Dict, Any, Iterable, Tuple

from .config import config
from .ollama_client import _ollama_chat  # Use the enhanced version

HOME = pathlib.Path.home()
MEM_DIR = pathlib.Path(os.environ.get("MEMORY_DIR", str(HOME / ".optimus")))
MEM_JSON = pathlib.Path(os.environ.get("MEMORY_FILE", str(MEM_DIR / "memory.json")))
NOTES_MD = pathlib.Path(os.environ.get("NOTES_FILE", str(MEM_DIR / "notes.md")))
MEM_DIR.mkdir(parents=True, exist_ok=True)

MAX_NOTES = 400
PRIMER_TARGET_CHARS = 1600
PRIMER_MAX_AGE_DAYS = 30
PRIMER_REBUILD_SECS = 6 * 3600
RELEVANT_TOP_K = 6
EXTRACT_MAX_BULLETS = 5


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _tokenize(s: str) -> List[str]:
    s = s.casefold()
    toks = re.findall(r"[\w\-./:@]+", s, flags=re.UNICODE)
    stop = {
        "the", "a", "an", "and", "or", "but", "to", "of", "for", "in", "on", "with", "at", "by", "is", "it", "this",
        "that",
        "det", "og", "at", "en", "et", "til", "af", "på", "i", "er", "som", "der"
    }
    return [t for t in toks if t not in stop]


def _overlap_score(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb: return 0.0
    inter = len(sa & sb)
    return inter / (len(sa) ** 0.5 * len(sb) ** 0.5)


# FIXED: More specific primer system prompt
PRIMER_SYS = """You are a terse system summarizer.
Input: bullet notes captured across sessions for the same user and project.
Task:
- Produce a compact "Context Primer" (<= 120 words; bullets allowed).
- Keep ONLY enduring facts that were explicitly stated: names, titles, preferences, configurations.
- Omit ALL hardware details, technical specifications, I2C/SPI addresses, file paths unless explicitly mentioned.
- Omit chit-chat, stale dates, and private info not needed for function.
Output ONLY the primer text based strictly on the input notes.
"""

# FIXED: More grounded extraction
EXTRACT_SYS = """Extract ONLY durable facts that were explicitly stated in the USER text.

RULES:
- Extract ONLY from the USER text, NOT from the assistant reply.
- If the fact is NOT explicitly in the USER text, DO NOT include it.
- Do NOT generalize or infer missing details.
- Do NOT include transient telemetry (uptime, CPU%, load, memory%, disk%).
- If there are no explicit durable facts, reply EXACTLY: NONE

Output format, strictly:
BULLETS:
- <fact text> | confidence=<0.0–1.0>
"""

# FIXED: Better span selection
SELECT_SYS = """You select durable USER facts to store from the provided text.

RULES:
- Choose ONLY content explicitly present in INPUT_TEXT.
- Prefer personal details: names, titles, preferences, important information.
- Do NOT invent or generalize. If nothing durable, reply EXACTLY: NONE.

Output JSON:
{"text":"<verbatim or minimally cleaned user fact>","confidence":0..1}
"""


class ConversationMemory:
    _DURABLE_PATS = [
        r"\bmy\s+name\s+is\b", r"\bmy\s+title\s+is\b", r"\bi\s+am\b", r"\bi'm\b",
        r"\bremember\b", r"\bsave\b", r"\bnote\b", r"\bstore\b",
        r"\bprefer\b", r"\blike\b", r"\bdislike\b", r"\bfavorite\b"
    ]
    _REMEMBER_PATS = r"\b(remember|note|save this|persist|store)\b"

    def __init__(self,
                 memory_file: str | os.PathLike = MEM_JSON,
                 notes_file: str | os.PathLike = NOTES_MD,
                 max_notes: int = MAX_NOTES,
                 primer_target_chars: int = PRIMER_TARGET_CHARS,
                 primer_max_age_days: int = PRIMER_MAX_AGE_DAYS,
                 primer_rebuild_secs: int = PRIMER_REBUILD_SECS):
        self.memory_path = pathlib.Path(memory_file)
        self.notes_path = pathlib.Path(notes_file)
        self.max_notes = max_notes
        self.primer_target_chars = primer_target_chars
        self.primer_max_age_days = primer_max_age_days
        self.primer_rebuild_secs = primer_rebuild_secs
        self.state: Dict[str, Any] = self._load_state()

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
        try:
            with open(self.notes_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception:
            pass

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

    def load_or_build_primer(self) -> str:
        if self.state.get("primer"):
            return self.state["primer"]
        return self.maybe_rebuild_primer(force=True)

    def maybe_rebuild_primer(self, force: bool = False) -> str:
        if force:
            return self._rebuild_primer()
        last_s = self.state.get("primer_ts", "")
        try:
            last = datetime.datetime.fromisoformat(last_s) if last_s else None
        except Exception:
            last = None
        need = not self.state.get("primer") or last is None
        if not need and (datetime.datetime.now() - last).total_seconds() > self.primer_rebuild_secs:
            need = True
        if self.state.get("notes") and len(self.state["notes"]) % 25 == 0:
            need = True
        return self._rebuild_primer() if need else self.state.get("primer", "")

    def _rebuild_primer(self) -> str:
        notes = self.state.get("notes", [])
        cutoff = datetime.datetime.now() - datetime.timedelta(days=self.primer_max_age_days)
        keep: List[str] = []
        for n in notes[-self.max_notes:]:
            try:
                ts = datetime.datetime.fromisoformat(n.get("ts", ""))
            except Exception:
                ts = datetime.datetime.now()
            if ts < cutoff:
                continue
            if n.get("confidence", 0.0) < 0.6:
                continue
            txt = n.get("text", "").strip()
            if txt:
                keep.append(f"- {txt}")
        if not keep:
            self.state["primer"] = ""
            self.state["primer_ts"] = _now_iso()
            self._save_state()
            return ""
        joined = "\n".join(keep)
        try:
            out = _ollama_chat(PRIMER_SYS, joined, model=config.default_model, temperature=0.1)
        except Exception:
            out = joined[:self.primer_target_chars]
        primer = (out or "").strip()[:self.primer_target_chars]
        self.state["primer"] = primer
        self.state["primer_ts"] = _now_iso()
        self._save_state()
        return primer

    def get_memory_context(self, user_input: str) -> str:
        notes = self.state.get("notes", [])
        if not notes: return ""
        q = _tokenize(user_input)
        scored: List[Tuple[float, str]] = []
        for n in notes:
            t = n.get("text", "")
            score = _overlap_score(q, _tokenize(t)) * (0.5 + 0.5 * n.get("confidence", 0.5))
            scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [t for s, t in scored[:RELEVANT_TOP_K] if s > 0.0]
        return "Relevant notes:\n" + "\n".join(f"- {t}" for t in top) if top else ""

    def durable_trigger(self, user_text: str, reply_text: str) -> bool:
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
        # FIXED: Only extract from user text, not reply
        block = f"USER(last): {user_text}"

        try:
            out = _ollama_chat(
                EXTRACT_SYS, block, model=config.default_model,
                temperature=0.1, deterministic=True
            )
        except Exception:
            return self._heuristic_extract(user_text, "")

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
                conf = 0.7
            if 4 <= len(txt) <= 240:
                notes.append((txt, conf))

        # Filter out assistant references
        BAD_TERMS = {"optimus", "say:", "i'll remember", "i will remember", "i'm optimus", "i am optimus"}
        filtered = [(t, c) for (t, c) in notes if all(bt not in t.lower() for bt in BAD_TERMS)]

        # Ground against user text
        source = user_text or ""
        grounded = self._filter_bullets_against_source(filtered[:EXTRACT_MAX_BULLETS], source_text=source)
        return grounded

    def _heuristic_extract(self, user_text: str, reply_text: str) -> list[tuple[str, float]]:
        lines: List[tuple[str, float]] = []

        # Extract personal information
        if name_match := re.search(r"\bmy\s+name\s+is\s+([^.!,;?]+)", user_text, re.I):
            lines.append((f"Name: {name_match.group(1).strip()}", 0.9))
        if title_match := re.search(r"\bmy\s+title\s+is\s+([^.!,;?]+)", user_text, re.I):
            lines.append((f"Title: {title_match.group(1).strip()}", 0.9))
        if name_match := re.search(r"\bi\s+(?:am|'m)\s+([^.!,;?]+)", user_text, re.I):
            lines.append((f"Name: {name_match.group(1).strip()}", 0.8))

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

    def _select_span_from_text(self, text: str) -> tuple[str, float] | None:
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
            grounded = self._filter_bullets_against_source([(span, conf)], source_text=text)
            return grounded[0] if grounded else None
        except Exception:
            return None

    def add_freeform(self, text: str, *, source: str = "user") -> None:
        if not text or not text.strip():
            return
        raw = text.strip()

        self.add_bullets([("Note: " + raw, 0.65)])

        picked = self._select_span_from_text(raw)

        chosen_text = picked[0] if picked else raw
        try:
            notes = self.extract_notes(chosen_text, "(remember request)", history_window=None)
        except Exception:
            notes = []

        if picked:
            self.add_bullets([picked])
        if notes:
            self.add_bullets(notes)

        for key, val in re.findall(r"\bmy\s+([a-z0-9 _\-]{1,40})\s+is\s+([^\.,;\n]+)", chosen_text, flags=re.I):
            kv = f"{key.strip().lower()}: {val.strip()}"
            self.add_bullets([(kv, max(0.75, picked[1] if picked else 0.8))])

        self.maybe_rebuild_primer()

    def recall(self, query: str | None = None, top_k: int = 5) -> list[dict]:
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
            try:
                ts = datetime.datetime.fromisoformat(n.get("ts", ""))
                age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
                score *= 1.0 / (1.0 + 0.1 * age_days)
            except Exception:
                pass
            out.append({"ts": n.get("ts", ""), "text": t, "score": float(score)})
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top_k]