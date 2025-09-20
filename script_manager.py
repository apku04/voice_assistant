from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class ScriptEntry:
    key: str
    phrases: List[str]
    script: str  # filename within scripts dir
    args: List[str]
    timeout_sec: int


class ScriptManager:
    """Loads a small allowlist of scripts from config.yaml and runs them.

    - Only executes files inside the scripts directory
    - Matches user text via simple phrase substring matching (case-insensitive)
    - Returns stdout (or stderr) to the caller for display/TTS
    """

    def __init__(self, scripts_dir: Optional[Path] = None):
        if scripts_dir is None:
            # Default to scripts folder co-located under voice_assistant
            scripts_dir = Path(__file__).resolve().parent / "scripts"
        self.scripts_dir = Path(scripts_dir)
        self.config_path = self.scripts_dir / "config.yaml"
        self._entries: Dict[str, ScriptEntry] = {}
        self.reload()

    # ---- config ----
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {"scripts": []}
        data: Dict[str, Any]
        txt = self.config_path.read_text(encoding="utf-8")
        # Try YAML first, fallback to JSON (YAML superset)
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(txt) or {}
        except Exception:
            data = json.loads(txt)
        if not isinstance(data, dict):
            return {"scripts": []}
        return data

    def reload(self) -> None:
        data = self._load_config()
        self._entries.clear()
        for item in data.get("scripts", []) or []:
            try:
                ent = ScriptEntry(
                    key=str(item["key"]).strip(),
                    phrases=[str(p).strip() for p in (item.get("phrases") or [])],
                    script=str(item["script"]).strip(),
                    args=[str(a) for a in (item.get("args") or [])],
                    timeout_sec=int(item.get("timeout_sec") or 8),
                )
                # Basic validation: file must live inside scripts_dir
                script_path = (self.scripts_dir / ent.script).resolve()
                if not str(script_path).startswith(str(self.scripts_dir.resolve())):
                    continue
                self._entries[ent.key] = ent
            except Exception:
                continue

    # ---- query ----
    def list_entries(self) -> List[ScriptEntry]:
        return list(self._entries.values())

    def match(self, text: str) -> Optional[ScriptEntry]:
        s = self._normalize(text)
        for ent in self._entries.values():
            for phrase in ent.phrases:
                p = self._normalize(phrase)
                if p and p in s:
                    return ent
        return None

    def _normalize(self, text: str) -> str:
        """Lowercase, remove punctuation, common articles, and collapse spaces."""
        if not text:
            return ""
        t = text.lower()
        # remove punctuation
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        # remove common articles
        t = re.sub(r"\b(the|a|an)\b", " ", t)
        # collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    # ---- run ----
    def run(self, key: str) -> Tuple[bool, str]:
        ent = self._entries.get(key)
        if not ent:
            return False, f"No such script: {key}"
        return self._run_entry(ent)

    def _run_entry(self, ent: ScriptEntry) -> Tuple[bool, str]:
        script_path = (self.scripts_dir / ent.script).resolve()
        if not str(script_path).startswith(str(self.scripts_dir.resolve())):
            return False, "Access denied"
        if not script_path.exists():
            return False, f"Script not found: {script_path.name}"

        cmd = [sys.executable, "-u", str(script_path), *ent.args]
        try:
            out = subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=ent.timeout_sec,
            )
            msg = (out or "").strip()
            return True, (msg if msg else "(ok)")
        except subprocess.CalledProcessError as e:
            return False, (e.output or str(e))
        except subprocess.TimeoutExpired:
            return False, f"Timed out after {ent.timeout_sec}s"
