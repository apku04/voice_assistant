# voice_assistant/diagnostics.py
import re
import shlex
import logging
from typing import Dict, Optional, Any

# Set up logging
logger = logging.getLogger(__name__)

CODE_BLOCK_RE = re.compile(r"```.*?```", flags=re.S)
# More specific regex patterns to avoid false positives
DIAG_SUGG_RE = re.compile(r"^(?:run|execute|start|initiate)\s+(/diag[^\n`]*?)\b", re.I | re.M)
BARE_DIAG_RE = re.compile(r"^(/diag(?:\s+[^\s`]+)*(?:\s+-v)?)", re.I | re.M)

def parse_diag_suggestion(text: str) -> Optional[Dict[str, Any]]:
    """Parse diagnostic suggestions from LLM response"""
    if not text:
        return None
        
    t = CODE_BLOCK_RE.sub(" ", text)
    m = DIAG_SUGG_RE.search(t) or BARE_DIAG_RE.search(t)
    if not m:
        return None
        
    raw = m.group(1).strip().strip("`")
    try:
        toks = shlex.split(raw)
    except Exception:
        toks = raw.split()
        
    flt, verbose = None, False
    for tok in toks[1:]:
        if tok == "-v":
            verbose = True
        elif not tok.startswith("-"):
            flt = tok
            
    return {"cmd": "diag", "filter": flt, "verbose": verbose}

def maybe_offer_diag(reply: str, allow: bool) -> Optional[Dict[str, Any]]:
    """Check if a diagnostic should be offered based on reply"""
    return parse_diag_suggestion(reply) if allow else None

# Update the system prompt to encourage memory usage
def make_system_prompt(lang: str) -> str:
    """Create system prompt based on language"""
    l = (lang or "en").lower()
    base_prompt = {
        "da": ("Du er en klassisk Hollywood-robot (Warhammer/StarCraft stil). "
               "KUN dansk. Svar ultrakort, mekanisk, autoritativt (1–2 sætninger). "
               "Ingen smalltalk. Ingen tal uden data. Hvis diagnostik uden data: svar 'Ukendt.' og instruér: 'Kør /diag'. "),
        "en": ("You are an advanced robot."
               "Your name is Optimus-Zeta-Prime-5421."
               "If asked for your name, answer exactly 'Optimus'. "
               "Do not invent numbers; if unknown, say 'Unknown'. "
               "Be concise (<= 4 short lines). Focus on health, top errors, concrete actions. "
               "Optionally start with a 'Say:' line for TTS.")
    }
    
    memory_instruction = {
        "da": "Du kan vælge at huske vigtig information som hardware, konfigurationer, eller brugerpræferencer. Brug udtryk som 'Jeg husker...' eller 'Noteret: ...' når du gemmer vigtig information.",
        "en": "You may choose to remember important information like hardware details, configurations, or user preferences. Use phrases like 'I'll remember that...' or 'Noted: ...' when storing important information."
    }
    
    return f"{base_prompt.get(l, base_prompt['en'])} {memory_instruction.get(l, memory_instruction['en'])}"

def diag_system_prompt(lang: str) -> str:
    """Create diagnostic-specific system prompt"""
    return make_system_prompt(lang)