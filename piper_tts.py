# voice_assistant/piper_tts.py
import os
import re
import shlex
import subprocess
import tempfile
import wave
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable

# Set up logging
logger = logging.getLogger(__name__)

# ---------- Simple cleaner ----------
BOT_PREFIX_RE   = re.compile(r"^\s*BOT\s*\([^)]+\)\s*:\s*", re.I)
CODE_BLOCK_RE   = re.compile(r"```.*?```", flags=re.S)
MARKUP_RE       = re.compile(r"(\*\*?|__|~~|`+)")
URL_RE          = re.compile(r"https?://\S+")
SENT_SPLIT_RE   = re.compile(r"(?<=[.!?])\s+")
BULLET_RE       = re.compile(r"^\s*(?:[-*•]|\d+[\).\]])\s+(.*)")
ITEM_RE         = re.compile(r"^\s*\d+[\).\]]\s+(.*)")
ADVISORY_RE     = re.compile(r"\b(please\s+note|note:|recommendations\s+are\s+based|further\s+investigation|may\s+not\s+be\s+comprehensive)\b", re.I)

def _clean(s: str) -> str:
    s = CODE_BLOCK_RE.sub(" ", s)
    s = URL_RE.sub("", s)
    s = MARKUP_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" .,:;")
    if s and s[-1] not in ".!?":
        s += "."
    return s

def _find_say(text: str) -> Optional[str]:
    for ln in text.splitlines():
        if ln.strip().lower().startswith("say:"):
            return _clean(ln.split(":", 1)[1].strip())
    return None

def _extract_bullets(text: str, max_items: int) -> List[str]:
    lines = text.splitlines()
    items: List[str] = []
    i = 0
    while i < len(lines) and len(items) < max_items:
        ln = lines[i]
        m = BULLET_RE.match(ln) or ITEM_RE.match(ln)
        if m:
            issue = _clean(m.group(1).strip())
            action = None
            # look ahead a couple of lines for "Concrete Action:"
            j = i + 1
            look_end = min(len(lines), i + 4)
            while j < look_end:
                s = lines[j].strip()
                if s.lower().startswith("concrete action"):
                    action = _clean(s.split(":", 1)[1].strip()) if ":" in s else _clean(s)
                    break
                # stop if we hit another bullet
                if BULLET_RE.match(lines[j]) or ITEM_RE.match(lines[j]):
                    break
                j += 1
            items.append(f"{issue} {'Action: ' + action if action else ''}".strip())
        i += 1
    return items

def _find_advisory(text: str) -> Optional[str]:
    # try explicit "Note:" lines first
    for ln in text.splitlines():
        if ADVISORY_RE.search(ln):
            return _clean(ln)
    # fallback: last sentence that looks advisory
    sents = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    for s in reversed(sents[-3:]):
        if ADVISORY_RE.search(s):
            return _clean(s)
    return None

# ---------- File-first Piper with tiny FX ----------
def _apply_fx(in_wav: str, out_wav: str, preset: str,
              *, gain_db: float = -8.0, pitch_cents: int = -120, bass_db: float = 2.0,
              peak_dbfs: Optional[float] = None) -> None:
    chain: List[str] = []
    if pitch_cents:
        chain += ["pitch", str(pitch_cents)]
    if preset == "metallic":
        chain += ["channels","1","highpass","180","lowpass","7000","tremolo","70","0.6",
                  "overdrive","8","6","equalizer","1500","2.0q","6","equalizer","4000","2.0q","5",
                  "compand","0.1,0.2","-60,-60,-20,-8,0,-6","0","-6","0.2","reverb","15"]
        if bass_db: chain += ["bass", f"{bass_db}"]
    elif preset == "radio":
        chain += ["channels","1","highpass","300","lowpass","3400",
                  "compand","0.02,0.20","6:-70,-60,-20,-10,-5,-5","6","-8","0.2"]
    elif preset == "mono":
        chain += ["channels","1"]
        if bass_db: chain += ["bass", f"{bass_db}"]
    if peak_dbfs is not None:
        chain += ["gain","-n", f"{peak_dbfs}"]
    elif gain_db:
        chain += ["gain", f"{gain_db}"]

    if chain:
        subprocess.run(["sox","-q", in_wav, out_wav] + chain,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        if in_wav != out_wav:
            subprocess.run(["sox","-q", in_wav, out_wav],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _speak_file_first(text: str, pre_silence_ms: int = 200) -> None:
    from .config import config
    
    model = config.piper_model
    if not model or not text:
        return
        
    alsa = config.alsa_dev
    preset = config.robot_preset.lower()
    
    # tunables via env (optional)
    pitch = int(os.environ.get("ROBOT_PITCH_CENTS", "-120") or "-120")
    bass = float(os.environ.get("ROBOT_BASS_DB", "2.0") or "2.0")
    gain = float(os.environ.get("ROBOT_GAIN_DB", "-8.0") or "-8.0")
    peakv = os.environ.get("ROBOT_PEAK_DBFS")
    peak = float(peakv) if peakv else None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        wav_path = wf.name
        
    try:
        # synth
        cmd = ["piper", "-m", model, "-f", wav_path]
        if config.piper_len:    cmd += ["--length_scale", config.piper_len]
        if config.piper_noise:  cmd += ["--noise_scale", config.piper_noise]
        if config.piper_noisew: cmd += ["--noise_w", config.piper_noisew]
        
        subprocess.run(cmd, input=text.encode("utf-8"), check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # pre-roll
        if pre_silence_ms > 0:
            tmp = wav_path + ".pad.wav"
            with wave.open(wav_path, "rb") as r:
                params = r.getparams()
                frames = r.readframes(r.getnframes())
                
            pre_frames = int(params.framerate * pre_silence_ms / 1000.0)
            silence = b"\x00" * pre_frames * params.nchannels * params.sampwidth
            with wave.open(tmp, "wb") as w:
                w.setparams(params)
                w.writeframes(silence + frames)
            os.replace(tmp, wav_path)
            
        # FX
        fx = wav_path + ".fx.wav"
        _apply_fx(wav_path, fx, preset, gain_db=gain, pitch_cents=pitch, bass_db=bass, peak_dbfs=peak)
        play = fx if os.path.exists(fx) else wav_path
        
        # play - wait for completion to avoid race conditions
        result = subprocess.run(["aplay", "-q", "-D", alsa, play], 
                               check=False, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"aplay failed with return code {result.returncode}: {result.stderr}")
            
    finally:
        # Small delay to ensure playback is complete before cleanup
        time.sleep(0.1)
        for p in (wav_path, wav_path + ".fx.wav", wav_path + ".pad.wav"):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                logger.warning(f"Failed to clean up file {p}: {e}")

# ---------- Tiny TTS wrapper ----------
class PiperTTS:
    def __init__(self,
                 voice_cmd: Optional[str] = None,
                 *,
                 voice_fn: Optional[Callable[[str], None]] = None,
                 led_manager: Any | None = None,
                 max_chunks: int = None,
                 max_chars: int = None,
                 pause_s: float = None,
                 prefix_zwsp: bool = True,
                 ):
        from .config import config
        
        self.voice_cmd = voice_cmd
        self.voice_fn = voice_fn
        self.led_manager = led_manager
        self.max_chunks = max_chunks or config.max_chunks
        self.max_chars = max_chars or config.max_chars
        self.pause_s = pause_s or config.pause_s
        self.prefix = "\u200B\u200B" if prefix_zwsp else ""

    # public
    def speak(self, text: str) -> None:
        chunks = self._extract_chunks(text)
        if not chunks:
            return
            
        if self.led_manager:
            self.led_manager.set_color("speak")
            
        try:
            for i, ch in enumerate(chunks, 1):
                payload = (self.prefix + ch) if self.prefix else ch
                if self.voice_fn:
                    self.voice_fn(payload)
                elif self.voice_cmd:
                    subprocess.run(shlex.split(self.voice_cmd) + [payload], check=False)
                else:
                    _speak_file_first(payload)
                    
                if i < len(chunks):
                    time.sleep(self.pause_s)
        finally:
            if self.led_manager:
                self.led_manager.set_color("idle")

    # internals
    def speak_chunks(self, chunks: List[str]):
        """Backward-compat wrapper: speak a list of short lines."""
        if not chunks:
            return
            
        # use the same engine path as speak(), but keep LED/pause once per list
        try:
            if self.led_manager:
                self.led_manager.set_color("speak")
                
            for i, ch in enumerate(chunks, 1):
                payload = (self.prefix + str(ch)) if self.prefix else str(ch)
                if self.voice_fn:
                    self.voice_fn(payload)
                elif self.voice_cmd:
                    subprocess.run(shlex.split(self.voice_cmd) + [payload], check=False)
                else:
                    _speak_file_first(payload)
                    
                if i < len(chunks):
                    time.sleep(self.pause_s)
        finally:
            if self.led_manager:
                self.led_manager.set_color("idle")

    def _extract_chunks(self, text: str) -> List[str]:
        if not text:
            return []
            
        text = BOT_PREFIX_RE.sub("", text)
        text = text.strip()

        chunks: List[str] = []

        # 1) Say: line (if any)
        say = _find_say(text)
        if say:
            chunks.append(say[:self.max_chars])

        # 2) First bullets (join with action if present)
        bullets = _extract_bullets(text, max_items=self.max_chunks - len(chunks))
        for b in bullets:
            chunks.append(b[:self.max_chars])
            if len(chunks) >= self.max_chunks:
                break

        # 3) If no bullets, just first sentences
        if len(chunks) == 0:
            cleaned = _clean(text)
            sents = [s.strip() for s in SENT_SPLIT_RE.split(cleaned) if s.strip()]
            for s in sents[: self.max_chunks]:
                chunks.append(s[:self.max_chars])

        # 4) Advisory tail if room
        if len(chunks) < self.max_chunks:
            adv = _find_advisory(text)
            if adv:
                chunks.append(adv[:self.max_chars])

        # hard cap
        return chunks[: self.max_chunks]