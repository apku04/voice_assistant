from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from .mood_llm import LLMMoodInterpreter, MoodFeatures


@dataclass
class BehaviorSnapshot:
    """Lightweight container for the latest perception readings."""

    mood: Optional[Tuple[str, float]]
    mood_features: Optional[Dict[str, Any]]
    left_hand: Optional[Tuple[str, float]]
    right_hand: Optional[Tuple[str, float]]
    raw: Dict[str, Any]

    def has_data(self) -> bool:
        return bool(self.mood or self.left_hand or self.right_hand)


class BehaviorMonitor:
    """Helper that bridges face_follow output to natural language summaries."""

    _MOOD_TERMS = {"mood", "face", "expression", "smile", "frown", "emotion", "facial"}
    _HAND_TERMS = {"hand", "gesture", "fist", "wave", "finger", "thumb", "sign"}

    def __init__(
        self,
        face_follow,
        mood_interpreter: Optional[LLMMoodInterpreter] = None,
        *,
        llm_passthrough: bool = False,
    ):
        self.face_follow = face_follow
        self.mood_interpreter = mood_interpreter
        self._llm_passthrough = bool(llm_passthrough)

    # ---- public API -----------------------------------------------------
    def snapshot(self) -> BehaviorSnapshot:
        raw = {}
        if not self.face_follow:
            return BehaviorSnapshot(None, None, None, raw)
        try:
            raw = self.face_follow.get_behavior_snapshot() or {}
        except Exception:
            raw = {}
        return BehaviorSnapshot(
            mood=self._norm_pair(raw.get("mood")),
            mood_features=self._norm_features(raw.get("mood_features")),
            left_hand=self._norm_pair(raw.get("left_hand")),
            right_hand=self._norm_pair(raw.get("right_hand")),
            raw=raw,
        )

    def requirements(self, user_text: str) -> Tuple[bool, bool]:
        if not user_text:
            return False, False
        txt = user_text.lower()
        wants_mood = any(term in txt for term in self._MOOD_TERMS)
        wants_hands = any(term in txt for term in self._HAND_TERMS)
        return wants_mood, wants_hands

    def build_context(
        self,
        snapshot: Optional[BehaviorSnapshot] = None,
        *,
        include_mood: bool = True,
        include_hands: bool = True,
    ) -> str:
        snap = snapshot or self.snapshot()
        entries = []
        if include_mood:
            if snap.mood or snap.mood_features:
                entries.append(self.describe_mood(snap.mood, snap.mood_features))
            else:
                entries.append("I can't see your face clearly.")
        if include_hands:
            if snap.left_hand:
                entries.append(self.describe_hand("left", snap.left_hand))
            else:
                entries.append("I can't see your left hand clearly.")
            if snap.right_hand:
                entries.append(self.describe_hand("right", snap.right_hand))
            else:
                entries.append("I can't see your right hand clearly.")
        return "\n".join(entries) if entries else ""

    def answer_query(
        self,
        user_text: str,
        snapshot: Optional[BehaviorSnapshot] = None,
    ) -> Optional[str]:
        wants_mood, wants_hands = self.requirements(user_text)
        if not (wants_mood or wants_hands):
            return None
        if self._llm_passthrough:
            return None
        snap = snapshot or self.snapshot()
        parts = []
        if wants_mood:
            if snap.mood or snap.mood_features:
                parts.append(self.describe_mood(snap.mood, snap.mood_features))
            else:
                parts.append("I don't have a clear view of your face right now.")
        if wants_hands:
            if snap.left_hand or snap.right_hand:
                if snap.left_hand:
                    parts.append(self.describe_hand("left", snap.left_hand))
                else:
                    parts.append("I can't see your left hand clearly.")
                if snap.right_hand:
                    parts.append(self.describe_hand("right", snap.right_hand))
                else:
                    parts.append("I can't see your right hand clearly.")
            else:
                parts.append("I can't see your hands clearly right now.")
        return " ".join(parts).strip()

    def describe_mood(
        self,
        mood: Optional[Tuple[str, float]],
        features: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self.mood_interpreter and features:
            try:
                mf = MoodFeatures.from_dict(features)
            except (TypeError, ValueError):
                mf = None
            if mf:
                llm_reply = self.mood_interpreter.interpret(mf, fallback=mood)
                if llm_reply:
                    return llm_reply
        if not mood:
            return "Mood is unknown."
        label, _ = mood
        return f"You seem {label.lower()}."

    def describe_hand(self, side: str, data: Tuple[str, float]) -> str:
        label, _ = data
        return f"Your {side} hand is {label.lower()}."

    # ---- helpers -------------------------------------------------------
    def _norm_pair(self, value: Optional[Tuple[str, float]]):
        if not value:
            return None
        try:
            label, conf = value
            if label is None:
                return None
            return str(label), float(conf)
        except Exception:
            return None

    def _norm_features(self, value: Optional[Mapping[str, Any]]):
        if not value:
            return None
        try:
            out: Dict[str, Any] = {}
            for key, item in value.items():
                if item is None:
                    continue
                if str(key) == "face_visible":
                    out[str(key)] = bool(item)
                else:
                    out[str(key)] = float(item)
            return out or None
        except Exception:
            return None
