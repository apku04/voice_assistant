from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class BehaviorSnapshot:
    """Lightweight container for the latest perception readings."""

    mood: Optional[Tuple[str, float]]
    left_hand: Optional[Tuple[str, float]]
    right_hand: Optional[Tuple[str, float]]
    raw: Dict[str, Optional[Tuple[str, float]]]

    def has_data(self) -> bool:
        return bool(self.mood or self.left_hand or self.right_hand)


class BehaviorMonitor:
    """Helper that bridges face_follow output to natural language summaries."""

    _MOOD_TERMS = {"mood", "face", "expression", "smile", "frown", "emotion", "facial"}
    _HAND_TERMS = {"hand", "gesture", "fist", "wave", "finger", "thumb", "sign"}

    def __init__(self, face_follow):
        self.face_follow = face_follow

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
        if include_mood and snap.mood:
            entries.append(self.describe_mood(snap.mood))
        if include_hands:
            if snap.left_hand:
                entries.append(self.describe_hand("left", snap.left_hand))
            if snap.right_hand:
                entries.append(self.describe_hand("right", snap.right_hand))
        return "\n".join(entries) if entries else ""

    def answer_query(
        self,
        user_text: str,
        snapshot: Optional[BehaviorSnapshot] = None,
    ) -> Optional[str]:
        wants_mood, wants_hands = self.requirements(user_text)
        if not (wants_mood or wants_hands):
            return None
        snap = snapshot or self.snapshot()
        parts = []
        if wants_mood:
            if snap.mood:
                parts.append(self.describe_mood(snap.mood))
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

    def describe_mood(self, mood: Optional[Tuple[str, float]]) -> str:
        if not mood:
            return "Mood is unknown."
        label, conf = mood
        return f"Observed mood: {label.lower()} (confidence {conf:.2f})."

    def describe_hand(self, side: str, data: Tuple[str, float]) -> str:
        label, conf = data
        return f"Your {side} hand looks {label.lower()} ({conf:.2f})."

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
