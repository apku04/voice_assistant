"""LLM-assisted mood interpretation utilities."""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..config import config
from ..ollama_client import chat_once


logger = logging.getLogger(__name__)


def _extract_json_object(text: str) -> Optional[str]:
    """Return the first balanced JSON object substring in text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


@dataclass(frozen=True)
class MoodFeatures:
    """Normalized facial metrics that describe the current mood."""

    mouth_ratio: float
    mouth_curve: float
    mouth_width: float
    mouth_height: float
    face_visible: bool = True

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MoodFeatures":
        return cls(
            mouth_ratio=float(payload.get("mouth_ratio", 0.0)),
            mouth_curve=float(payload.get("mouth_curve", 0.0)),
            mouth_width=float(payload.get("mouth_width", 0.0)),
            mouth_height=float(payload.get("mouth_height", 0.0)),
            face_visible=bool(payload.get("face_visible", True)),
        )

    def prompt(self) -> str:
        return (
            "Facial landmark measurements:\n"
            f"- mouth opening to width ratio: {self.mouth_ratio:.4f}\n"
            f"- mouth corner curvature score: {self.mouth_curve:.4f}\n"
            f"- mouth width (pixels): {self.mouth_width:.1f}\n"
            f"- mouth height (pixels): {self.mouth_height:.1f}\n"
            f"- face visible: {str(self.face_visible).lower()}"
        )


class LLMMoodInterpreter:
    """Ask the configured LLM to describe the user's mood from features."""

    def __init__(self, *, model: Optional[str] = None, cooldown: Optional[float] = None):
        self.model = model or config.perception_llm_model
        self.cooldown = cooldown if cooldown is not None else config.perception_llm_cooldown
        self._last: Tuple[Optional[MoodFeatures], Optional[str], float] = (None, None, 0.0)

    def interpret(
        self,
        features: MoodFeatures,
        *,
        fallback: Optional[Tuple[str, float]] = None,
    ) -> Optional[str]:
        if not config.perception_llm_enabled:
            logger.info("LLM mood disabled via config")
            return None

        last_features, last_reply, last_ts = self._last
        now = time.time()
        if last_features == features and (now - last_ts) < self.cooldown:
            logger.info("LLM mood using cached reply: %s", last_reply)
            return last_reply

        system_prompt = (
            "You are an expression analyst. Map raw facial metrics to a single emotion "
            "label from {happy, neutral, sad, surprised, uncertain}. Respond with strict JSON "
            "matching {\"label\": str, \"confidence\": number 0-1}. Do not include any text "
            "outside of the JSON object."
        )
        user_text = features.prompt()

        try:
            result = chat_once(
                history=[],
                sys_prompt=system_prompt,
                user_text=user_text,
                current_model=self.model,
                deterministic=False,
            )
            reply = result.get("reply", "").strip()
            logger.info("LLM mood raw reply: %s", reply)
        except Exception as exc:
            logger.warning("LLM mood request failed: %s", exc)
            reply = ""

        parsed: Optional[Dict[str, Any]] = None
        if reply:
            json_blob = _extract_json_object(reply)
            if json_blob:
                try:
                    parsed = json.loads(json_blob)
                except json.JSONDecodeError:
                    logger.info("LLM mood JSON decode failed for blob=%s", json_blob)
                    parsed = None
            else:
                logger.info("LLM mood reply missing JSON object: %s", reply)

        if not parsed:
            if fallback:
                label, _ = fallback
                logger.info("LLM mood fallback to deterministic label=%s", label)
                return f"You seem {label.lower()}."
            logger.info("LLM mood returning None (no parsed response, no fallback)")
            return None

        label = str(parsed.get("label", fallback[0] if fallback else "neutral")).strip() or "neutral"
        try:
            confidence = float(parsed.get("confidence", fallback[1] if fallback else 0.5))
        except (TypeError, ValueError):
            confidence = fallback[1] if fallback else 0.5

        if fallback:
            fb_label, fb_conf = fallback
            if fb_label and fb_label.lower() != label.lower():
                prefer_fallback = fb_conf >= confidence or fb_conf >= 0.6
                if prefer_fallback:
                    logger.info(
                        "LLM mood aligning with deterministic label='%s' (LLM label was '%s')",
                        fb_label,
                        label,
                    )
                    label = fb_label
                    confidence = max(confidence, fb_conf)

        sentence = f"You seem {label.lower()}."

        self._last = (features, sentence, now)
        logger.info("LLM mood interpreted: %s (confidence %.2f)", sentence, confidence)
        return sentence


__all__ = ["LLMMoodInterpreter", "MoodFeatures"]
