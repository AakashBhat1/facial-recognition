"""Adaptive user prompts based on confidence and quality diagnostics.

Maps rejection reasons and confidence scores to actionable user messages.
"""

from __future__ import annotations

from dataclasses import dataclass

from services.quality_filter import QualityResult


@dataclass(frozen=True)
class PromptSuggestion:
    """Immutable prompt to show the user."""

    message: str
    category: str  # "lighting", "distance", "stability", "pose", "general"


# Priority-ordered mapping from rejection reason to prompt.
_REASON_PROMPTS: list[tuple[str, PromptSuggestion]] = [
    ("brightness", PromptSuggestion("Please improve the lighting", "lighting")),
    ("face_size", PromptSuggestion("Please move closer to the camera", "distance")),
    ("blur", PromptSuggestion("Please hold still", "stability")),
    ("yaw", PromptSuggestion("Please face the camera directly", "pose")),
    ("pitch", PromptSuggestion("Please face the camera directly", "pose")),
]


def generate_prompt(
    confidence: float,
    quality_result: QualityResult | None = None,
) -> PromptSuggestion | None:
    """Generate an actionable prompt for the user.

    Returns ``None`` if confidence is sufficient (no prompt needed).
    """
    if confidence >= 0:
        return None  # match is above threshold — no prompt needed

    if quality_result is not None:
        for reason, prompt in _REASON_PROMPTS:
            if reason in quality_result.rejection_reasons:
                return prompt

    return PromptSuggestion("Recognition failed. Please try again", "general")
