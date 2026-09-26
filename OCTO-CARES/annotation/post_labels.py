"""Canonical post-label definitions and ordering."""

from __future__ import annotations

from dataclasses import dataclass


DECISION_RULE_TEXT = (
    "Answer Yes if any part of the post satisfies the category, not only if it is the main theme."
)


@dataclass(frozen=True)
class LabelDefinition:
    key: str
    name_en: str
    name_zh: str
    definition_en: str
    definition_zh: str = ""


POST_LABELS = (
    LabelDefinition(
        key="positive_informational_self_disclosure",
        name_en="Positive informational self-disclosure",
        name_zh="正面信息自我披露",
        definition_en=(
            "Describes an event, experience, or personal circumstance in the "
            "author's own life that is positive, beneficial, or lead life in a good direction."
        
        )
    ),
    LabelDefinition(
        key="negative_informational_self_disclosure",
        name_en="Negative informational self-disclosure",
        name_zh="负面信息自我披露",
        definition_en=(
            "Describes an event, experience, or personal circumstance in the "
            "author's own life that is negative, harmful, worsening, difficult, "
            "or lead life in a bad direction."
        )
    ),
    LabelDefinition(
        key="neutral_informational_self_disclosure",
        name_en="Neutral informational self-disclosure",
        name_zh="中性信息自我披露",
        definition_en=(
            "Describes a factual event, experience, or personal circumstance in "
            "the author's own life without presenting it as clearly positive or "
            "negative in the current context."
        ),
    ),
    LabelDefinition(
        key="positive_emotional_self_disclosure",
        name_en="Positive emotional self-disclosure",
        name_zh="正面情绪自我披露",
        definition_en=(
            "Express positive emotions, such as happiness, relief, gratitude, hope, reassurance, pride, or excitement. Expressing positive and forward-looking emotional states or subjective feelings. The key lies in capturing the speaker's genuine positive experiences within their own mind, rather than relying on the objective nature of the event. Even if it is describing something good, if the tone is calm and neutral, it should not be classified under this category. The key to judgment lies in understanding the speaker's thoughts. This can be done by directly expressing feelings or implicitly revealing them, such as through emotional words, adverbs, and adjectives."
        )
    ),
    LabelDefinition(
        key="negative_emotional_self_disclosure",
        name_en="Negative emotional self-disclosure",
        name_zh="负面情绪自我披露",
        definition_en=(
            "Express negative emotions, such as fear, sadness, anxiety, loneliness, anger, hopelessness, shame, frustration, or feeling overwhelmed. Expressing negative or negative emotional states or subjective troubles. The focus should be on the speaker's feelings of pain, anger, anxiety, depression, helplessness, guilt, etc., rather than the negative nature of the event itself. Even if it is describing something bad, if the tone is restrained and without emotion, it does not fall into this category. The key to judgment lies in understanding the speaker's thoughts. This can be done by directly expressing feelings or implicitly revealing them, such as through emotional words, adverbs, and adjectives."
        )
    ),
    LabelDefinition(
        key="seek_emotional_support",
        name_en="Seek emotional support",
        name_zh="寻求情感支持",
        definition_en=(
            "Attempts to obtain understanding, comfort, encouragement, "
            "affirmation, empathy, caring, sympathy, or emotional reassurance "
            "from readers or the community. The request may be explicit or "
            "indirect, but it must show an expectation of emotional response."
        )
    ),
    LabelDefinition(
        key="seek_informational_support",
        name_en="Seek informational support",
        name_zh="寻求信息支持",
        definition_en=(
            "Attempts to obtain knowledge, explanations, advice, experiential "
            "information, guidance, referrals, solutions, or decision support "
            "from readers or the community."
        )
    ),
    LabelDefinition(
        key="seek_companionship",
        name_en="Seek companionship",
        name_zh="寻求陪伴",
        definition_en=(
            "Asks if there is anyone who is like oneself, or if there is anyone who shares the same/similar experience."
            "Or attempts to obtain social presence, continued interaction, "
            "conversation, contact, shared activity, or relational companionship."
        )
    ),
)

POST_LABEL_KEYS = tuple(label.key for label in POST_LABELS)
