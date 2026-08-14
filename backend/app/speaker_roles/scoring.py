"""Deterministic, explainable heuristic scoring for DOCTOR/PATIENT role identification.

This is NOT voice biometrics, NOT an LLM judgement, and NOT a statistically
calibrated probability. It is a transparent, hand-weighted keyword/pattern
scorer over the ASR text attributed to each anonymous diarization speaker.
The weights below are reasonable, documented heuristics -- they have not been
tuned or validated against a labeled dataset. Formal evaluation (with a real
labeled evaluation set) is a later, separate step.

role_confidence is a normalized heuristic score in [0.0, 1.0], not a
calibrated probability and not an accuracy figure. A confidence of 0.90 does
NOT mean "90% likely correct" -- see the "note" field embedded in every
evidence payload.
"""
from dataclasses import dataclass, field

# (cue phrase, weight) -- weight is an arbitrary-but-documented relative
# strength, not a probability. Longer/more-specific phrases carry more
# weight than short generic ones, so no single common word can decide a role
# on its own. These are heuristic signals, not absolute rules.
DOCTOR_CUES: list[tuple[str, float]] = [
    ("what brings you in", 2.0),
    ("how are you feeling", 2.0),
    ("any pain", 1.5),
    ("any allergies", 1.5),
    ("let's check", 1.5),
    ("let me check", 1.5),
    ("take this medication", 2.0),
    ("i'm prescribing", 2.0),
    ("i will prescribe", 2.0),
    ("we'll run a test", 1.5),
    ("we will run some tests", 1.5),
    ("how long have you had", 2.0),
    ("how long has this been", 2.0),
    ("on a scale of", 1.5),
    ("any other symptoms", 1.5),
]

PATIENT_CUES: list[tuple[str, float]] = [
    ("i have been feeling", 1.5),
    ("i've been feeling", 1.5),
    ("it hurts", 1.5),
    ("since yesterday", 1.0),
    ("since last", 1.0),
    ("i've been taking", 1.0),
    ("i take", 1.0),
    ("my symptoms", 1.5),
    ("i feel", 1.5),
    ("i am experiencing", 1.5),
    ("i'm experiencing", 1.5),
    ("i have", 1.0),
]

# Doctors tend to ask more questions; patients tend to narrate in first
# person. Small, capped, comparative signals -- not standalone rules.
QUESTION_WEIGHT = 0.3
FIRST_PERSON_WEIGHT = 0.15
FIRST_PERSON_MARKERS = ("i ", "i'm", "i’ve", "i've", "my ")
FIRST_PERSON_CAP_PER_SEGMENT = 3

# Decision thresholds (documented, not derived from data yet).
CONFIDENT_THRESHOLD = 0.65  # minimum normalized confidence to assign a role
MIN_MARGIN = 0.15  # minimum gap between doctor/patient confidence to decide

# Two-speaker consistency fallback.
TWO_SPEAKER_STRONG_THRESHOLD = 0.80  # the "anchor" speaker must be this confident
TWO_SPEAKER_INFERRED_CONFIDENCE = 0.5  # deliberately conservative: inference, not direct evidence
# The other speaker only inherits the opposite role if it has literally ZERO
# matched evidence of its own -- any nonzero evidence, however small, blocks
# the inference so weak-but-real evidence is never overridden.


@dataclass
class SpeakerEvidence:
    speaker_label: str
    doctor_score: float = 0.0
    patient_score: float = 0.0
    matched_doctor_cues: list[str] = field(default_factory=list)
    matched_patient_cues: list[str] = field(default_factory=list)
    question_count: int = 0
    first_person_count: int = 0
    segment_count: int = 0


@dataclass
class RoleClassification:
    speaker_label: str
    role: str  # "DOCTOR" | "PATIENT" | "UNKNOWN"
    confidence: float  # 0.0-1.0 heuristic score, NOT a calibrated probability
    doctor_confidence: float
    patient_confidence: float
    evidence: dict


def score_speaker(speaker_label: str, texts: list[str]) -> SpeakerEvidence:
    """Deterministic keyword/pattern scoring over one speaker's ASR text."""
    evidence = SpeakerEvidence(speaker_label=speaker_label, segment_count=len(texts))
    for text in texts:
        lowered = text.lower()
        for cue, weight in DOCTOR_CUES:
            if cue in lowered:
                evidence.doctor_score += weight
                evidence.matched_doctor_cues.append(cue)
        for cue, weight in PATIENT_CUES:
            if cue in lowered:
                evidence.patient_score += weight
                evidence.matched_patient_cues.append(cue)
        if lowered.strip().endswith("?"):
            evidence.question_count += 1
            evidence.doctor_score += QUESTION_WEIGHT
        first_person_hits = sum(lowered.count(marker) for marker in FIRST_PERSON_MARKERS)
        if first_person_hits:
            evidence.first_person_count += first_person_hits
            evidence.patient_score += FIRST_PERSON_WEIGHT * min(
                first_person_hits, FIRST_PERSON_CAP_PER_SEGMENT
            )
    return evidence


def classify_speaker(evidence: SpeakerEvidence) -> RoleClassification:
    """Turn raw scores into a role + confidence using fixed, documented thresholds."""
    total = evidence.doctor_score + evidence.patient_score
    if total <= 0:
        doctor_confidence = 0.0
        patient_confidence = 0.0
    else:
        doctor_confidence = evidence.doctor_score / total
        patient_confidence = evidence.patient_score / total

    role = "UNKNOWN"
    confidence = max(doctor_confidence, patient_confidence)
    margin = abs(doctor_confidence - patient_confidence)

    if doctor_confidence >= CONFIDENT_THRESHOLD and margin >= MIN_MARGIN:
        role = "DOCTOR"
        confidence = doctor_confidence
    elif patient_confidence >= CONFIDENT_THRESHOLD and margin >= MIN_MARGIN:
        role = "PATIENT"
        confidence = patient_confidence

    return RoleClassification(
        speaker_label=evidence.speaker_label,
        role=role,
        confidence=round(confidence, 4),
        doctor_confidence=round(doctor_confidence, 4),
        patient_confidence=round(patient_confidence, 4),
        evidence={
            "method": "deterministic heuristic keyword/pattern scoring",
            "doctor_score": round(evidence.doctor_score, 4),
            "patient_score": round(evidence.patient_score, 4),
            "matched_doctor_cues": evidence.matched_doctor_cues,
            "matched_patient_cues": evidence.matched_patient_cues,
            "question_count": evidence.question_count,
            "first_person_count": evidence.first_person_count,
            "segment_count": evidence.segment_count,
            "note": (
                "role_confidence is a normalized heuristic score in [0.0, 1.0], "
                "not a calibrated probability and not an accuracy figure."
            ),
        },
    )


def apply_two_speaker_consistency(
    classifications: list[RoleClassification], evidences: dict[str, SpeakerEvidence]
) -> list[RoleClassification]:
    """If exactly two speakers exist and one is strongly DOCTOR/PATIENT while the
    other is UNKNOWN with literally zero evidence of its own, infer the
    opposite role for the second speaker with a deliberately conservative,
    fixed confidence. Never overrides a speaker that has any nonzero evidence
    of its own, however small -- weak-but-real evidence is never forced.
    """
    if len(classifications) != 2:
        return classifications

    a, b = classifications
    for strong, weak in ((a, b), (b, a)):
        if not (
            strong.role in ("DOCTOR", "PATIENT")
            and strong.confidence >= TWO_SPEAKER_STRONG_THRESHOLD
            and weak.role == "UNKNOWN"
        ):
            continue

        weak_evidence = evidences[weak.speaker_label]
        if weak_evidence.doctor_score == 0.0 and weak_evidence.patient_score == 0.0:
            opposite_role = "PATIENT" if strong.role == "DOCTOR" else "DOCTOR"
            weak.role = opposite_role
            weak.confidence = TWO_SPEAKER_INFERRED_CONFIDENCE
            weak.evidence["two_speaker_consistency_applied"] = True
            weak.evidence["note"] = (
                "Role inferred via two-speaker consistency: the other speaker was "
                f"confidently classified as {strong.role} (confidence "
                f"{strong.confidence}) and this speaker had no independent lexical "
                "evidence of its own. This is an inference, not direct evidence, "
                "so confidence is deliberately conservative."
            )
    return classifications
