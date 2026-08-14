"""Deterministic, template-based SOAP claim generator baseline.

CRITICAL RESEARCH REQUIREMENT: this generator must never invent facts. Every
claim is built directly from either (a) an already-extracted medical entity
(Step 12 -- SYMPTOM, MEDICATION, ALLERGY, MEASUREMENT, MEDICAL_TEST,
MEDICAL_PROCEDURE, DURATION) or (b) an explicit cue-phrase match against a
DOCTOR-attributed utterance. If a section has no qualifying evidence, the
caller leaves it as "Not documented." rather than filling in a guess.

This is a downstream summarization/formatting component, NOT the project's
research novelty (that remains Evidence-Linked Hallucination Detection). It
does not diagnose, does not infer beyond what was said, and does not decide
whether its own claims are correct -- that determination (evidence
retrieval + NLI + hallucination detection) is Step 14, not implemented here.

generation_confidence on each claim reflects generation/evidence-availability
strength (how well the underlying extraction/cue match went), never clinical
correctness and never a calibrated probability.
"""
from dataclasses import dataclass

# Fixed confidence for claims derived from a cue-phrase match rather than a
# Step 12 extracted entity (which already carries its own confidence_score).
CUE_PHRASE_CLAIM_CONFIDENCE = 0.6

ASSESSMENT_CUES = [
    "this appears to be",
    "this looks like",
    "this seems to be",
    "my assessment is",
    "this is consistent with",
    "the diagnosis is",
    "i believe this is",
    "this could be",
]

FOLLOW_UP_CUES = ["return in", "follow up in", "follow-up in", "come back in"]
REFERRAL_CUES = ["i'll refer you", "i will refer you", "referring you to"]

SUBJECTIVE = "SUBJECTIVE"
OBJECTIVE = "OBJECTIVE"
ASSESSMENT = "ASSESSMENT"
PLAN = "PLAN"


@dataclass
class GeneratedClaim:
    section: str
    text: str
    confidence: float


def _entity_text(entity) -> str:
    return entity.normalized_value or entity.entity_text


def _find_assessment_phrase(segment_text: str) -> str | None:
    lowered = segment_text.lower()
    for cue in ASSESSMENT_CUES:
        idx = lowered.find(cue)
        if idx == -1:
            continue
        after = segment_text[idx + len(cue) :]
        end = len(after)
        for punct in (".", "!", "?"):
            pos = after.find(punct)
            if pos != -1:
                end = min(end, pos)
        phrase = after[:end].strip()
        if phrase:
            return phrase
    return None


def _has_cue(segment_text: str, cues: list[str]) -> bool:
    lowered = segment_text.lower()
    return any(cue in lowered for cue in cues)


def _entity_confidence(entity) -> float:
    return float(entity.confidence_score) if entity.confidence_score is not None else CUE_PHRASE_CLAIM_CONFIDENCE


def _subjective_claims(segment_entities: list) -> list[GeneratedClaim]:
    claims: list[GeneratedClaim] = []

    symptoms = [e for e in segment_entities if e.entity_type == "SYMPTOM"]
    durations = [e for e in segment_entities if e.entity_type == "DURATION"]
    # Only fuse a duration into a symptom claim when it's unambiguous (exactly
    # one duration mentioned in this same utterance) -- never guessed.
    duration_text = _entity_text(durations[0]) if len(durations) == 1 else None

    for symptom in symptoms:
        text = _entity_text(symptom)
        if symptom.negated:
            claim_text = f"Patient denies {text}."
        elif symptom.historical:
            claim_text = f"Patient reports history of {text}."
        elif duration_text:
            claim_text = f"Patient reports {text} for {duration_text}."
        else:
            claim_text = f"Patient reports {text}."
        claims.append(GeneratedClaim(SUBJECTIVE, claim_text, _entity_confidence(symptom)))

    for allergy in (e for e in segment_entities if e.entity_type == "ALLERGY"):
        text = _entity_text(allergy)
        claim_text = f"Patient denies allergy to {text}." if allergy.negated else f"Patient reports allergy to {text}."
        claims.append(GeneratedClaim(SUBJECTIVE, claim_text, _entity_confidence(allergy)))

    for medication in (e for e in segment_entities if e.entity_type == "MEDICATION"):
        text = _entity_text(medication)
        claim_text = (
            f"Patient denies taking {text}." if medication.negated else f"Patient reports taking {text}."
        )
        claims.append(GeneratedClaim(SUBJECTIVE, claim_text, _entity_confidence(medication)))

    # Patient-reported condition mentions (e.g. "I had asthma as a child.").
    # historical=True keeps these as reported history, never a current
    # diagnosis -- and this never feeds ASSESSMENT, which only comes from an
    # explicit doctor assessment cue (see _assessment_claims).
    for diagnosis in (e for e in segment_entities if e.entity_type == "DIAGNOSIS_MENTION"):
        text = _entity_text(diagnosis)
        if diagnosis.negated:
            claim_text = f"Patient denies {text}."
        elif diagnosis.historical:
            claim_text = f"Patient reports history of {text}."
        else:
            claim_text = f"Patient reports {text}."
        claims.append(GeneratedClaim(SUBJECTIVE, claim_text, _entity_confidence(diagnosis)))

    return claims


def _objective_claims(segment_entities: list) -> list[GeneratedClaim]:
    return [
        GeneratedClaim(
            OBJECTIVE, f"Measurement documented: {_entity_text(m)}.", _entity_confidence(m)
        )
        for m in segment_entities
        if m.entity_type == "MEASUREMENT"
    ]


def _plan_claims(segment_entities: list, segment_text: str | None) -> list[GeneratedClaim]:
    claims: list[GeneratedClaim] = []

    for medication in (e for e in segment_entities if e.entity_type == "MEDICATION"):
        text = _entity_text(medication)
        claim_text = f"Doctor advised against {text}." if medication.negated else f"Doctor advised {text}."
        claims.append(GeneratedClaim(PLAN, claim_text, _entity_confidence(medication)))

    for test in (e for e in segment_entities if e.entity_type == "MEDICAL_TEST"):
        claims.append(
            GeneratedClaim(PLAN, f"Doctor ordered {_entity_text(test)}.", _entity_confidence(test))
        )

    for procedure in (e for e in segment_entities if e.entity_type == "MEDICAL_PROCEDURE"):
        claims.append(
            GeneratedClaim(
                PLAN, f"Doctor recommended {_entity_text(procedure)}.", _entity_confidence(procedure)
            )
        )

    if segment_text:
        if _has_cue(segment_text, FOLLOW_UP_CUES):
            durations = [e for e in segment_entities if e.entity_type == "DURATION"]
            if len(durations) == 1:
                claims.append(
                    GeneratedClaim(
                        PLAN,
                        f"Doctor advised follow-up in {_entity_text(durations[0])}.",
                        CUE_PHRASE_CLAIM_CONFIDENCE,
                    )
                )
            else:
                claims.append(
                    GeneratedClaim(PLAN, "Doctor advised a follow-up visit.", CUE_PHRASE_CLAIM_CONFIDENCE)
                )

        if _has_cue(segment_text, REFERRAL_CUES):
            claims.append(
                GeneratedClaim(PLAN, "Doctor advised referral to a specialist.", CUE_PHRASE_CLAIM_CONFIDENCE)
            )

    return claims


def _assessment_claims(segment_text: str | None) -> list[GeneratedClaim]:
    if not segment_text:
        return []
    phrase = _find_assessment_phrase(segment_text)
    if not phrase:
        return []
    return [
        GeneratedClaim(
            ASSESSMENT, f"Doctor assessed the condition as {phrase}.", CUE_PHRASE_CLAIM_CONFIDENCE
        )
    ]


def generate_claims(segments: list, entities: list) -> list[GeneratedClaim]:
    """segments: ordered SpeakerSegment rows (id, inferred_role, segment_text).
    entities: MedicalEntity rows already excluding PHI_* types (caller's job).

    Deterministic: identical input always yields identical, ordered output.
    Iterates segments in their existing sequence_index order, so claim order
    mirrors conversation order.
    """
    entities_by_segment: dict = {}
    for entity in entities:
        entities_by_segment.setdefault(entity.speaker_segment_id, []).append(entity)

    claims: list[GeneratedClaim] = []
    for segment in segments:
        segment_entities = entities_by_segment.get(segment.id, [])

        if segment.inferred_role == "PATIENT":
            claims.extend(_subjective_claims(segment_entities))

        claims.extend(_objective_claims(segment_entities))

        if segment.inferred_role == "DOCTOR":
            claims.extend(_plan_claims(segment_entities, segment.segment_text))
            claims.extend(_assessment_claims(segment.segment_text))

    return claims
