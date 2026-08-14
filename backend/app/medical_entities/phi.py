"""Deterministic, regex/heuristic PHI (Protected Health Information) detection
and de-identification.

This is a SEPARATE concern from medical entity extraction: PHI detection asks
"is this sensitive personal information", not "is this a clinical fact".

IMPORTANT LIMITATIONS (documented, not hidden):
  - PERSON_NAME detection here is a simple cue-phrase + capitalization
    heuristic, NOT a validated named-entity-recognition model. Expect both
    false positives and false negatives on real-world text.
  - PHI detection here is a prototype safety layer, not a certified
    de-identification system. It does NOT constitute HIPAA compliance or any
    other formal privacy certification -- that requires a separate,
    dedicated compliance review.
  - Detected entity_text for PHI is always a redaction placeholder (e.g.
    "[PERSON_NAME]"), never the raw matched value -- callers must not log or
    persist the raw PHI substring outside of building the de-identified text.
"""
import re
from dataclasses import dataclass

PHI_MATCH_CONFIDENCE = 0.6
PERSON_NAME_MATCH_CONFIDENCE = 0.5  # lowest confidence: heuristic, most error-prone

EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# 10-digit numbers, optionally with a country code / separators (e.g. Indian
# mobile numbers, or "+91 98765 43210" / "987-654-3210" style formats).
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\d{3,5}[-.\s]?\d{3}[-.\s]?\d{3,4}\b")

DATE_PATTERN = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{2,4}\b",
    re.IGNORECASE,
)

IDENTIFIER_PATTERN = re.compile(
    r"\b(?:MRN|Patient ID|ID|Aadhaar|SSN)[:\s]*[A-Za-z0-9-]{4,}\b", re.IGNORECASE
)

ADDRESS_PATTERN = re.compile(
    r"\b\d{1,5}[,]?\s+[A-Za-z0-9\s]{2,30}?"
    r"(?:Street|St\.|Road|Rd\.|Avenue|Ave\.|Nagar|Colony|Lane)\b",
    re.IGNORECASE,
)

# PERSON_NAME: only triggers after a strong cue phrase or an honorific, to
# keep false positives bounded (a bare "John Smith" with no cue is NOT
# flagged -- a known, documented limitation of this prototype heuristic).
_NAME = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}"
PERSON_NAME_PATTERN = re.compile(
    rf"\b(?:my name is|i am|this is|call me|patient name is|name:)\s+({_NAME})"
    rf"|\b(?:Mr|Mrs|Ms|Dr)\.?\s+({_NAME})",
    re.IGNORECASE,
)

PHI_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": EMAIL_PATTERN,
    "PHONE_NUMBER": PHONE_PATTERN,
    "DATE": DATE_PATTERN,
    "IDENTIFIER": IDENTIFIER_PATTERN,
    "ADDRESS": ADDRESS_PATTERN,
}


@dataclass
class PhiDetection:
    phi_type: str  # PERSON_NAME | PHONE_NUMBER | EMAIL | DATE | ADDRESS | IDENTIFIER
    start_offset: int
    end_offset: int
    confidence: float


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and a_end > b_start


def detect_phi(text: str) -> list[PhiDetection]:
    """Detect PHI spans in text. Returns offsets only -- callers are
    responsible for never logging/persisting the raw substring itself.
    """
    if not text:
        return []

    detections: list[PhiDetection] = []

    for match in PERSON_NAME_PATTERN.finditer(text):
        group = match.group(1) or match.group(2)
        if not group:
            continue
        start = match.start(1) if match.group(1) else match.start(2)
        end = match.end(1) if match.group(1) else match.end(2)
        detections.append(
            PhiDetection(
                phi_type="PERSON_NAME",
                start_offset=start,
                end_offset=end,
                confidence=PERSON_NAME_MATCH_CONFIDENCE,
            )
        )

    for phi_type, pattern in PHI_PATTERNS.items():
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            if any(_overlaps(start, end, d.start_offset, d.end_offset) for d in detections):
                continue
            detections.append(
                PhiDetection(
                    phi_type=phi_type,
                    start_offset=start,
                    end_offset=end,
                    confidence=PHI_MATCH_CONFIDENCE,
                )
            )

    detections.sort(key=lambda d: d.start_offset)
    return detections


def deidentify_text(text: str, detections: list[PhiDetection] | None = None) -> str:
    """Return a de-identified copy of text with every detected PHI span
    replaced by a "[PHI_TYPE]" placeholder. The original text is never
    modified -- this always returns a new string.
    """
    if not text:
        return text
    if detections is None:
        detections = detect_phi(text)

    result = text
    for detection in sorted(detections, key=lambda d: d.start_offset, reverse=True):
        placeholder = f"[{detection.phi_type}]"
        result = result[: detection.start_offset] + placeholder + result[detection.end_offset :]
    return result
