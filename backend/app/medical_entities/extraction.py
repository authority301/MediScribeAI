"""Deterministic, rule/pattern-based medical entity extraction baseline.

This is an INFORMATION EXTRACTION layer, not a diagnostic system:
  - It only surfaces terms that are literally present in the transcript text.
  - It does NOT infer conditions, diagnoses, or clinical facts that are not
    explicitly stated ("my mother has diabetes" must not become a patient
    diagnosis -- see _is_family_referenced below).
  - confidence_score is a heuristic extraction/match-strength score, NOT
    clinical certainty and NOT a calibrated probability.

No ML model, no LLM, no external API -- entirely local regex/keyword
matching over already-transcribed text, so it works fully offline.
"""
import re
import unicodedata
from dataclasses import dataclass

from app.medical_entities.vocabulary import VOCAB_BY_ENTITY_TYPE

VOCAB_MATCH_CONFIDENCE = 0.75
PATTERN_MATCH_CONFIDENCE = 0.7

NEGATION_CUES = [
    "no ",
    "not ",
    "n't",
    "never",
    "without",
    "denies",
    "denied",
]

HISTORICAL_CUES = [
    "as a child",
    "in the past",
    "used to have",
    "used to get",
    "previously",
    "years ago",
    "in my childhood",
    "when i was young",
    "history of",
    "had it before",
]

# Conservative safety rule: if a clinical term is mentioned in the same
# sentence as a family-member reference, we do NOT extract it at all rather
# than risk attributing someone else's condition to the patient (there is no
# "subject" field in this schema to record the distinction otherwise).
FAMILY_REFERENCE_CUES = [
    "my mother",
    "my father",
    "my sister",
    "my brother",
    "my wife",
    "my husband",
    "my son",
    "my daughter",
    "my grandmother",
    "my grandfather",
    "his mother",
    "her mother",
    "his father",
    "her father",
    "their mother",
    "their father",
]

# DURATION / FREQUENCY / MEASUREMENT are pattern-based, not vocabulary-based.
_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|a|an|couple of|few"
)
_DURATION_UNIT = r"(?:day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)"
DURATION_PATTERN = re.compile(
    rf"\b(?:\d+|{_NUMBER_WORDS})\s*{_DURATION_UNIT}\b", re.IGNORECASE
)

FREQUENCY_PATTERN = re.compile(
    r"\b(?:once|twice|thrice|\d+\s*times)\s*(?:a|per)\s*(?:day|week|month|hour)\b"
    r"|\b(?:daily|weekly|twice a day|every morning|every night|every day)\b",
    re.IGNORECASE,
)

MEASUREMENT_PATTERN = re.compile(
    r"\b\d{2,3}\s*/\s*\d{2,3}\s*(?:mmhg)?\b"  # blood pressure, e.g. 120/80
    r"|\b\d+(?:\.\d+)?\s*(?:mg|ml|kg|lbs|°c|°f|degrees?)\b",
    re.IGNORECASE,
)

REGEX_ENTITY_TYPES: dict[str, re.Pattern] = {
    "DURATION": DURATION_PATTERN,
    "FREQUENCY": FREQUENCY_PATTERN,
    "MEASUREMENT": MEASUREMENT_PATTERN,
}


@dataclass
class ExtractedEntity:
    entity_type: str
    entity_text: str
    normalized_text: str
    start_offset: int
    end_offset: int
    confidence: float
    negated: bool
    historical: bool


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split text into sentences, returning (start, end) offsets into text."""
    spans = []
    start = 0
    for match in re.finditer(r"[.!?]+", text):
        end = match.end()
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def _is_negated(sentence: str, match_start_in_sentence: int) -> bool:
    before = sentence[:match_start_in_sentence].lower()
    return any(cue in before for cue in NEGATION_CUES)


def _is_historical(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(cue in lowered for cue in HISTORICAL_CUES)


def _is_family_referenced(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(cue in lowered for cue in FAMILY_REFERENCE_CUES)


def _is_word_char(char: str) -> bool:
    # Regex \w/\b are ASCII-centric: Tamil vowel signs and virama are
    # combining marks (category Mn), which \w does not count as word chars,
    # so \b creates a spurious boundary mid-word for Tamil. Unicode general
    # category (letter/mark/digit vs. everything else) works for both.
    if char == "_":
        return True
    return unicodedata.category(char)[0] in ("L", "M", "N")


def _has_lexical_boundary(text: str, start: int, end: int) -> bool:
    """True if text[start:end] is not a substring of a larger word."""
    if start > 0 and _is_word_char(text[start - 1]):
        return False
    if end < len(text) and _is_word_char(text[end]):
        return False
    return True


def _find_vocab_matches(text: str) -> list[ExtractedEntity]:
    matches: list[tuple[int, int, str, str, str]] = []  # start, end, type, text, normalized
    claimed: list[tuple[int, int]] = []

    def _overlaps_claimed(start: int, end: int) -> bool:
        return any(start < c_end and end > c_start for c_start, c_end in claimed)

    for entity_type, canonical_map in VOCAB_BY_ENTITY_TYPE.items():
        # Longer variants first so more-specific phrases win over substrings
        # (e.g. "high fever" is preferred over a bare "fever" at the same spot).
        variants = sorted(
            ((canonical, variant) for canonical, forms in canonical_map.items() for variant in forms),
            key=lambda pair: len(pair[1]),
            reverse=True,
        )
        for canonical, variant in variants:
            for match in re.finditer(re.escape(variant), text, re.IGNORECASE):
                start, end = match.start(), match.end()
                if not _has_lexical_boundary(text, start, end):
                    continue  # reject partial-word matches, e.g. "ear" inside "appears"
                if _overlaps_claimed(start, end):
                    continue
                claimed.append((start, end))
                matches.append((start, end, entity_type, match.group(0), canonical))

    entities = []
    sentences = _sentence_spans(text)
    for start, end, entity_type, matched_text, canonical in matches:
        sentence_start, sentence_end = next(
            ((s, e) for s, e in sentences if s <= start < e), (0, len(text))
        )
        sentence = text[sentence_start:sentence_end]

        if _is_family_referenced(sentence):
            continue  # conservative: do not attribute another person's mention to the patient

        negated = _is_negated(sentence, start - sentence_start)
        historical = _is_historical(sentence)

        entities.append(
            ExtractedEntity(
                entity_type=entity_type,
                entity_text=matched_text,
                normalized_text=canonical,
                start_offset=start,
                end_offset=end,
                confidence=VOCAB_MATCH_CONFIDENCE,
                negated=negated,
                historical=historical,
            )
        )
    return entities


def _find_pattern_matches(text: str) -> list[ExtractedEntity]:
    entities = []
    sentences = _sentence_spans(text)
    for entity_type, pattern in REGEX_ENTITY_TYPES.items():
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            sentence_start, sentence_end = next(
                ((s, e) for s, e in sentences if s <= start < e), (0, len(text))
            )
            sentence = text[sentence_start:sentence_end]
            if _is_family_referenced(sentence):
                continue
            entities.append(
                ExtractedEntity(
                    entity_type=entity_type,
                    entity_text=match.group(0),
                    normalized_text=match.group(0),
                    start_offset=start,
                    end_offset=end,
                    confidence=PATTERN_MATCH_CONFIDENCE,
                    negated=False,
                    historical=False,
                )
            )
    return entities


def extract_entities(text: str) -> list[ExtractedEntity]:
    """Extract clinical entities from a single piece of text (e.g. one speaker
    segment's ASR text). Offsets are relative to `text`, not any larger
    document. Deterministic: identical input always yields identical output.
    """
    if not text:
        return []
    return _find_vocab_matches(text) + _find_pattern_matches(text)
