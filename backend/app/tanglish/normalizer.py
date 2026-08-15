"""Prototype Tanglish-aware clinical semantic normalization (Step 14F).

============================================================================
WHAT THIS IS -- AND IS NOT
============================================================================
This is NOT a Tamil -> English machine translator. It is a deterministic,
rule-based CLINICAL SEMANTIC NORMALIZER: it detects a small, documented set
of Romanized-Tamil clinical conversational patterns (negation, current
state, historical state, duration, frequency) attached to a recognized
clinical entity, and rewrites them into an explicit English sentence in the
same style as the project's SOAP claims ("Patient has fever for two days.").
Clinical English terms already present in the input (fever, aspirin, blood
pressure, ...) are preserved verbatim, never translated.

Anything not covered by these rules -- unrecognized vocabulary, ambiguous
constructions, non-clinical conversation -- passes through UNCHANGED rather
than being guessed at. This is a deliberate research-integrity choice: a
wrong invented normalization would be worse than no normalization.

English-only input is detected and returned completely unchanged (the
"bypassable" requirement) so this layer cannot degrade the existing English
baseline.

See backend/app/tanglish/README.md for the full architecture description,
supported patterns, and limitations.

============================================================================
NEGATION SAFETY
============================================================================
Negation is never detected via an unsafe bare substring check (e.g.
`"la" in text`). Every trigger is matched as a whole word/phrase with
regex word boundaries, and only fires when an adjacent recognized clinical
entity is found nearby in the same clause -- see `_associate_entities_with_triggers`.
"""
import re
from dataclasses import dataclass

from app.tanglish import lexicon
from app.tanglish.config import ENTITY_TRIGGER_WINDOW_CHARS
from app.tanglish.schemas import NormalizedTextResult, SemanticFeatures, Transformation

LANGUAGE_ENGLISH = "ENGLISH"
LANGUAGE_TANGLISH = "TANGLISH"
LANGUAGE_MIXED = "MIXED"
LANGUAGE_UNKNOWN = "UNKNOWN"

_WORD_TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")

_FIRST_PERSON_MARKERS = {"enakku", "naan", "en", "ennaku", "ennoda"}


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_TOKEN_RE.finditer(text)]


def detect_language(text: str) -> str:
    """Lightweight, deterministic, intentionally conservative heuristic.
    Not linguistically rigorous -- see module/README limitations."""
    if not text or not text.strip():
        return LANGUAGE_UNKNOWN

    tokens = set(_tokens(text))
    if not tokens:
        return LANGUAGE_UNKNOWN

    tamil_marker_hit = bool(tokens & lexicon.TAMIL_SINGLE_WORD_MARKERS) or any(
        phrase in " ".join(_tokens(text)) for phrase in lexicon.TAMIL_PHRASE_MARKERS
    )
    english_structure_hit = bool(tokens & lexicon.ENGLISH_STRUCTURE_WORDS)

    if not tamil_marker_hit:
        return LANGUAGE_ENGLISH
    if not english_structure_hit:
        return LANGUAGE_TANGLISH
    return LANGUAGE_MIXED


def _has_clinical_english(text: str) -> bool:
    return _ENTITY_PATTERN.search(text) is not None


# ---------------------------------------------------------------------------
# Span-based entity/trigger extraction
# ---------------------------------------------------------------------------
@dataclass
class _Span:
    start: int
    end: int
    text: str


@dataclass
class _EntityMatch(_Span):
    canonical: str
    entity_type: str


@dataclass
class _TriggerMatch(_Span):
    category: str  # NEG_STATE | NEG_ACTION | CURRENT | HIST_VERB


def _compile_term_pattern(terms: list[str]) -> re.Pattern:
    ordered = sorted(terms, key=len, reverse=True)
    parts = []
    for term in ordered:
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        parts.append(escaped)
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


_ENTITY_PATTERN = _compile_term_pattern(list(lexicon.CLINICAL_TERMS.keys()))
_NEG_STATE_PATTERN = _compile_term_pattern(lexicon.NEGATION_TRIGGERS_STATE)
_NEG_ACTION_PATTERN = _compile_term_pattern(lexicon.NEGATION_TRIGGERS_ACTION)
_CURRENT_PATTERN = _compile_term_pattern(lexicon.CURRENT_STATE_TRIGGERS)
_HIST_VERB_PATTERN = _compile_term_pattern(lexicon.HISTORICAL_VERB_TRIGGERS)
_HIST_TIME_CHILDHOOD_PATTERN = _compile_term_pattern(lexicon.HISTORICAL_TIME_MARKERS_CHILDHOOD)
_HIST_TIME_PAST_PATTERN = _compile_term_pattern(lexicon.HISTORICAL_TIME_MARKERS_PAST)

# Numeral word alternatives shared by duration/frequency extraction. Tamil
# numeral words AND already-English numeral words (e.g. "three" inside an
# otherwise Tanglish/mixed clause -- Step 14H) are both recognized, plus a
# bare-digit fallback for arbitrary Arabic numerals.
_numeral_alternatives = (
    sorted(list(lexicon.TAMIL_NUMERALS.keys()) + list(lexicon.ENGLISH_NUMBER_WORD_VALUES.keys()), key=len, reverse=True)
    + [r"\d+"]
)
_duration_unit_alternatives = sorted(lexicon.DURATION_UNIT_WORDS.keys(), key=len, reverse=True)
_DURATION_PATTERN = re.compile(
    r"\b(?P<num>" + "|".join(_numeral_alternatives) + r")\s*-?\s*"
    r"(?P<unit>" + "|".join(_duration_unit_alternatives) + r")\b",
    re.IGNORECASE,
)

_freq_numeral_alternatives = sorted(list(lexicon.TAMIL_NUMERALS.keys()), key=len, reverse=True) + [r"\d+"]
_FREQUENCY_PATTERN = re.compile(
    r"\b(?P<num>" + "|".join(_freq_numeral_alternatives) + r")\s*-?\s*(?P<unit>thadava|velai)\b",
    re.IGNORECASE,
)

_TIME_OF_DAY_PATTERN = _compile_term_pattern(list(lexicon.TIME_OF_DAY_MARKERS.keys()))

# ---------------------------------------------------------------------------
# Measurement values (Step 14H). A NUMBER (Arabic, optionally decimal)
# followed by a recognized measurement unit word/symbol. Deliberately
# conservative: no unit conversion is attempted, the matched span is
# preserved verbatim. See _measurement_phrase_for_entity for the (narrower)
# bare-number fallback used when no unit word is present at all.
# ---------------------------------------------------------------------------
_measurement_unit_alternatives = sorted(lexicon.MEASUREMENT_UNIT_TERMS, key=len, reverse=True)
_MEASUREMENT_PATTERN = re.compile(
    r"\b(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>" + "|".join(re.escape(u) for u in _measurement_unit_alternatives) + r")",
    re.IGNORECASE,
)
_BARE_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")

# ---------------------------------------------------------------------------
# Family / third-person attribution (Step 14H). See lexicon.py for the
# marker vocabulary and the specific-vs-generic distinction.
# ---------------------------------------------------------------------------
_FAMILY_SPECIFIC_PATTERN = _compile_term_pattern(list(lexicon.FAMILY_SPECIFIC_RELATION_MARKERS.keys()))
_FAMILY_PLURAL_PATTERN = _compile_term_pattern(list(lexicon.FAMILY_PLURAL_RELATION_MARKERS.keys()))
_FAMILY_GENERIC_PATTERN = _compile_term_pattern(lexicon.FAMILY_GENERIC_MARKERS)
_FIRST_PERSON_PATTERN = _compile_term_pattern(list(_FIRST_PERSON_MARKERS))


def _find_entities(clause: str) -> list[_EntityMatch]:
    matches = []
    for m in _ENTITY_PATTERN.finditer(clause):
        surface = m.group(0).lower()
        canonical, entity_type = None, None
        for term, (canon, etype) in lexicon.CLINICAL_TERMS.items():
            if re.fullmatch(re.escape(term).replace(r"\ ", r"\s+"), surface, re.IGNORECASE):
                canonical, entity_type = canon, etype
                break
        if canonical is None:
            continue
        matches.append(_EntityMatch(start=m.start(), end=m.end(), text=m.group(0), canonical=canonical, entity_type=entity_type))
    return matches


def _find_triggers(clause: str) -> list[_TriggerMatch]:
    triggers = []
    for m in _NEG_STATE_PATTERN.finditer(clause):
        triggers.append(_TriggerMatch(start=m.start(), end=m.end(), text=m.group(0), category="NEG_STATE"))
    for m in _NEG_ACTION_PATTERN.finditer(clause):
        triggers.append(_TriggerMatch(start=m.start(), end=m.end(), text=m.group(0), category="NEG_ACTION"))
    for m in _HIST_VERB_PATTERN.finditer(clause):
        triggers.append(_TriggerMatch(start=m.start(), end=m.end(), text=m.group(0), category="HIST_VERB"))
    for m in _CURRENT_PATTERN.finditer(clause):
        triggers.append(_TriggerMatch(start=m.start(), end=m.end(), text=m.group(0), category="CURRENT"))
    return triggers


# Priority order: negation is the highest-priority failure mode (per Step
# 14C findings), then historical, then current state.
_CATEGORY_PRIORITY = {"NEG_STATE": 0, "NEG_ACTION": 0, "HIST_VERB": 1, "CURRENT": 2}


@dataclass
class _Fact:
    entity: _EntityMatch
    trigger: _TriggerMatch


def _associate_entities_with_triggers(entities: list[_EntityMatch], triggers: list[_TriggerMatch]) -> list[_Fact]:
    """Greedily associate each trigger with the nearest PRECEDING,
    not-yet-claimed clinical entity within ENTITY_TRIGGER_WINDOW_CHARS.
    Processes triggers in priority order (negation first) so that when a
    trigger's window could plausibly overlap another, negation wins ties.
    A trigger with no adjacent entity within window is simply dropped --
    no fact is invented.
    """
    ordered_triggers = sorted(triggers, key=lambda t: (_CATEGORY_PRIORITY.get(t.category, 99), t.start))
    claimed_entity_ids: set[int] = set()
    facts: list[_Fact] = []

    for trig in ordered_triggers:
        best_entity = None
        best_distance = None
        for ent in entities:
            if id(ent) in claimed_entity_ids:
                continue
            if ent.end <= trig.start:
                distance = trig.start - ent.end
                if distance <= ENTITY_TRIGGER_WINDOW_CHARS and (best_distance is None or distance < best_distance):
                    best_entity = ent
                    best_distance = distance
        if best_entity is not None:
            claimed_entity_ids.add(id(best_entity))
            facts.append(_Fact(entity=best_entity, trigger=trig))

    facts.sort(key=lambda f: f.entity.start)
    return facts


# ---------------------------------------------------------------------------
# Duration / frequency / historical qualifier extraction
# ---------------------------------------------------------------------------
def _numeral_value(token: str) -> int | None:
    token = token.lower()
    if token in lexicon.TAMIL_NUMERALS:
        return lexicon.TAMIL_NUMERALS[token]
    if token in lexicon.ENGLISH_NUMBER_WORD_VALUES:
        return lexicon.ENGLISH_NUMBER_WORD_VALUES[token]
    if token.isdigit():
        return int(token)
    return None


def _duration_phrase(clause: str) -> tuple[str, str] | None:
    """Returns (original_span_text, canonical_phrase) or None."""
    m = _DURATION_PATTERN.search(clause)
    if not m:
        return None
    value = _numeral_value(m.group("num"))
    if value is None:
        return None
    unit = lexicon.DURATION_UNIT_WORDS.get(m.group("unit").lower())
    if unit is None:
        return None
    number_word = lexicon.ENGLISH_NUMBER_WORDS.get(value, str(value))
    plural = "" if value == 1 else "s"
    return m.group(0), f"for {number_word} {unit}{plural}"


def _frequency_phrase(clause: str) -> tuple[str, str] | None:
    m = _FREQUENCY_PATTERN.search(clause)
    if not m:
        return None
    value = _numeral_value(m.group("num"))
    if value is None:
        return None
    if value == 1:
        phrase = "once"
    elif value == 2:
        phrase = "twice"
    else:
        number_word = lexicon.ENGLISH_NUMBER_WORDS.get(value, str(value))
        phrase = f"{number_word} times"
    return m.group(0), phrase


def _historical_qualifier(clause: str) -> str | None:
    if _HIST_TIME_CHILDHOOD_PATTERN.search(clause):
        return "during childhood"
    if _HIST_TIME_PAST_PATTERN.search(clause):
        return "in the past"
    return None


# ---------------------------------------------------------------------------
# Measurement value preservation (Step 14H).
#
# Scoped deliberately narrowly: only fires for entities already typed
# "measurement" in the lexicon (heart rate, temperature, oxygen
# saturation/level, blood pressure, weight, blood sugar), and only searches
# a bounded window AFTER that specific entity -- never a bare clause-wide
# scan -- so it cannot accidentally attach an unrelated number (e.g. a
# duration count) to the wrong entity. Preference order: a number
# immediately paired with a recognized unit word/symbol (most reliable);
# otherwise a bare number with no unit at all (e.g. "sugar 98") is still
# preserved rather than silently dropped, since losing the value entirely
# was the original failure mode this addresses. No unit conversion is
# attempted -- the matched span is preserved verbatim.
# ---------------------------------------------------------------------------
def _measurement_phrase_for_entity(clause: str, entity: "_EntityMatch") -> tuple[str, str] | None:
    if entity.entity_type != "measurement":
        return None
    window_end = min(len(clause), entity.end + ENTITY_TRIGGER_WINDOW_CHARS)
    search_region = clause[entity.end:window_end]

    m = _MEASUREMENT_PATTERN.search(search_region)
    if m:
        return m.group(0), f"of {m.group(0)}"

    m_bare = _BARE_NUMBER_PATTERN.search(search_region)
    if m_bare:
        return m_bare.group(0), f"of {m_bare.group(0)}"

    return None


# ---------------------------------------------------------------------------
# Family / third-person attribution safety (Step 14H).
#
# Default is PATIENT (matches the pre-existing Step 14F behavior for every
# clause with no attribution marker at all, so existing regression cases
# are unaffected). A recognized family/third-person marker overrides that
# default. A clause naming BOTH a first-person marker AND a family marker
# (e.g. "Enakku appa-ku sugar irukku") is genuinely ambiguous for this
# rule-based system -- rather than guessing which reading is correct, it is
# reported UNKNOWN rather than defaulting to Patient (safety-first, per
# README). A specific relation marker (mother/father/brother/... ) is
# preferred over a generic marker (family/relatives/avanga/...) when both
# are present in the same clause, since the specific relation is more
# informative and was actually stated -- not fabricated.
# ---------------------------------------------------------------------------
def _determine_attribution(clause: str) -> tuple[str, str, bool]:
    """Returns (attribution_label, subject_text, subject_is_plural)."""
    has_first_person = bool(_FIRST_PERSON_MARKERS & set(_tokens(clause)))
    specific_match = _FAMILY_SPECIFIC_PATTERN.search(clause)
    plural_match = _FAMILY_PLURAL_PATTERN.search(clause)
    generic_match = _FAMILY_GENERIC_PATTERN.search(clause)
    has_family_marker = bool(specific_match or plural_match or generic_match)

    if has_first_person and has_family_marker:
        return "UNKNOWN", "Someone", False
    if specific_match:
        relation = lexicon.FAMILY_SPECIFIC_RELATION_MARKERS[specific_match.group(0).lower()]
        return "FAMILY_THIRD_PERSON", f"Patient's {relation}", False
    if plural_match:
        relation = lexicon.FAMILY_PLURAL_RELATION_MARKERS[plural_match.group(0).lower()]
        return "FAMILY_THIRD_PERSON", f"Patient's {relation}", True
    if generic_match:
        return "FAMILY_THIRD_PERSON", "Family members", True
    return "PATIENT", "Patient", False


def _attribution_marker_span(clause: str, attribution_label: str) -> str | None:
    """Best-effort source span for the attribution transformation trace."""
    if attribution_label == "UNKNOWN":
        first_person_match = _FIRST_PERSON_PATTERN.search(clause)
        family_match = (
            _FAMILY_SPECIFIC_PATTERN.search(clause)
            or _FAMILY_PLURAL_PATTERN.search(clause)
            or _FAMILY_GENERIC_PATTERN.search(clause)
        )
        spans = [m.group(0) for m in (first_person_match, family_match) if m]
        return " / ".join(spans) if spans else None
    if attribution_label == "FAMILY_THIRD_PERSON":
        match = (
            _FAMILY_SPECIFIC_PATTERN.search(clause)
            or _FAMILY_PLURAL_PATTERN.search(clause)
            or _FAMILY_GENERIC_PATTERN.search(clause)
        )
        return match.group(0) if match else None
    return None


# ---------------------------------------------------------------------------
# Verb-phrase composition
# ---------------------------------------------------------------------------
def _verb_phrase(entity_type: str, canonical_entity: str, category: str, is_plural: bool) -> str:
    is_medication = entity_type == "medication"
    if category in ("NEG_STATE", "NEG_ACTION"):
        if is_medication:
            verb = "do not take" if is_plural else "does not take"
        else:
            verb = "do not have" if is_plural else "does not have"
        return f"{verb} {canonical_entity}"
    if category == "HIST_VERB":
        # Past tense has no singular/plural distinction in English.
        verb = "took" if is_medication else "had"
        return f"{verb} {canonical_entity}"
    # CURRENT
    if is_medication:
        verb = "take" if is_plural else "takes"
    else:
        verb = "have" if is_plural else "has"
    return f"{verb} {canonical_entity}"


_RULE_NAME_BY_CATEGORY = {
    "NEG_STATE": "TANGLISH_NEGATION_STATE",
    "NEG_ACTION": "TANGLISH_NEGATION_ACTION",
    "HIST_VERB": "TANGLISH_HISTORICAL",
    "CURRENT": "TANGLISH_CURRENT_STATE",
}


def _compose_clause(
    clause: str, include_subject: bool, subject: str, is_plural: bool
) -> tuple[str, list[Transformation], SemanticFeatures, bool]:
    """Returns (composed_text, transformations, features, facts_found).

    `subject`/`is_plural` are determined by the caller (via
    _determine_attribution) BEFORE this function runs, since that
    determination is independent of entity/trigger extraction -- this keeps
    attribution detection and clinical-fact extraction as two separate,
    independently testable concerns.
    """
    features = SemanticFeatures()
    transformations: list[Transformation] = []

    entities = _find_entities(clause)
    for ent in entities:
        if ent.canonical not in features.clinical_entities:
            features.clinical_entities.append(ent.canonical)

    triggers = _find_triggers(clause)
    facts = _associate_entities_with_triggers(entities, triggers)

    if not facts:
        return clause.strip(), transformations, features, False

    duration = _duration_phrase(clause)
    frequency = _frequency_phrase(clause)
    historical_qualifier = _historical_qualifier(clause)

    phrases = []
    for fact in facts:
        phrase = _verb_phrase(fact.entity.entity_type, fact.entity.canonical, fact.trigger.category, is_plural)

        if fact.trigger.category == "HIST_VERB" and historical_qualifier:
            phrase = f"{phrase} {historical_qualifier}"

        if duration:
            phrase = f"{phrase} {duration[1]}"

        measurement = _measurement_phrase_for_entity(clause, fact.entity)
        if measurement:
            phrase = f"{phrase} {measurement[1]}"
            if measurement[1] not in features.measurements:
                features.measurements.append(measurement[1])
            transformations.append(
                Transformation(rule="TANGLISH_MEASUREMENT", input_span=measurement[0], normalized_span=measurement[1])
            )

        if frequency:
            phrase = f"{phrase} {frequency[1]}"

        phrases.append(phrase)

        rule = _RULE_NAME_BY_CATEGORY[fact.trigger.category]
        span_start = min(fact.entity.start, fact.trigger.start)
        span_end = max(fact.entity.end, fact.trigger.end)
        transformations.append(
            Transformation(rule=rule, input_span=clause[span_start:span_end], normalized_span=phrase)
        )

        if fact.trigger.category in ("NEG_STATE", "NEG_ACTION"):
            features.negations.append(fact.entity.canonical)
        elif fact.trigger.category == "HIST_VERB":
            features.historical_mentions.append(fact.entity.canonical)
        elif fact.trigger.category == "CURRENT":
            features.current_mentions.append(fact.entity.canonical)

    if duration:
        features.durations.append(duration[1])
        transformations.append(Transformation(rule="TANGLISH_DURATION", input_span=duration[0], normalized_span=duration[1]))
    if frequency:
        features.frequencies.append(frequency[1])
        transformations.append(Transformation(rule="TANGLISH_FREQUENCY", input_span=frequency[0], normalized_span=frequency[1]))

    joined = " and ".join(phrases)
    composed = f"{subject} {joined}" if include_subject else joined
    return composed, transformations, features, True


# ---------------------------------------------------------------------------
# Clause splitting (preserves the original conjunction word for rejoining)
# ---------------------------------------------------------------------------
_CONJUNCTION_SPLIT_RE = re.compile(r"\s+(but|and)\s+", re.IGNORECASE)


def _split_clauses(text: str) -> list[tuple[str, str | None]]:
    """Returns [(clause_text, conjunction_before_this_clause_or_None), ...]."""
    parts = _CONJUNCTION_SPLIT_RE.split(text)
    # re.split with a capturing group interleaves: [clause0, conj1, clause1, conj2, clause2, ...]
    clauses: list[tuple[str, str | None]] = [(parts[0], None)]
    i = 1
    while i < len(parts):
        conjunction = parts[i].lower()
        clause_text = parts[i + 1] if i + 1 < len(parts) else ""
        clauses.append((clause_text, conjunction))
        i += 2
    return clauses


def _merge_semantic_features(accumulated: SemanticFeatures, new: SemanticFeatures) -> None:
    for field_name in (
        "negations", "historical_mentions", "current_mentions", "durations", "frequencies",
        "measurements", "clinical_entities",
    ):
        getattr(accumulated, field_name).extend(getattr(new, field_name))


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def normalize(text: str) -> NormalizedTextResult:
    """Normalize one premise/utterance. Deterministic and side-effect-free.

    English-only input is returned unchanged (bypass). Unknown/empty input
    is returned unchanged with detected_language=UNKNOWN. Tanglish/mixed
    input is normalized clause-by-clause; unrecognized clauses pass through
    unchanged rather than being guessed at.
    """
    language = detect_language(text)

    if language in (LANGUAGE_ENGLISH, LANGUAGE_UNKNOWN):
        return NormalizedTextResult(
            original_text=text,
            normalized_text=text,
            detected_language=language,
            is_code_switched=False,
        )

    clauses = _split_clauses(text.strip().rstrip("."))
    composed_parts: list[str] = []
    all_transformations: list[Transformation] = []
    aggregated_features = SemanticFeatures()
    previous_subject: str | None = None

    for index, (clause_text, conjunction) in enumerate(clauses):
        # Attribution is determined BEFORE fact extraction (it only depends
        # on clause text, not on which entities/triggers fire) so subject
        # continuity across "but"/"and" can be decided up front: the first
        # clause always states its subject; a later clause only repeats the
        # subject if attribution actually changed (e.g. patient -> family),
        # matching the pre-existing single-subject behavior whenever every
        # clause's attribution is the same (the common case).
        attribution_label, subject, is_plural = _determine_attribution(clause_text)
        include_subject = (index == 0) or (subject != previous_subject)

        composed, transformations, features, facts_found = _compose_clause(
            clause_text, include_subject, subject, is_plural
        )
        composed_parts.append(composed)
        all_transformations.extend(transformations)
        _merge_semantic_features(aggregated_features, features)

        if facts_found:
            previous_subject = subject
            if attribution_label != "PATIENT":
                aggregated_features.attributions.append(attribution_label)
                marker_span = _attribution_marker_span(clause_text, attribution_label)
                if marker_span:
                    all_transformations.append(
                        Transformation(
                            rule=f"TANGLISH_ATTRIBUTION_{attribution_label}",
                            input_span=marker_span,
                            normalized_span=subject,
                        )
                    )

        if index > 0 and conjunction:
            composed_parts[-1] = composed_parts[-1]  # placeholder for readability; join handled below

    # Rebuild final sentence with original conjunctions preserved.
    sentence = composed_parts[0]
    for (clause_text, conjunction), composed in zip(clauses[1:], composed_parts[1:]):
        sentence = f"{sentence} {conjunction} {composed}"
    sentence = sentence.strip()
    if sentence and not sentence.endswith("."):
        sentence += "."

    if _FIRST_PERSON_MARKERS & set(_tokens(text)):
        aggregated_features.attributions.append("PATIENT_SELF_REPORT")

    aggregated_features.clinical_entities = list(dict.fromkeys(aggregated_features.clinical_entities))

    is_code_switched = language == LANGUAGE_MIXED or (
        language == LANGUAGE_TANGLISH and _has_clinical_english(text)
    )

    return NormalizedTextResult(
        original_text=text,
        normalized_text=sentence,
        detected_language=language,
        is_code_switched=is_code_switched,
        transformations=all_transformations,
        semantic_features=aggregated_features,
    )
