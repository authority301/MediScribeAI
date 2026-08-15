"""Prototype Tanglish clinical vocabulary and pattern lexicon (Step 14F).

============================================================================
LIMITATIONS -- READ BEFORE EXTENDING
============================================================================
This is NOT an exhaustive Tamil-English dictionary and does NOT implement
general Tamil-English machine translation. It covers a defined, documented
subset of common Romanized-Tamil clinical conversational patterns similar to
those used in the Step 14E gold benchmark. Tanglish has no universal
spelling standard -- Tamil-English code-switching varies by speaker, region,
and register. Entries here are common variants, not a complete
enumeration. See backend/app/tanglish/README.md for the full research-
integrity statement.
"""

# ---------------------------------------------------------------------------
# Clinical entity vocabulary.
# term (lowercase, as it would appear in transliterated text) -> (canonical
# English surface form to preserve in normalized output, entity_type)
# entity_type in {"symptom", "medication", "measurement", "procedure", "diagnosis"}
#
# Clinical English terms are intentionally listed here so the normalizer can
# RECOGNIZE them (for negation/state attachment) -- they are never
# translated or altered, per the "do not unnecessarily rewrite clinical
# English" requirement.
# ---------------------------------------------------------------------------
CLINICAL_TERMS: dict[str, tuple[str, str]] = {
    # symptoms
    "fever": ("fever", "symptom"),
    "cough": ("cough", "symptom"),
    "headache": ("headache", "symptom"),
    "chest pain": ("chest pain", "symptom"),
    "stomach pain": ("stomach pain", "symptom"),
    "vayitru vali": ("stomach pain", "symptom"),
    "thalai vali": ("headache", "symptom"),
    "back pain": ("back pain", "symptom"),
    "joint pain": ("joint pain", "symptom"),
    "sore throat": ("sore throat", "symptom"),
    "breathing difficult": ("difficulty breathing", "symptom"),
    "rash": ("rash", "symptom"),
    "itching": ("itching", "symptom"),
    # diagnoses
    "diabetes": ("diabetes", "diagnosis"),
    "sugar problem": ("diabetes", "diagnosis"),
    "asthma": ("asthma", "diagnosis"),
    "bronchitis": ("bronchitis", "diagnosis"),
    "migraine": ("migraine", "diagnosis"),
    "pneumonia": ("pneumonia", "diagnosis"),
    "tb": ("TB", "diagnosis"),
    # medications
    "paracetamol": ("paracetamol", "medication"),
    "aspirin": ("aspirin", "medication"),
    "insulin": ("insulin", "medication"),
    "antibiotics": ("antibiotics", "medication"),
    "medicine": ("medicine", "medication"),
    "tablet": ("tablet", "medication"),
    "eye drops": ("eye drops", "medication"),
    # measurements
    "blood pressure": ("blood pressure", "measurement"),
    "temperature": ("temperature", "measurement"),
    "oxygen saturation": ("oxygen saturation", "measurement"),
    "oxygen level": ("oxygen saturation", "measurement"),
    "heart rate": ("heart rate", "measurement"),
    "weight": ("weight", "measurement"),
    # procedures
    "ecg": ("ECG", "procedure"),
    "x-ray": ("X-ray", "procedure"),
    "blood test": ("blood test", "procedure"),
    "scan": ("scan", "procedure"),
    "lab work": ("lab work", "procedure"),
    "operation": ("operation", "procedure"),
}

# ---------------------------------------------------------------------------
# Negation triggers. These require an adjacent clinical entity to fire --
# the normalizer never negates based on a bare substring match (e.g. it
# never treats every occurrence of "la" as negation).
#
# STATE negation ("does not have <entity>"): entity typically precedes the
# trigger describing presence/absence of a symptom/diagnosis/measurement.
# Ordered longest-phrase-first so alternation prefers the more specific form.
# ---------------------------------------------------------------------------
NEGATION_TRIGGERS_STATE = [
    "illainga",
    "illai nu",
    "illanu",
    "illai",
    "kidayathu",
    "kedayathu",
    "ille",
    "illa",
]

# ACTION negation ("does not take/do <entity>"): typically follows an object
# such as a medication.
NEGATION_TRIGGERS_ACTION = [
    "edukkala",
    "edukala",
    "saapdala",
    "varala",
    "pogala",
    "pannala",
    "irukkala",
]

# ---------------------------------------------------------------------------
# Current-state triggers ("has <entity>" / positive, present tense).
# ---------------------------------------------------------------------------
CURRENT_STATE_TRIGGERS = [
    "irukku nu",
    "irukirathu",
    "irukuthu",
    "irukken",
    "irukku",
]

# ---------------------------------------------------------------------------
# Historical markers.
# HISTORICAL_VERB_TRIGGERS require an adjacent entity ("asthma irundhudhu").
# HISTORICAL_TIME_MARKERS are phrase-level flags (no entity attachment) that
# qualify HOW the historical fact should be phrased (e.g. "during
# childhood") when a historical verb trigger has already fired in the same
# clause.
# ---------------------------------------------------------------------------
HISTORICAL_VERB_TRIGGERS = [
    "irundhudhu",
    "irundhuchu",
    "irunthuchu",
]

HISTORICAL_TIME_MARKERS_CHILDHOOD = [
    "chinna vayasula",
    "school days-la",
    "school days la",
    "childhood-la",
    "childhood la",
]

HISTORICAL_TIME_MARKERS_PAST = [
    "pala varusham munnadi",
    "years back",
    "munnaadi",
    "munnadi",
]

# ---------------------------------------------------------------------------
# Duration. TAMIL_NUMERALS covers a small prototype subset (1-10); larger or
# unusual numeral phrases are not converted and pass through unrecognized
# (documented limitation).
# ---------------------------------------------------------------------------
TAMIL_NUMERALS: dict[str, int] = {
    "oru": 1,
    "rendu": 2,
    "randu": 2,
    "moonu": 3,
    "munu": 3,
    "naalu": 4,
    "anju": 5,
    "ainthu": 5,
    "aaru": 6,
    "ezhu": 7,
    "ettu": 8,
    "onbadhu": 9,
    "pathu": 10,
}

ENGLISH_NUMBER_WORDS: dict[int, str] = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}

# Duration unit words that indicate "day" or "week" when preceded by a
# recognized numeral (Tamil or digit).
DURATION_UNIT_WORDS: dict[str, str] = {
    "naala": "day",
    "naal": "day",
    "vaarathula": "week",
    "vaaram": "week",
    "varam": "week",
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
}

# ---------------------------------------------------------------------------
# Frequency.
# ---------------------------------------------------------------------------
FREQUENCY_UNIT_WORDS: dict[str, str] = {
    "thadava": "time",
    "velai": "time",
    "time": "time",
    "times": "time",
}

TIME_OF_DAY_MARKERS: dict[str, str] = {
    "morning-la": "in the morning",
    "morning la": "in the morning",
    "night-la": "at night",
    "night la": "at night",
}

# ---------------------------------------------------------------------------
# Language detection vocabulary.
# ---------------------------------------------------------------------------
# All single-word Tamil markers used by the rules above, plus a handful of
# additional common Tanglish function words that signal Tamil grammatical
# structure without necessarily triggering a specific normalization rule
# (used only for language detection, not for fact extraction).
TAMIL_SINGLE_WORD_MARKERS = set(
    NEGATION_TRIGGERS_STATE
    + NEGATION_TRIGGERS_ACTION
    + CURRENT_STATE_TRIGGERS
    + HISTORICAL_VERB_TRIGGERS
    + list(TAMIL_NUMERALS.keys())
    + [
        "enakku", "naan", "en", "ennaku", "ennoda", "ungalukku", "unga",
        "ungal", "avanga", "avaru", "romba", "konjam", "sonna", "nu",
        "irundha", "iruken", "vandhudhu", "vandhachu",
    ]
)

TAMIL_PHRASE_MARKERS = (
    HISTORICAL_TIME_MARKERS_CHILDHOOD + HISTORICAL_TIME_MARKERS_PAST + ["illai nu", "irukku nu"]
)

# Words indicating full English clause/sentence structure (as opposed to a
# bare clinical noun, which can appear inside Tanglish text too). Used only
# to distinguish MIXED (Tamil markers + full English structure) from pure
# TANGLISH (Tamil markers + bare English clinical nouns only).
ENGLISH_STRUCTURE_WORDS = {
    "is", "are", "was", "were", "has", "have", "had", "does", "do", "did",
    "not", "no", "patient", "reports", "report", "since", "for", "during",
    "currently", "presently", "but", "and", "the", "a", "an", "was", "with",
}
