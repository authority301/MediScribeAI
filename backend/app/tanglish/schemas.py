"""Structured representation types for Tanglish normalization (Step 14F).

Plain dataclasses (matching the style of app/nli/aggregation.py's
EvidenceNliResult) -- no pydantic/FastAPI dependency needed since this
module exposes no HTTP endpoint (see backend/app/tanglish/README.md for why).
"""
from dataclasses import dataclass, field


@dataclass
class Transformation:
    """One normalization rule firing, recorded for research explainability
    and debugging. Never includes more than the input/output text spans
    already present in the (synthetic, or locally-processed) input text --
    no additional patient data is attached."""

    rule: str
    input_span: str
    normalized_span: str


@dataclass
class SemanticFeatures:
    negations: list[str] = field(default_factory=list)
    historical_mentions: list[str] = field(default_factory=list)
    current_mentions: list[str] = field(default_factory=list)
    durations: list[str] = field(default_factory=list)
    frequencies: list[str] = field(default_factory=list)
    measurements: list[str] = field(default_factory=list)
    clinical_entities: list[str] = field(default_factory=list)
    # PATIENT_SELF_REPORT (whole-text first-person marker present) plus, per
    # clause, FAMILY_THIRD_PERSON / UNKNOWN when that clause's subject is not
    # the patient (Step 14H attribution safety -- see README).
    attributions: list[str] = field(default_factory=list)


@dataclass
class NormalizedTextResult:
    original_text: str
    normalized_text: str
    detected_language: str  # ENGLISH | TANGLISH | MIXED | UNKNOWN
    is_code_switched: bool
    transformations: list[Transformation] = field(default_factory=list)
    semantic_features: SemanticFeatures = field(default_factory=SemanticFeatures)
