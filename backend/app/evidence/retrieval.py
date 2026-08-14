"""Deterministic lexical evidence retrieval baseline.

============================================================================
IMPORTANT RESEARCH PRINCIPLE
============================================================================
retrieval_score measures lexical RELEVANCE between a SOAP claim and a
candidate transcript segment. It is NOT a judgement of clinical truth, NOT a
confidence that the claim is correct, and NOT a hallucination/contradiction
classification.

Example: a negated segment ("I don't have fever.") can score highly relevant
to a claim ("Patient reports fever.") purely because "fever" matches
lexically -- retrieving it is CORRECT behavior for this step. Whether that
evidence actually supports, contradicts, or fails to ground the claim is
Step 14B's job (NLI verification), not this module's. This distinction is
deliberate and must not be collapsed here.
============================================================================

No embeddings, no vector database, no sentence-transformers, no LLM -- pure
Python token-overlap scoring, fully deterministic and explainable.
"""
from dataclasses import dataclass

from app.evidence.normalization import content_tokens

# Role preference per SOAP section: a RANKING SIGNAL, never a hard filter.
# A doctor may repeat a patient's symptom and a patient may repeat a doctor's
# instruction, so every role stays eligible; matching the section's
# preferred role only nudges the score up or down.
SECTION_PREFERRED_ROLE: dict[str, str | None] = {
    "SUBJECTIVE": "PATIENT",
    "ASSESSMENT": "DOCTOR",
    "PLAN": "DOCTOR",
    "OBJECTIVE": None,  # no preference -- measurements may be stated by either party
}
ROLE_MATCH_MULTIPLIER = 1.0
ROLE_UNKNOWN_MULTIPLIER = 0.95  # UNKNOWN/no role info stays eligible, near-neutral
ROLE_MISMATCH_MULTIPLIER = 0.85  # confirmed wrong role: a penalty, never exclusion
ROLE_NO_PREFERENCE_MULTIPLIER = 1.0

# Weighting of the three lexical similarity components (sum to 1.0), applied
# to the CLAIM-side token/bigram sets so every ratio stays within [0, 1].
WEIGHTED_OVERLAP_WEIGHT = 0.5
TOKEN_OVERLAP_WEIGHT = 0.3
BIGRAM_OVERLAP_WEIGHT = 0.2


@dataclass
class EvidenceCandidate:
    speaker_segment_id: object
    retrieval_score: float
    rank: int


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _token_weights(claim_tokens: list[str], segment_token_lists: list[list[str]]) -> dict[str, float]:
    """Lightweight, dependency-free IDF-like weighting.

    A claim token that appears in FEWER of this claim's own candidate
    segments is more distinctive and gets a higher weight. Computed only
    across this single claim's candidate set (deterministic; no external
    corpus, no model, no scikit-learn dependency).
    """
    num_segments = max(len(segment_token_lists), 1)
    raw_weights: dict[str, float] = {}
    for token in set(claim_tokens):
        doc_freq = sum(1 for tokens in segment_token_lists if token in tokens)
        raw_weights[token] = 1.0 if doc_freq == 0 else num_segments / doc_freq
    max_weight = max(raw_weights.values(), default=1.0)
    return {token: weight / max_weight for token, weight in raw_weights.items()}


def _lexical_similarity(
    claim_tokens: list[str], segment_tokens: list[str], token_weights: dict[str, float]
) -> float:
    """Combine token overlap, IDF-weighted token overlap, and bigram (near-
    exact phrase) overlap into a single [0, 1] relevance score. All three
    component ratios are computed relative to the CLAIM side, so each stays
    within [0, 1] and the weighted sum (weights summing to 1.0) does too.
    """
    if not claim_tokens:
        return 0.0

    claim_set = set(claim_tokens)
    segment_set = set(segment_tokens)
    overlap = claim_set & segment_set

    token_overlap_ratio = len(overlap) / len(claim_set)

    claim_weight_total = sum(token_weights.get(t, 0.0) for t in claim_set) or 1.0
    overlap_weight_total = sum(token_weights.get(t, 0.0) for t in overlap)
    weighted_overlap_ratio = overlap_weight_total / claim_weight_total

    claim_bigrams = _bigrams(claim_tokens)
    segment_bigrams = _bigrams(segment_tokens)
    bigram_overlap_ratio = (
        len(claim_bigrams & segment_bigrams) / len(claim_bigrams) if claim_bigrams else 0.0
    )

    score = (
        WEIGHTED_OVERLAP_WEIGHT * weighted_overlap_ratio
        + TOKEN_OVERLAP_WEIGHT * token_overlap_ratio
        + BIGRAM_OVERLAP_WEIGHT * bigram_overlap_ratio
    )
    return max(0.0, min(1.0, score))


def _role_adjustment(section: str, segment_role: str | None) -> float:
    preferred = SECTION_PREFERRED_ROLE.get(section)
    if preferred is None:
        return ROLE_NO_PREFERENCE_MULTIPLIER
    if segment_role is None or segment_role == "UNKNOWN":
        return ROLE_UNKNOWN_MULTIPLIER
    if segment_role == preferred:
        return ROLE_MATCH_MULTIPLIER
    return ROLE_MISMATCH_MULTIPLIER


def retrieve_evidence(
    claim_section: str,
    claim_text: str,
    segments: list,
    top_k: int,
    min_score: float,
) -> list[EvidenceCandidate]:
    """Deterministic lexical retrieval for one SOAP claim.

    Ranks `segments` (SpeakerSegment rows) by relevance to claim_text,
    applies a role-based ranking adjustment (never a hard filter -- every
    segment, including UNKNOWN-role or negated-text ones, is scored), and
    returns up to top_k candidates whose final score is >= min_score.
    Returns an empty list if nothing clears the threshold; evidence is never
    fabricated to fill quota.
    """
    claim_tokens = content_tokens(claim_text)
    if not claim_tokens:
        return []

    eligible_segments = [s for s in segments if s.segment_text]
    segment_token_lists = [content_tokens(s.segment_text) for s in eligible_segments]
    token_weights = _token_weights(claim_tokens, segment_token_lists)

    scored: list[tuple] = []
    for segment, segment_tokens in zip(eligible_segments, segment_token_lists):
        lexical = _lexical_similarity(claim_tokens, segment_tokens, token_weights)
        adjustment = _role_adjustment(claim_section, segment.inferred_role)
        score = max(0.0, min(1.0, lexical * adjustment))
        scored.append((segment, score))

    # Stable sort: ties keep their original (sequence_index) order.
    scored.sort(key=lambda pair: pair[1], reverse=True)

    candidates: list[EvidenceCandidate] = []
    for rank, (segment, score) in enumerate(scored, start=1):
        if score < min_score:
            break  # sorted descending -- everything after this also fails
        candidates.append(
            EvidenceCandidate(speaker_segment_id=segment.id, retrieval_score=score, rank=rank)
        )
        if len(candidates) >= top_k:
            break

    return candidates
