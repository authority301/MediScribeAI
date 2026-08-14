"""Deterministic temporal-overlap alignment between ASR segments and diarization turns.

Pure interval arithmetic only -- no LLM, no semantic guessing. A segment is
assigned to the diarization turn with the strongest temporal overlap; if no
turn overlaps it at all, no speaker is assigned (never fabricated).
"""
from app.diarization.model import DiarizationTurn


def _overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def best_matching_turn_index(
    segment_start_ms: int, segment_end_ms: int, turns: list[DiarizationTurn]
) -> tuple[int | None, str | None, float | None]:
    """Return (turn_index, speaker_label, overlap_ratio) for the diarization
    turn with the strongest temporal overlap against
    [segment_start_ms, segment_end_ms].

    Returns (None, None, None) if no turn overlaps at all -- the caller must
    not assign a speaker in that case. overlap_ratio is overlap duration
    divided by the ASR segment's own duration, in (0, 1] -- a simple,
    explainable confidence measure, not a model-reported probability.
    """
    segment_duration = segment_end_ms - segment_start_ms
    if segment_duration <= 0 or not turns:
        return None, None, None

    best_index = None
    best_overlap = 0
    for index, turn in enumerate(turns):
        overlap = _overlap_ms(segment_start_ms, segment_end_ms, turn.start_ms, turn.end_ms)
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index

    if best_index is None:
        return None, None, None

    return best_index, turns[best_index].speaker_label, best_overlap / segment_duration
