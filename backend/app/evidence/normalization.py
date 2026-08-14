"""Deterministic, conservative claim/segment text normalization for lexical retrieval.

No LLM, no external NLP API, no stemming -- medical terms must survive intact
("diabetes" must never become "diabet", "paracetamol" must stay whole).
Numbers with decimal points ("101.2") and simple attached units are kept as
single tokens rather than being split or mangled by punctuation stripping.
"""
import re
import unicodedata

# Order matters: decimal numbers and unit-attached numbers are matched before
# the generic \w+ fallback, so "101.2" and "101.2°f" survive as one token
# each instead of being split apart by punctuation handling.
_TOKEN_RE = re.compile(r"\d+\.\d+|\d+°[cf]?|\d+%|\w+")

# Generic function words only. NEVER removes numbers, units, or short
# clinical tokens purely for being short (e.g. "no", "bp" are NOT stopwords
# -- clinical phrases can hinge on exactly such short tokens).
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "and", "or", "but", "for", "with",
    "this", "that", "it", "as", "i", "you", "he", "she", "they", "we",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "your", "my", "his", "her", "their", "our",
}


def normalize_text(text: str) -> str:
    """Unicode-normalize (NFKC), lowercase, and collapse whitespace.

    Conservative: does not strip punctuation and does not stem words.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(text: str) -> list[str]:
    """Deterministic tokenization: Unicode-normalize + lowercase, then extract
    word/number tokens. Decimal numbers stay intact as single tokens instead
    of being split by a period-stripping step.
    """
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", text).lower()
    return _TOKEN_RE.findall(normalized)


def content_tokens(text: str) -> list[str]:
    """Tokens with generic stopwords removed. Medically meaningful short
    tokens (numbers, units, negation words like "no") are never removed.
    """
    return [t for t in tokenize(text) if t not in STOPWORDS]
