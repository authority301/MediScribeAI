"""Lazily-loaded, reusable local multilingual NLI model.

Runs entirely on this machine via Hugging Face Transformers + local PyTorch
inference. No cloud API, no LLM API. Model weights are fetched once from
Hugging Face Hub (a public, ungated model) on first use; this module never
loads the model at import time, mirroring the ASR (app/asr/model.py) and
diarization (app/diarization/model.py) lazy-singleton pattern.

============================================================================
INPUT ORDER (never reversed)
============================================================================
premise    = the retrieved transcript speaker-segment text (Step 14A evidence)
hypothesis = the SOAP claim text

============================================================================
RESEARCH INTEGRITY
============================================================================
NLI prediction != medical truth. See app/nli/aggregation.py's module
docstring for the full explanation of SUPPORTED/CONTRADICTED/UNGROUNDED.
"""
import threading

from app.nli.config import NLI_DEVICE, NLI_MODEL_NAME

_lock = threading.Lock()
_tokenizer = None
_model = None

_LABEL_MAP = {
    "entailment": "ENTAILMENT",
    "neutral": "NEUTRAL",
    "contradiction": "CONTRADICTION",
}


class NliScores:
    __slots__ = ("entailment", "neutral", "contradiction", "label")

    def __init__(self, entailment: float, neutral: float, contradiction: float, label: str):
        self.entailment = entailment
        self.neutral = neutral
        self.contradiction = contradiction
        self.label = label


def get_model():
    """Return the process-wide (tokenizer, model) pair, loading on first use only."""
    global _tokenizer, _model
    if _model is None:
        with _lock:
            if _model is None:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
                model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
                model.to(torch.device(NLI_DEVICE))
                model.eval()
                _tokenizer = tokenizer
                _model = model
    return _tokenizer, _model


def run_nli(premise: str, hypothesis: str) -> NliScores:
    """Run local NLI on (premise, hypothesis). premise is ALWAYS the
    retrieved transcript evidence segment; hypothesis is ALWAYS the SOAP
    claim. This order is never reversed.

    Label order is read from the model's own config.id2label rather than
    assumed by position, since checkpoints can differ.
    """
    import torch

    tokenizer, model = get_model()
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[0].tolist()

    scores_by_label: dict[str, float] = {}
    for index, probability in enumerate(probabilities):
        raw_label = model.config.id2label[index].lower()
        scores_by_label[raw_label] = probability

    entailment = scores_by_label.get("entailment", 0.0)
    neutral = scores_by_label.get("neutral", 0.0)
    contradiction = scores_by_label.get("contradiction", 0.0)

    winning_raw_label = max(scores_by_label, key=scores_by_label.get)

    return NliScores(
        entailment=entailment,
        neutral=neutral,
        contradiction=contradiction,
        label=_LABEL_MAP[winning_raw_label],
    )
