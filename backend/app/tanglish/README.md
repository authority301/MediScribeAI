# Tanglish-Aware Clinical NLI Preprocessing (Step 14F)

A deterministic, rule-based preprocessing layer that attempts to make
Tamil-English ("Tanglish") code-switched clinical text more explicit before
it reaches the existing Step 14B NLI verifier. **Step 14B itself is
unmodified and remains the frozen research baseline.**

## Motivation

Step 14C's baseline evaluation of the unmodified Step 14B pipeline against
the Step 14E gold benchmark showed a substantial English → Tanglish
performance gap:

| | Accuracy | Macro F1 |
|---|---|---|
| English | 76.92% | 0.7726 |
| Tanglish | 43.24% | 0.4099 |
| Mixed | 56.00% | 0.5343 |

Failure analysis pointed at negation and code-switching as the dominant
failure themes. This step implements a **hypothesis**: that rewriting
Tanglish clinical utterances into explicit English sentences — while
preserving negation, temporal framing, and clinical entities — may improve
how well the *unmodified* NLI model can judge entailment/contradiction
against SOAP claims.

**This hypothesis is not tested here.** Step 14F only implements the
preprocessing layer. The controlled comparison (does it actually help?) is
Step 14G's job.

## What this is NOT

- **Not a Tamil → English machine translator.** No general natural-language
  translation is attempted.
- **Not a new NLI model.** The existing `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
  model, its thresholds (0.70/0.70), and its aggregation rule are reused
  unchanged from `app/nli/`.
- **Not a general Tamil NLP system.** Coverage is a defined, documented
  subset of patterns relevant to short clinical utterances.
- **Not wired into production.** Nothing in `app/tanglish/` is imported by
  `app/main.py` or the `/consultations/.../soap/evidence/verify` endpoint.
  Tanglish normalization is never silently applied to real consultations.

## Architecture

```
Tanglish / mixed clinical text
        |
detect_language()              -- ENGLISH | TANGLISH | MIXED | UNKNOWN
        |
        v
[English or empty/unknown?] --yes--> return unchanged (bypass)
        | no
        v
split into clauses on "but"/"and", preserving the conjunction
        |
        v
per clause:
  find recognized clinical entities (fever, aspirin, chest pain, ...)
  find negation / current-state / historical triggers
  associate each trigger with the nearest preceding entity
    (within a bounded character window -- never a bare substring match)
  extract duration / frequency phrases independently
  compose an explicit English clause: "Patient {verb phrase} [qualifiers]"
  unrecognized clauses pass through UNCHANGED
        |
        v
rejoin clauses with the original conjunction
        |
        v
NormalizedTextResult (original_text, normalized_text, detected_language,
                       is_code_switched, transformations, semantic_features)
        |
        v
app.tanglish.service.verify(mode="tanglish_aware")
        |
        v
app.nli.model.run_nli()            <- SAME function as production
        |
        v
app.nli.aggregation.aggregate_claim_verification()   <- SAME function, SAME thresholds
```

`app/tanglish/service.py` exposes two explicit modes:

- `verify(premise, hypothesis, mode="baseline")` — premise goes to
  `run_nli` unchanged (identical behavior to Step 14B).
- `verify(premise, hypothesis, mode="tanglish_aware")` — premise is
  normalized first, then the normalized text goes to the same `run_nli`.

Both modes call the exact same `app.nli.model.run_nli` and
`app.nli.aggregation.aggregate_claim_verification` functions — no NLI logic
is duplicated.

## Why no HTTP endpoint

The spec allowed for `POST /evaluation/tanglish-normalize` /
`POST /evaluation/tanglish-verify` "if necessary," but was explicit that an
unnecessary production-facing API should be avoided when a service-level
interface is sufficient. Step 14G (the controlled comparison) invokes both
modes **programmatically** — the same pattern already used by Step 14C's
`app/evaluation/` module, which also has no HTTP routes. Adding routes here
would mean new auth/ownership/precondition surface area for no actual
consumer, so this step ships `app/tanglish/service.py` as a plain importable
module instead. `main.py` and all existing routes are unchanged.

## Supported semantic patterns (prototype subset)

- **Negation** (highest priority, per the Step 14C failure analysis):
  `illa`, `illai`, `illainga`, `ille`, `illanu`, `illai nu`, `kidayathu`,
  `kedayathu` (state negation — "does not have X"); `edukkala`, `edukala`,
  `saapdala`, `varala`, `pogala`, `pannala`, `irukkala` (action negation —
  "does not take X"). Every trigger is matched as a whole word/phrase with
  regex word boundaries and only fires when a recognized clinical entity is
  found nearby — **never** a bare substring check (e.g. "vanilla" is not
  treated as containing "illa").
- **Current state**: `irukku`, `irukku nu`, `irukirathu`, `irukuthu`,
  `irukken` → "has/takes X".
- **Historical state**: `irundhudhu`, `irundhuchu`, `irunthuchu` → "had/took
  X"; qualified with "during childhood" when `chinna vayasula` / `school
  days-la` / `childhood-la` is also present, or "in the past" when
  `munnaadi` / `munnadi` / `pala varusham munnadi` is present.
- **Duration**: Tamil numerals 1–10 (`oru`...`pathu`) or digits + `naala` /
  `naal` / `vaaram` / `vaarathula` / `days` / `weeks` → "for two days" etc.
- **Frequency**: Tamil numerals + `thadava` / `velai` → "once" / "twice" /
  "N times". A small set of time-of-day markers (`morning-la`, `night-la`)
  is also recognized.
- **Clinical entities**: a fixed vocabulary of common symptoms,
  medications, measurements, procedures, and diagnoses (see
  `lexicon.py::CLINICAL_TERMS`) — these are recognized, never translated.
- **Code-switching / composition**: a single sentence can contain multiple
  clauses joined by "but"/"and"; each clause is normalized independently
  and the original conjunction is preserved, so multiple propositions (one
  positive, one negated) are never collapsed into one.
- **Attribution (light)**: first-person markers (`enakku`, `naan`, `en`,
  ...) are recorded in `semantic_features.attributions` as
  `"PATIENT_SELF_REPORT"` when present. This is descriptive metadata only —
  the normalized sentence subject is always "Patient", matching this
  project's existing SOAP-claim convention.

## Example transformations

| Input | Normalized output |
|---|---|
| `Enakku fever irukku.` | `Patient has fever.` |
| `Enakku fever illa.` | `Patient does not have fever.` |
| `Enakku chest pain illa.` | `Patient does not have chest pain.` |
| `Enakku fever rendu naala irukku.` | `Patient has fever for two days.` |
| `Chinna vayasula asthma irundhudhu.` | `Patient had asthma during childhood.` |
| `Naan paracetamol edukkala.` | `Patient does not take paracetamol.` |
| `Enakku fever irukku but chest pain illa.` | `Patient has fever but does not have chest pain.` |
| `Patient reports fever for two days.` (English) | unchanged |

## Transformation trace

Every fired rule is recorded as a `Transformation(rule, input_span,
normalized_span)`, e.g. `{"rule": "TANGLISH_NEGATION_STATE", "input_span":
"chest pain illa", "normalized_span": "does not have chest pain"}`. This
exists for research explainability/debugging. Traces only ever contain
spans of the input text itself (synthetic in the benchmark; whatever text
the caller passes in production) — no additional data is attached or
logged.

## Limitations (read before relying on this for anything beyond the prototype)

- Tanglish has no universal spelling standard; Tamil-English code-switching
  varies by speaker, region, and register. The vocabulary here covers
  common variants observed in the Step 14E benchmark and similar
  synthetic examples — it is **not exhaustive**.
- This is not a general Tamil NLP system and makes no claim of complete
  Tamil or Tanglish understanding.
- Verb-form coverage is limited (e.g. only the `irukku`-family of current-
  state verbs and a fixed list of negation verbs are recognized); other
  conjugations pass through unchanged rather than being guessed at.
- Context-dependent interpretation can be genuinely ambiguous; this system
  does not attempt to resolve ambiguity — it either applies a matched rule
  or leaves text unchanged.
- Language detection (`detect_language`) is a lightweight, deterministic
  heuristic, not a linguistically rigorous classifier. Text with no
  recognized Tamil marker is treated as `ENGLISH` by default (a
  conservative bypass), which means genuinely unrecognizable non-English,
  non-Tanglish text could also be classified `ENGLISH`; only empty/
  whitespace input is classified `UNKNOWN`.
- No clinical validity or medical accuracy is claimed for the normalized
  output — it is a text-level semantic restatement, not a clinical
  judgment.
- This is described as a **proposed / prototype Tanglish-aware clinical
  semantic normalization approach**. No claim of novelty is made here;
  novelty (if any) would need to be established through literature review
  and comparative experiments in a later step.

## Known failure modes identified by Step 14G (documented, not yet fixed)

Step 14G (`backend/app/evaluation_tanglish/`) ran the real 101-case Step 14E
benchmark through both the baseline and Tanglish-aware systems and analyzed
every case where Tanglish-aware preprocessing changed a correct baseline
prediction into an incorrect one ("regressed" cases -- see
`datasets/nli_evaluation/results/tanglish_comparison/failure_analysis.json`).
That analysis surfaced two **generic** normalization gaps, documented here
for future work. Neither has been fixed yet -- fixing either would change
normalizer output and require rerunning Step 14G, which is out of scope for
this documentation update. These are limitations of the pattern coverage
described above, not new claims about the system's behavior.

1. **Numeral / measurement loss.** Numerical expressions written in English
   form inside an otherwise Tanglish/mixed clause -- e.g. `"for three
   days"`, `"98 percent"`, `"88 beats per minute"` -- are not captured by
   the duration pattern (`_DURATION_PATTERN`, which only matches Tamil
   numeral words or bare digits immediately followed by a recognized unit
   word like `naala`/`vaaram`) or by any measurement-value pattern (no such
   pattern exists yet -- measurement *entities* are recognized via
   `lexicon.CLINICAL_TERMS`, but their numeric *values* are not extracted or
   preserved). The composed sentence keeps the recognized entity but drops
   the value/duration, e.g. producing `"Patient has cough."` instead of
   `"Patient has cough for three days."`, or `"Patient has heart rate."`
   instead of a sentence that preserves `"88 beats per minute"`. This can
   weaken NLI evidence matching whenever the SOAP claim being verified
   includes that dropped numerical/duration detail, since the normalized
   premise then supports less of the claim than the original text actually
   stated. Observed in Step 14G's regressed cases MX-001, MX-006, MX-008,
   and TL-011 (regression: previously-correct SUPPORTED/CONTRADICTED
   predictions became UNGROUNDED once the value was dropped).

2. **Family-history / third-person misattribution.** The normalizer has no
   family-member or third-person-subject detection. A clause reporting that
   *someone other than the patient* has a condition -- e.g. `"Family-la sila
   peruku diabetes irukku."` ("some people in the family have diabetes") --
   still matches the `CURRENT` trigger (`irukku`) adjacent to the
   `diabetes` entity and gets composed with the fixed subject `"Patient"`,
   producing `"Patient has diabetes."` This silently converts a
   family-history statement into a patient-level claim, which can turn a
   correct `UNGROUNDED` prediction (family history is not evidence for a
   claim about the patient's own diagnosis) into an incorrect `SUPPORTED`
   one. Observed in Step 14G's regressed case MX-016. Root cause: no
   family/third-person marker vocabulary or attribution logic exists
   alongside the existing first-person (`_FIRST_PERSON_MARKERS`) detection.

These are reported as **generic** linguistic gaps in the rule coverage, not
as issues with any specific benchmark case -- the case IDs above are cited
only as the observed instances that surfaced each gap during evaluation, not
as targets for case-specific patches. Per this project's research-integrity
practice, no benchmark-specific exception (e.g. `if case_id == ...`) should
ever be added to `normalizer.py` to address them.

## Research integrity

- Step 14B (`backend/app/nli/`) — the model, its configuration, its
  thresholds, its aggregation rule, and its label mapping — is **frozen**
  and was not modified by this step.
- `datasets/nli_evaluation/benchmark.json` and the Step 14C results were
  not modified by this step.
- No network calls, no model downloads, and no new ML model are introduced.
  Normalization is pure Python string/regex processing.
- This step does not measure or claim any accuracy improvement. That
  measurement is explicitly reserved for Step 14G.
