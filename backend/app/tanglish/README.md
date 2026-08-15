# Tanglish-Aware Clinical NLI Preprocessing (Step 14F, refined in Step 14H)

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
- **Duration**: Tamil numerals 1–10 (`oru`...`pathu`), English number words
  (`one`...`ten`), or digits + `naala` / `naal` / `vaaram` / `vaarathula` /
  `day(s)` / `week(s)` / `month(s)` → "for two days" etc. (English-word and
  digit numerals were extended in Step 14H — see "Numeric / measurement
  preservation" below.)
- **Frequency**: Tamil numerals + `thadava` / `velai` → "once" / "twice" /
  "N times". A small set of time-of-day markers (`morning-la`, `night-la`)
  is also recognized.
- **Measurement values** (Step 14H): a number (Arabic, optionally decimal)
  adjacent to a `measurement`-typed clinical entity (heart rate,
  temperature, oxygen saturation, blood pressure, weight, blood sugar) is
  preserved verbatim, with or without a recognized unit word/symbol. See
  "Numeric / measurement preservation" below.
- **Clinical entities**: a fixed vocabulary of common symptoms,
  medications, measurements, procedures, and diagnoses (see
  `lexicon.py::CLINICAL_TERMS`) — these are recognized, never translated.
- **Code-switching / composition**: a single sentence can contain multiple
  clauses joined by "but"/"and"; each clause is normalized independently
  and the original conjunction is preserved, so multiple propositions (one
  positive, one negated) are never collapsed into one.
- **Attribution**: every clause is classified `PATIENT` / `FAMILY_THIRD_PERSON`
  / `UNKNOWN` before composition (Step 14H — see "Attribution safety"
  below). First-person markers (`enakku`, `naan`, `en`, ...) additionally
  set `"PATIENT_SELF_REPORT"` in `semantic_features.attributions` as a
  whole-text descriptive signal, independent of the per-clause
  classification.

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
| `Enakku cough irukku for three days.` | `Patient has cough for three days.` |
| `Enakku heart rate 88 beats per minute irukku.` | `Patient has heart rate of 88 beats per minute.` |
| `En oxygen level 98 percent irundhudhu.` | `Patient had oxygen saturation of 98 percent.` |
| `Family-la sila peruku diabetes irukku.` | `Family members have diabetes.` |
| `Amma diabetes irukku.` | `Patient's mother has diabetes.` |
| `Family-la appa-ku sugar 98 irundhudhu.` | `Patient's father had blood sugar of 98.` |

## Numeric / measurement preservation (Step 14H)

**Motivation.** Step 14G's failure analysis found that when a clause was
rewritten by the composer, a numerical detail that wasn't itself part of a
recognized *duration* pattern (Tamil numeral / English number word / digit
immediately followed by `day`/`week`/`month`) was silently dropped rather
than preserved — e.g. `"Enakku cough irukku for three days."` became
`"Patient has cough."`, losing the duration entirely; a bare
value-without-unit like `"sugar 98"` fared the same way. This is documented
as **"Numeral / measurement loss"** in the "Known failure modes" section
below.

**What was added:**

1. **English-word and digit numerals now feed duration extraction, not just
   Tamil numerals.** `_numeral_value()` and the duration-pattern numeral
   alternation now also recognize `one`...`ten` (English words) alongside
   the pre-existing Tamil numeral words and bare Arabic digits. This is why
   `"for three days"` (English words, inside an otherwise Tanglish clause)
   is now recognized exactly like `"rendu naala"` (Tamil) already was.
   `DURATION_UNIT_WORDS` also now recognizes `month`/`months`.
2. **A new, narrowly-scoped measurement-value extractor**
   (`_measurement_phrase_for_entity`) fires only for entities the lexicon
   already types `"measurement"` (heart rate, temperature, oxygen
   saturation/level, blood pressure, weight, blood sugar) and only searches
   a bounded window immediately after that specific entity — never a
   clause-wide scan, so it can never attach an unrelated number (e.g. a
   duration count elsewhere in the same clause) to the wrong entity.
   - If a number is immediately paired with a recognized unit word/symbol
     (`percent`, `%`, `bpm`, `beats per minute`, `degree(s)`, `°F`, `°C`,
     `mg`, `ml`, `g`, `kg`, `cm`, `mm` — see
     `lexicon.py::MEASUREMENT_UNIT_TERMS`), the full span (e.g.
     `"88 beats per minute"`, `"101.2 degree"`, `"98 percent"`) is
     preserved verbatim as `"of <span>"`.
   - If no unit word is present at all (e.g. `"sugar 98"`), the bare number
     is still preserved rather than dropped — losing the value entirely was
     the original failure mode.
   - **No unit conversion or canonicalization is attempted.** The matched
     span is inserted into the composed sentence exactly as written; this
     is preservation, not a measurement parser.
3. Every measurement match is recorded in
   `semantic_features.measurements` and as a `TANGLISH_MEASUREMENT`
   transformation, independent of the `TANGLISH_DURATION` /
   `TANGLISH_FREQUENCY` entries.

**Deliberately NOT attempted (conservative by design):** unit conversion
(mg↔g, °F↔°C), range/compound values (`"120/80"` blood pressure), numbers
not adjacent to a recognized measurement entity, or any numeral beyond the
existing 1–10 Tamil/English word tables (larger quantities must already be
written as Arabic digits, which the bare-digit fallback always accepted).

## Attribution safety (Step 14H)

**Motivation.** The pre-existing composer always used the fixed subject
`"Patient"` for every rewritten clause. Step 14G found this unsafe for
clauses that actually describe a family member or third party — e.g.
`"Family-la sila peruku diabetes irukku."` ("some people in the family have
diabetes") was composed as `"Patient has diabetes."`, converting a
family-history statement into an (incorrect) patient-level claim. This is
documented as **"Family-history / third-person misattribution"** below.

**What was added.** Every clause is classified into one of three
attribution categories (`_determine_attribution`) BEFORE composition:

| Category | When | Composed subject |
|---|---|---|
| `PATIENT` | Default — no family/third-person marker present (matches all pre-existing Step 14F behavior), or only a first-person marker (`enakku`, `naan`, `en`, ...) is present | `"Patient"` |
| `FAMILY_THIRD_PERSON` | A specific relation marker is present (`amma`/`appa`/`mother`/`father`/`brother`/`sister`/`husband`/`wife`/`son`/`daughter`/`parents`, incl. `amma-ku`/`appa-ku`) | `"Patient's <relation>"`, e.g. `"Patient's mother"` |
| `FAMILY_THIRD_PERSON` | Only a *generic* marker is present, naming no specific relation (`family`, `family-la`, `relatives`, `avanga`, `avar`, `aval`, `avanga-ku`, `avangalukku`) | `"Family members"` (generic — see safety rule below) |
| `UNKNOWN` | BOTH a first-person marker AND a family/third-person marker appear in the same clause (genuinely ambiguous to this rule-based system, e.g. `"Enakku appa diabetes irukku"`) | `"Someone"` |

**Safety rule (the critical property):** a clinical entity in a clause with
a strong family/third-person marker is **never** assigned to `"Patient"`.
When the signal is ambiguous (conflicting first-person + family markers in
one clause), the result is `UNKNOWN` (subject `"Someone"`) rather than
guessing — this project's normal safety-first default of *not fabricating*
extends here to *not mis-assigning to the patient either*. A specific
relation marker is preferred over a co-occurring generic marker (the more
informative, *actually stated* detail wins) — see
`"Family-la appa-ku sugar 98 irundhudhu."` → `"Patient's father had blood
sugar of 98."` in the examples table above.

**No fabrication:** when only a generic marker is present (no named
relation), the normalizer uses the generic `"Family members"` subject
rather than inventing which relative is meant.

**Recorded for research explainability:** whenever a clause's attribution
is not `PATIENT`, the label (`"FAMILY_THIRD_PERSON"` or `"UNKNOWN"`) is
appended to `semantic_features.attributions`, and a
`TANGLISH_ATTRIBUTION_<LABEL>` transformation records the matched marker
span and the subject it produced.

**Interaction with numeric preservation:** the two mechanisms are
independent — attribution is decided purely from clause text (before
entity/fact extraction), and measurement extraction is decided per-entity
(after facts are found) — so a clause can simultaneously get a non-Patient
subject AND preserve a numeric value, e.g. `"Family-la appa-ku sugar 98
irundhudhu."` → `"Patient's father had blood sugar of 98."`

**Deliberately NOT attempted (conservative by design):** true grammatical/
dependency parsing of who a number or condition belongs to when multiple
entities and multiple people are named in one clause; gender-accurate
pronoun resolution for `avar`/`aval` (treated identically, as a generic
third-person marker, rather than inferring "he" vs "she"); any relation
vocabulary beyond the fixed list above.

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
- **(Step 14H)** Measurement-value preservation performs no unit
  conversion, no plausibility/range checking, and no parsing of compound
  values (e.g. a blood-pressure reading written as `"120/80"`); it
  preserves whatever numeral + unit text is present, verbatim.
- **(Step 14H)** The bare-number fallback (a number with no unit word at
  all, e.g. `"sugar 98"`) is scoped to a fixed character window after a
  `measurement`-typed entity specifically to avoid false positives, but it
  is still a coarse heuristic — if two numbers happen to appear near the
  same measurement entity, only the first one found is used.
- **(Step 14H)** Attribution classification is a same-clause marker
  co-occurrence check, not a parse of *which* entity in a multi-entity
  clause belongs to *which* person; a clause with more than one clinical
  entity and a family marker attributes ALL entities in that clause to the
  same determined subject.
- **(Step 14H)** The generic family subject `"Family members"` is used for
  any unspecified/generic third-person marker (`family`, `relatives`,
  `avanga`, `avar`, `aval`, ...) regardless of whether the source implied
  one person or several — this avoids fabricating a specific relationship,
  but is not grammatically precise for a singular pronoun like `avar`/`aval`.
- **(Step 14H)** The `UNKNOWN` ambiguity rule is a simple co-occurrence
  check (first-person marker AND family marker both present in one
  clause) — it does not attempt to determine which reading is actually
  intended; it exists purely to avoid guessing wrong in either direction.
- As with the rest of this module: **this remains a deterministic,
  rule-based heuristic preprocessing layer.** It has not been validated as
  a clinical NLP system, makes no claim of linguistic completeness for
  Tamil-English code-switching, and its output should not be treated as a
  substitute for clinical judgment or a validated medical NLP annotation.

## Known failure modes identified by Step 14G (Step 14H status noted below)

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

1. **Numeral / measurement loss** — **Status: partially addressed in Step
   14H.**

   *Original finding (Step 14G):* numerical expressions written in English
   form inside an otherwise Tanglish/mixed clause -- e.g. `"for three
   days"`, `"98 percent"`, `"88 beats per minute"` -- were not captured by
   the duration pattern (which only matched Tamil numeral words or bare
   digits immediately followed by a recognized unit word like
   `naala`/`vaaram`), and no measurement-value pattern existed at all
   (measurement *entities* were recognized via `lexicon.CLINICAL_TERMS`,
   but their numeric *values* were not extracted or preserved). The
   composed sentence kept the recognized entity but dropped the
   value/duration, e.g. producing `"Patient has cough."` instead of
   `"Patient has cough for three days."` This could weaken NLI evidence
   matching whenever the SOAP claim being verified included that dropped
   numerical/duration detail. Observed in Step 14G's regressed cases
   MX-001, MX-006, MX-008, and TL-011.

   *Step 14H change:* English-form duration numerals (e.g. `"three"` in
   `"for three days"`) are now recognized identically to Tamil numerals
   (`_numeral_value` / `_DURATION_PATTERN` extended with
   `lexicon.ENGLISH_NUMBER_WORD_VALUES`), and a new, narrowly-scoped
   measurement-value extractor (`_measurement_phrase_for_entity`) now
   preserves numbers adjacent to `measurement`-typed entities, with or
   without a recognized unit word. See "Numeric / measurement preservation"
   above for the full mechanism and its scope. Unit tests in
   `backend/tests/test_tanglish.py` (`test_numeric_*`,
   `test_numeric_step_14g_regressed_cases_now_preserve_value`) confirm the
   Step 14G MX-001/MX-006/MX-008/TL-011 premises now preserve their numeric
   detail through normalization.

   *Why "partially" and not "fully" addressed:* no unit
   conversion/canonicalization is attempted; compound values (e.g. a blood
   pressure reading written as `"120/80"`) are not parsed; and the fix only
   covers entities the lexicon already types `"measurement"` — a number
   next to an entity of a different type (e.g. a medication dosage) is
   still not extracted. Most importantly: **no Step 14G-style 101-case
   benchmark evaluation has been rerun yet to measure the actual effect on
   Tanglish/Mixed accuracy** — that controlled comparison is explicitly
   reserved for Step 14I. This status reflects unit-level test evidence
   only, not a rerun benchmark comparison.

2. **Family-history / third-person misattribution** — **Status: addressed
   by a generic rule in Step 14H** (see caveats below).

   *Original finding (Step 14G):* the normalizer had no family-member or
   third-person-subject detection at all. A clause reporting that *someone
   other than the patient* has a condition -- e.g. `"Family-la sila peruku
   diabetes irukku."` ("some people in the family have diabetes") -- still
   matched the `CURRENT` trigger (`irukku`) adjacent to the `diabetes`
   entity and was composed with the fixed subject `"Patient"`, producing
   `"Patient has diabetes."` This silently converted a family-history
   statement into a patient-level claim, which could turn a correct
   `UNGROUNDED` prediction into an incorrect `SUPPORTED` one. Observed in
   Step 14G's regressed case MX-016.

   *Step 14H change:* every clause is now classified `PATIENT` /
   `FAMILY_THIRD_PERSON` / `UNKNOWN` before composition
   (`_determine_attribution`), using a generic family/third-person marker
   vocabulary (`lexicon.FAMILY_SPECIFIC_RELATION_MARKERS`,
   `FAMILY_PLURAL_RELATION_MARKERS`, `FAMILY_GENERIC_MARKERS`) — see
   "Attribution safety" above. The original MX-016 premise now normalizes
   to `"Family members have diabetes."`, no longer `"Patient has
   diabetes."` (see `test_attribution_step_14g_regressed_case_no_longer_
   misattributed`).

   *Caveats on "addressed":* the fix is a same-clause marker
   co-occurrence heuristic, not true parsing — see the Step 14H limitations
   above (generic `"Family members"` subject, `UNKNOWN` co-occurrence rule,
   no multi-entity/multi-person disambiguation). And as with finding #1,
   **no Step 14G-style benchmark rerun has been performed yet** to confirm
   this generic rule doesn't introduce new regressions elsewhere in the
   101-case benchmark or measure its net effect — that is Step 14I's job.

Both findings are, and remain, reported as **generic** linguistic gaps in
the rule coverage, not as issues with any specific benchmark case -- the
case IDs above are cited only as the observed instances that surfaced each
gap during evaluation, and (for #1) as regression-test evidence that the
resulting generic rule actually resolves them, never as targets for
case-specific patches. Per this project's research-integrity practice, no
benchmark-specific exception (e.g. `if case_id == ...` / `if premise ==
...`) exists anywhere in `normalizer.py`.

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
- **(Step 14H)** The same guarantees hold for the numeric-preservation and
  attribution-safety refinement: no new dependency, no network calls, no
  database access, no LLM/external API/vector database/RAG, no model of any
  kind, and no changes to `backend/app/nli/`, Step 14A retrieval, Step 14C
  evaluation code, `benchmark.json`, or the Step 14G result files under
  `datasets/nli_evaluation/results/tanglish_comparison/`. Step 14H does not
  itself measure or claim any accuracy improvement over Step 14G's results
  — a new controlled 101-case comparison against those refined rules is
  Step 14I's job, not this step's.
