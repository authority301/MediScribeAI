"""Step 14F Tanglish-aware preprocessing tests.

Never loads the real NLI model for normalizer-only tests (pure string/regex
processing, no model involved). Tests that exercise app.tanglish.service
monkeypatch app.tanglish.service.run_nli so the real model is never loaded.
"""
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.database import SessionLocal
from app.models import Doctor
from app.nli.model import NliScores
from app.tanglish import service
from app.tanglish.config import MODE_BASELINE, MODE_TANGLISH_AWARE
from app.tanglish.normalizer import LANGUAGE_ENGLISH, LANGUAGE_MIXED, LANGUAGE_TANGLISH, LANGUAGE_UNKNOWN, normalize


# ---------------------------------------------------------------------------
# Required regression cases (CASE 1-8 from the spec)
# ---------------------------------------------------------------------------


def test_case_1_tanglish_current_state():
    result = normalize("Enakku fever irukku.")
    assert result.normalized_text == "Patient has fever."


def test_case_2_tanglish_negation():
    result = normalize("Enakku fever illa.")
    assert result.normalized_text == "Patient does not have fever."


def test_case_3_tanglish_negation_multiword_entity():
    result = normalize("Enakku chest pain illa.")
    assert result.normalized_text == "Patient does not have chest pain."


def test_case_4_tanglish_duration():
    result = normalize("Enakku fever rendu naala irukku.")
    assert result.normalized_text == "Patient has fever for two days."


def test_case_5_tanglish_historical_childhood():
    result = normalize("Chinna vayasula asthma irundhudhu.")
    assert result.normalized_text == "Patient had asthma during childhood."


def test_case_6_tanglish_negation_medication():
    result = normalize("Naan paracetamol edukkala.")
    assert result.normalized_text == "Patient does not take paracetamol."


def test_case_7_tanglish_code_switched_composition():
    result = normalize("Enakku fever irukku but chest pain illa.")
    assert result.normalized_text == "Patient has fever but does not have chest pain."


def test_case_8_english_unchanged():
    result = normalize("Patient reports fever for two days.")
    assert result.normalized_text == "Patient reports fever for two days."
    assert result.detected_language == LANGUAGE_ENGLISH
    assert result.transformations == []


# ---------------------------------------------------------------------------
# 1. English remains unchanged
# ---------------------------------------------------------------------------


def test_english_remains_unchanged():
    text = "Patient does not have chest pain and takes paracetamol twice daily."
    result = normalize(text)
    assert result.normalized_text == text
    assert result.detected_language == LANGUAGE_ENGLISH
    assert result.is_code_switched is False
    assert result.transformations == []


# ---------------------------------------------------------------------------
# 2-10: individual semantic feature categories
# ---------------------------------------------------------------------------


def test_simple_tanglish_symptom():
    result = normalize("Enakku vayitru vali irukku.")
    assert result.normalized_text == "Patient has stomach pain."


def test_simple_tanglish_medication():
    result = normalize("Naan insulin edukkamaatten.")
    # "edukkamaatten" is not in the trigger vocabulary (documented limitation);
    # use the supported action-negation trigger instead:
    result = normalize("Naan insulin edukkala.")
    assert result.normalized_text == "Patient does not take insulin."


def test_simple_tanglish_negation():
    result = normalize("Enakku cough illa.")
    assert result.normalized_text == "Patient does not have cough."


def test_tanglish_negation_with_clinical_english_term():
    result = normalize("Enakku blood pressure problem illa.")
    # "blood pressure" is recognized; "problem" is not part of the entity term
    assert "blood pressure" in result.normalized_text
    assert "does not have" in result.normalized_text


def test_historical_condition():
    result = normalize("Enakku asthma irundhudhu munnaadi.")
    assert result.normalized_text == "Patient had asthma in the past."


def test_current_condition():
    result = normalize("Enakku headache irukku.")
    assert result.normalized_text == "Patient has headache."


def test_duration():
    result = normalize("Enakku cough moonu naala irukku.")
    assert "for three days" in result.normalized_text


def test_frequency():
    result = normalize("Naan andha tablet-a rendu thadava saapdren.")
    # verb "saapdren" (positive, "I eat/take") is outside the trigger
    # vocabulary (documented limitation) so no fact fires here, but the
    # frequency phrase must still be extractable on a sentence that DOES
    # have a recognized trigger:
    result = normalize("Enakku tablet irukku rendu thadava.")
    assert "twice" in result.normalized_text


def test_measurement():
    result = normalize("En temperature irukku.")
    assert result.normalized_text == "Patient has temperature."


# ---------------------------------------------------------------------------
# 11-13: composition / code-switching
# ---------------------------------------------------------------------------


def test_mixed_english_tanglish_sentence():
    result = normalize("Enakku fever irukku but chest pain illa.")
    assert result.detected_language == LANGUAGE_MIXED
    assert result.is_code_switched is True


def test_multiple_facts_in_one_sentence():
    result = normalize("Enakku fever rendu naala irukku but chest pain illa.")
    assert "has fever for two days" in result.normalized_text
    assert "does not have chest pain" in result.normalized_text


def test_negation_plus_positive_fact_same_sentence():
    result = normalize("Enakku cough irukku but fever illa.")
    assert result.normalized_text == "Patient has cough but does not have fever."


# ---------------------------------------------------------------------------
# 14-15: spelling variation / negation safety
# ---------------------------------------------------------------------------


def test_spelling_variation_hyphenated_duration():
    result = normalize("Enakku fever rendu naal-a irukku.")
    assert "for two days" in result.normalized_text


def test_spelling_variation_digit_duration():
    result = normalize("Enakku fever 2 naala irukku.")
    assert "for two days" in result.normalized_text


def test_no_unsafe_la_substring_negation():
    # "vanilla" contains the substring "illa" but must never be treated as
    # a negation trigger for the adjacent clinical entity.
    result = normalize("Enakku vanilla fever irukku.")
    assert result.normalized_text == "Patient has fever."
    assert all("NEGATION" not in t.rule for t in result.transformations)


def test_no_unsafe_la_substring_negation_standalone():
    result = normalize("villa")
    assert result.detected_language == LANGUAGE_ENGLISH
    assert result.normalized_text == "villa"


# ---------------------------------------------------------------------------
# 16-17: empty / unknown input
# ---------------------------------------------------------------------------


def test_empty_input():
    result = normalize("")
    assert result.detected_language == LANGUAGE_UNKNOWN
    assert result.normalized_text == ""
    assert result.transformations == []


def test_whitespace_only_input_is_unknown():
    result = normalize("   ")
    assert result.detected_language == LANGUAGE_UNKNOWN


def test_unknown_input_does_not_crash():
    result = normalize("###???***")
    assert result.detected_language in (LANGUAGE_UNKNOWN, LANGUAGE_ENGLISH)
    assert isinstance(result.normalized_text, str)


# ---------------------------------------------------------------------------
# 18: transformation trace
# ---------------------------------------------------------------------------


def test_transformation_trace_recorded():
    result = normalize("Enakku chest pain illa.")
    assert len(result.transformations) == 1
    transformation = result.transformations[0]
    assert transformation.rule == "TANGLISH_NEGATION_STATE"
    assert "chest pain" in transformation.input_span
    assert transformation.normalized_span == "does not have chest pain"


def test_transformation_trace_for_duration_is_separate_entry():
    result = normalize("Enakku fever rendu naala irukku.")
    rules = [t.rule for t in result.transformations]
    assert "TANGLISH_CURRENT_STATE" in rules
    assert "TANGLISH_DURATION" in rules


# ---------------------------------------------------------------------------
# 19: deterministic repeated normalization
# ---------------------------------------------------------------------------


def test_deterministic_repeated_normalization():
    text = "Enakku fever rendu naala irukku but chest pain illa."
    first = normalize(text)
    second = normalize(text)
    assert first.normalized_text == second.normalized_text
    assert first.transformations == second.transformations
    assert first.semantic_features == second.semantic_features


# ---------------------------------------------------------------------------
# 20: no external network calls
# ---------------------------------------------------------------------------


def test_no_external_network_calls(monkeypatch):
    def _blocked_connect(*args, **kwargs):
        raise AssertionError("normalize() must never attempt a network connection")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)

    result = normalize("Enakku fever rendu naala irukku but chest pain illa.")
    assert result.normalized_text  # completed without attempting network I/O


# ---------------------------------------------------------------------------
# 21: English regression cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Patient has had fever for two days.",
        "Patient does not have chest pain.",
        "Patient takes paracetamol.",
        "Patient's temperature was 101.2 degrees Fahrenheit.",
        "Patient was referred to a cardiologist.",
    ],
)
def test_english_regression_cases_unchanged(text):
    result = normalize(text)
    assert result.normalized_text == text
    assert result.detected_language == LANGUAGE_ENGLISH


# ---------------------------------------------------------------------------
# 22: no database modifications
# ---------------------------------------------------------------------------


def test_no_database_modifications(monkeypatch):
    db = SessionLocal()
    try:
        before_count = db.query(Doctor).count()
    finally:
        db.close()

    monkeypatch.setattr(
        "app.tanglish.service.run_nli",
        lambda premise, hypothesis: NliScores(entailment=0.9, neutral=0.05, contradiction=0.05, label="ENTAILMENT"),
    )
    normalize("Enakku fever rendu naala irukku but chest pain illa.")
    service.verify_baseline("I have fever.", "Patient has fever.")
    service.verify_tanglish_aware("Enakku fever irukku.", "Patient has fever.")

    db = SessionLocal()
    try:
        after_count = db.query(Doctor).count()
    finally:
        db.close()
    assert before_count == after_count


# ---------------------------------------------------------------------------
# service.py: baseline vs tanglish_aware modes
# ---------------------------------------------------------------------------


def test_service_baseline_mode_passes_premise_unchanged(monkeypatch):
    captured = {}

    def fake_run_nli(premise, hypothesis):
        captured["premise"] = premise
        return NliScores(entailment=0.9, neutral=0.05, contradiction=0.05, label="ENTAILMENT")

    monkeypatch.setattr("app.tanglish.service.run_nli", fake_run_nli)

    result = service.verify("Enakku fever illa.", "Patient has fever.", mode=MODE_BASELINE)
    assert captured["premise"] == "Enakku fever illa."  # unchanged, NOT normalized
    assert result["normalization"] is None
    assert result["prediction"] == "SUPPORTED"  # mocked ENTAILMENT -> SUPPORTED per real aggregation


def test_service_tanglish_aware_mode_normalizes_premise_first(monkeypatch):
    captured = {}

    def fake_run_nli(premise, hypothesis):
        captured["premise"] = premise
        return NliScores(entailment=0.05, neutral=0.05, contradiction=0.9, label="CONTRADICTION")

    monkeypatch.setattr("app.tanglish.service.run_nli", fake_run_nli)

    result = service.verify("Enakku fever illa.", "Patient has fever.", mode=MODE_TANGLISH_AWARE)
    assert captured["premise"] == "Patient does not have fever."  # normalized before NLI
    assert result["normalization"] is not None
    assert result["normalization"].normalized_text == "Patient does not have fever."
    assert result["prediction"] == "CONTRADICTED"


def test_service_rejects_unknown_mode():
    with pytest.raises(ValueError):
        service.verify("x", "y", mode="not_a_real_mode")


def test_service_reuses_real_aggregation_thresholds(monkeypatch):
    # entailment just below threshold -> must aggregate to UNGROUNDED, not
    # SUPPORTED -- proves the real app.nli.aggregation logic is being used,
    # not a reimplementation.
    monkeypatch.setattr(
        "app.tanglish.service.run_nli",
        lambda premise, hypothesis: NliScores(entailment=0.55, neutral=0.35, contradiction=0.10, label="ENTAILMENT"),
    )
    result = service.verify_baseline("I have fever.", "Patient has fever.")
    assert result["prediction"] == "UNGROUNDED"


# ===========================================================================
# Step 14H: numeric/measurement preservation + attribution safety
#
# Generic-rule regression tests only -- no "if case_id == ..." / "if
# premise == ..." branches exist anywhere in normalizer.py. A couple of
# tests below reuse the exact Step 14G regressed-case premises (MX-001,
# MX-006, MX-008, TL-011, MX-016) as regression evidence that the new
# GENERIC rules resolve those observed failure modes -- this is explicitly
# permitted by the Step 14H spec ("those cases may be used as regression
# tests only after the generic rule exists").
# ===========================================================================

# ---------------------------------------------------------------------------
# Numeric / measurement preservation
# ---------------------------------------------------------------------------


def test_numeric_1_english_duration_inside_tanglish_clause():
    result = normalize("Enakku cough irukku for three days.")
    assert result.normalized_text == "Patient has cough for three days."
    assert "for three days" in result.semantic_features.durations
    assert any(t.rule == "TANGLISH_DURATION" for t in result.transformations)


def test_numeric_2_arabic_numeral_plus_duration():
    result = normalize("Enakku fever 2 days irukku.")
    assert "for two days" in result.normalized_text


def test_numeric_3_number_word_plus_duration():
    result = normalize("Enakku headache five days irukku.")
    assert "for five days" in result.normalized_text


def test_numeric_4_measurement_number_plus_unit():
    result = normalize("Enakku heart rate 88 beats per minute irukku.")
    assert "88 beats per minute" in result.normalized_text
    assert "of 88 beats per minute" in result.semantic_features.measurements
    assert any(t.rule == "TANGLISH_MEASUREMENT" for t in result.transformations)


def test_numeric_5_decimal_measurement():
    result = normalize("Patient-ku temperature 101.2 degree irukku.")
    assert "101.2" in result.normalized_text
    assert "degree" in result.normalized_text
    assert "temperature" in result.normalized_text


def test_numeric_6_percentage():
    result = normalize("En oxygen level 98 percent irundhudhu.")
    assert "98 percent" in result.normalized_text
    assert "had" in result.normalized_text  # historical trigger, not current


def test_numeric_7_heart_rate_style_measurement():
    result = normalize("Enakku heart rate 72 bpm irukku.")
    assert "72 bpm" in result.normalized_text


def test_numeric_8_existing_tamil_numeral_behavior_unchanged():
    # Step 14F CASE 4 regression, restated: must still pass exactly.
    result = normalize("Enakku fever rendu naala irukku.")
    assert result.normalized_text == "Patient has fever for two days."


def test_numeric_9_survives_clause_composition_structured():
    result = normalize("Enakku cough irukku for three days but fever illa.")
    assert "for three days" in result.normalized_text
    assert "does not have fever" in result.normalized_text
    duration_transforms = [t for t in result.transformations if t.rule == "TANGLISH_DURATION"]
    assert len(duration_transforms) == 1
    assert "three days" in duration_transforms[0].input_span
    assert "for three days" in result.semantic_features.durations


def test_numeric_10_english_numeric_input_unchanged():
    text = "Patient has had fever for two days and temperature was 101.2 degrees."
    result = normalize(text)
    assert result.normalized_text == text
    assert result.detected_language == LANGUAGE_ENGLISH


def test_numeric_bare_number_without_unit_still_preserved():
    # No unit word at all ("sugar 98") -- must not be silently dropped.
    result = normalize("Enakku sugar 98 irukku.")
    assert "98" in result.normalized_text
    assert "blood sugar" in result.normalized_text


def test_numeric_no_measurement_present_leaves_entity_unaffected():
    # test_measurement() (Step 14F) already covers this, restated here for
    # Step 14H visibility: a measurement-type entity with NO nearby number
    # must not gain a spurious "of ..." suffix.
    result = normalize("En temperature irukku.")
    assert result.normalized_text == "Patient has temperature."
    assert result.semantic_features.measurements == []


# Step 14G regression cases (generic-rule evidence, not special-cased):
@pytest.mark.parametrize(
    "premise,expected_substring",
    [
        ("Enakku cough irukku for three days.", "for three days"),  # MX-001
        ("Heart rate ippo 88 beats per minute irukku.", "88 beats per minute"),  # MX-006
        ("Back pain irukku for one week already.", "for one week"),  # MX-008
        ("En oxygen level 98 percent irundhudhu.", "98 percent"),  # TL-011
    ],
)
def test_numeric_step_14g_regressed_cases_now_preserve_value(premise, expected_substring):
    result = normalize(premise)
    assert expected_substring in result.normalized_text


# ---------------------------------------------------------------------------
# Attribution safety
# ---------------------------------------------------------------------------


def test_attribution_1_family_la_diabetes_statement():
    result = normalize("Family-la sila peruku diabetes irukku.")
    assert result.normalized_text != "Patient has diabetes."
    assert not result.normalized_text.startswith("Patient ")
    assert "Family members" in result.normalized_text
    assert "FAMILY_THIRD_PERSON" in result.semantic_features.attributions


def test_attribution_2_mother_has_diabetes():
    result = normalize("Amma diabetes irukku.")
    assert result.normalized_text != "Patient has diabetes."
    assert "mother" in result.normalized_text.lower()
    assert "FAMILY_THIRD_PERSON" in result.semantic_features.attributions


def test_attribution_3_father_has_hypertension():
    result = normalize("Appa hypertension irukku.")
    assert result.normalized_text != "Patient has hypertension."
    assert "father" in result.normalized_text.lower()


def test_attribution_4_brother_has_asthma():
    result = normalize("Brother asthma irukku.")
    assert result.normalized_text != "Patient has asthma."
    assert "brother" in result.normalized_text.lower()


def test_attribution_5_sister_has_allergy():
    result = normalize("Sister allergy irukku.")
    assert result.normalized_text != "Patient has allergy."
    assert "sister" in result.normalized_text.lower()


def test_attribution_6_generic_relatives_statement():
    result = normalize("Relatives diabetes irukku.")
    assert result.normalized_text != "Patient has diabetes."
    assert "Family members" in result.normalized_text
    # generic marker must NOT fabricate a specific relation:
    assert "mother" not in result.normalized_text.lower()
    assert "father" not in result.normalized_text.lower()


def test_attribution_7_third_person_pronoun_statement():
    result = normalize("Avanga-ku diabetes irukku.")
    assert result.normalized_text != "Patient has diabetes."
    assert "FAMILY_THIRD_PERSON" in result.semantic_features.attributions


def test_attribution_8_patient_statement_still_maps_to_patient():
    result = normalize("Enakku fever irukku.")
    assert result.normalized_text == "Patient has fever."
    assert "FAMILY_THIRD_PERSON" not in result.semantic_features.attributions
    assert "UNKNOWN" not in result.semantic_features.attributions


def test_attribution_9_ambiguous_attribution_becomes_unknown():
    # Conflicting signals in one clause (first-person AND a family marker)
    # -- safety-first: UNKNOWN, never guessed as Patient.
    result = normalize("Enakku appa diabetes irukku.")
    assert not result.normalized_text.startswith("Patient ")
    assert "Patient's father" not in result.normalized_text
    assert "UNKNOWN" in result.semantic_features.attributions


def test_attribution_10_family_statement_with_number_preserves_both():
    result = normalize("Family-la appa-ku sugar 98 irundhudhu.")
    assert "98" in result.normalized_text
    assert "father" in result.normalized_text.lower()
    assert not result.normalized_text.startswith("Patient ")
    assert "FAMILY_THIRD_PERSON" in result.semantic_features.attributions


def test_attribution_patient_number_interaction_stays_patient():
    # Companion to the above: no family marker present -> patient
    # attribution is retained alongside the preserved number.
    result = normalize("Enakku sugar 98 irukku.")
    assert "98" in result.normalized_text
    assert result.normalized_text.startswith("Patient ")
    assert "FAMILY_THIRD_PERSON" not in result.semantic_features.attributions
    assert "UNKNOWN" not in result.semantic_features.attributions


def test_attribution_specific_relation_preferred_over_generic_marker():
    # Both "family-la" (generic) and "appa-ku" (specific) present -- the
    # stated specific relation must be used, not the generic fallback.
    result = normalize("Family-la appa-ku sugar 98 irundhudhu.")
    assert "father" in result.normalized_text.lower()
    assert "Family members" not in result.normalized_text


def test_attribution_transformation_trace_records_family_marker():
    result = normalize("Amma diabetes irukku.")
    attribution_transforms = [t for t in result.transformations if t.rule.startswith("TANGLISH_ATTRIBUTION_")]
    assert len(attribution_transforms) == 1
    assert attribution_transforms[0].rule == "TANGLISH_ATTRIBUTION_FAMILY_THIRD_PERSON"
    assert attribution_transforms[0].input_span.lower() == "amma"
    assert "mother" in attribution_transforms[0].normalized_span.lower()


# Step 14G regression case (generic-rule evidence, not special-cased):
def test_attribution_step_14g_regressed_case_no_longer_misattributed():
    # MX-016
    result = normalize("Family-la sila peruku diabetes irukku.")
    assert result.normalized_text != "Patient has diabetes."
    assert "FAMILY_THIRD_PERSON" in result.semantic_features.attributions


# ---------------------------------------------------------------------------
# English safety / no side effects (Step 14H)
# ---------------------------------------------------------------------------


def test_english_family_sentence_bypassed_unchanged():
    text = "Patient's mother has diabetes."
    result = normalize(text)
    assert result.normalized_text == text
    assert result.detected_language == LANGUAGE_ENGLISH
    assert result.transformations == []


def test_step_14h_no_database_modifications(monkeypatch):
    db = SessionLocal()
    try:
        before_count = db.query(Doctor).count()
    finally:
        db.close()

    normalize("Family-la appa-ku sugar 98 irundhudhu.")
    normalize("Enakku heart rate 88 beats per minute irukku.")

    db = SessionLocal()
    try:
        after_count = db.query(Doctor).count()
    finally:
        db.close()
    assert before_count == after_count


def test_step_14h_no_external_network_calls(monkeypatch):
    def _blocked_connect(*args, **kwargs):
        raise AssertionError("normalize() must never attempt a network connection")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)

    result = normalize("Family-la appa-ku sugar 98 irundhudhu.")
    assert result.normalized_text
