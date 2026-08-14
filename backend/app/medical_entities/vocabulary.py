"""Prototype medical vocabulary for the deterministic, rule-based baseline extractor.

This is a SMALL, HAND-CURATED prototype list -- it is NOT comprehensive
medical knowledge and NOT a validated clinical terminology (no SNOMED/ICD
mapping, no comprehensive Tamil medical NLP). It exists to establish a
reproducible, testable baseline that can later be compared against a
transformer-based extractor. Extend it here rather than scattering terms
through route/service code.

Each entity_type maps canonical_term -> list of surface-form variants
(English and a few common Tamil / code-switched forms where practical).
Matching any variant produces normalized_text = canonical_term; entity_text
always preserves what was actually said.
"""

# entity_type: SYMPTOM
SYMPTOM_VOCAB: dict[str, list[str]] = {
    "fever": ["fever", "high fever", "temperature", "காய்ச்சல்"],
    "cough": ["cough", "coughing", "இருமல்"],
    "headache": ["headache", "தலைவலி"],
    "stomach pain": ["stomach pain", "stomach ache", "abdominal pain", "வயிற்று வலி"],
    "chest pain": ["chest pain"],
    "nausea": ["nausea", "nauseous"],
    "vomiting": ["vomiting", "throwing up", "வாந்தி"],
    "diarrhea": ["diarrhea", "diarrhoea", "loose motions", "loose motion"],
    "fatigue": ["fatigue", "tiredness"],
    "dizziness": ["dizziness", "dizzy"],
    "sore throat": ["sore throat", "throat pain", "தொண்டை வலி"],
    "shortness of breath": ["shortness of breath", "breathlessness", "difficulty breathing"],
    "back pain": ["back pain"],
    "joint pain": ["joint pain"],
    "rash": ["rash", "skin rash"],
    "chills": ["chills"],
}

# entity_type: MEDICATION
MEDICATION_VOCAB: dict[str, list[str]] = {
    "paracetamol": ["paracetamol", "acetaminophen", "crocin", "dolo"],
    "ibuprofen": ["ibuprofen", "brufen"],
    "amoxicillin": ["amoxicillin", "amoxyclav"],
    "aspirin": ["aspirin"],
    "insulin": ["insulin"],
    "metformin": ["metformin"],
    "antibiotic": ["antibiotic", "antibiotics"],
    "painkiller": ["painkiller", "pain killer", "pain reliever"],
    "antihistamine": ["antihistamine", "cetirizine"],
}

# entity_type: ALLERGY
ALLERGY_VOCAB: dict[str, list[str]] = {
    "penicillin allergy": ["allergic to penicillin", "penicillin allergy"],
    "dust allergy": ["dust allergy", "allergic to dust"],
    "peanut allergy": ["peanut allergy", "allergic to peanuts", "allergic to peanut"],
    "pollen allergy": ["pollen allergy", "allergic to pollen"],
    "food allergy": ["food allergy"],
    "drug allergy": ["drug allergy"],
}

# entity_type: BODY_PART
BODY_PART_VOCAB: dict[str, list[str]] = {
    "head": ["head", "தலை"],
    "chest": ["chest", "மார்பு"],
    "stomach": ["stomach", "abdomen", "வயிறு"],
    "back": ["back"],
    "throat": ["throat", "தொண்டை"],
    "leg": ["legs", "leg"],
    "arm": ["arms", "arm"],
    "knee": ["knee"],
    "ear": ["ear"],
    "eye": ["eyes", "eye"],
    "hand": ["hands", "hand"],
    "foot": ["feet", "foot"],
}

# entity_type: MEDICAL_TEST
MEDICAL_TEST_VOCAB: dict[str, list[str]] = {
    "blood test": ["blood test", "blood work"],
    "x-ray": ["x-ray", "xray"],
    "ecg": ["electrocardiogram", "ecg", "ekg"],
    "mri": ["mri"],
    "ct scan": ["ct scan", "cat scan"],
    "ultrasound": ["ultrasound", "sonography"],
    "blood pressure check": ["blood pressure check", "bp check"],
    "urine test": ["urine test"],
}

# entity_type: MEDICAL_PROCEDURE
PROCEDURE_VOCAB: dict[str, list[str]] = {
    "surgery": ["surgery", "operation"],
    "biopsy": ["biopsy"],
    "vaccination": ["vaccination", "vaccine", "injection"],
    "dressing": ["wound dressing", "dressing"],
    "stitches": ["stitches", "suturing"],
}

# entity_type: DIAGNOSIS_MENTION
# NOTE: these are lexical MENTIONS of a diagnosis term appearing in speech --
# never a diagnosis produced/inferred by this system (see extraction.py's
# family-reference handling, which prevents attributing a family member's
# mentioned condition to the patient).
DIAGNOSIS_VOCAB: dict[str, list[str]] = {
    "diabetes": ["diabetes", "diabetic"],
    "hypertension": ["hypertension", "high blood pressure"],
    "asthma": ["asthma"],
    "pneumonia": ["pneumonia"],
    "migraine": ["migraine"],
    "arthritis": ["arthritis"],
    "anemia": ["anemia", "anaemia"],
    "tuberculosis": ["tuberculosis"],
    "covid-19": ["covid-19", "covid", "coronavirus"],
}

VOCAB_BY_ENTITY_TYPE: dict[str, dict[str, list[str]]] = {
    "SYMPTOM": SYMPTOM_VOCAB,
    "MEDICATION": MEDICATION_VOCAB,
    "ALLERGY": ALLERGY_VOCAB,
    "BODY_PART": BODY_PART_VOCAB,
    "MEDICAL_TEST": MEDICAL_TEST_VOCAB,
    "MEDICAL_PROCEDURE": PROCEDURE_VOCAB,
    "DIAGNOSIS_MENTION": DIAGNOSIS_VOCAB,
}
