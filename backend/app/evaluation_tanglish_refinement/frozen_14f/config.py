"""Step 14F configuration: mode names and tunable-but-not-tuned constants
for the Tanglish-aware preprocessing layer. No env vars are strictly
required -- this module never talks to a model or network -- but mode
constants live here so callers never hard-code magic strings.
"""

MODE_BASELINE = "baseline"
MODE_TANGLISH_AWARE = "tanglish_aware"

VALID_MODES = (MODE_BASELINE, MODE_TANGLISH_AWARE)

# Maximum character distance allowed between a clinical entity and a
# trigger word (negation/current/historical) for the two to be associated
# as one fact. Generous enough to span an intervening duration phrase
# (e.g. "fever rendu naala irukku"), but bounded so unrelated words late in
# a long sentence are never incorrectly attached to an early entity.
ENTITY_TRIGGER_WINDOW_CHARS = 40
