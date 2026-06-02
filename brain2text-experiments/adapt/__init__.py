"""
adapt/
------
Between-trial / between-session ADAPTATION procedures (test-time training).

These are NOT Stack forward stages (encoder→memory→projector→decoder); they are
procedures that run *around* the inference loop, refining a small set of model
parameters from the model's own pseudo-labels. DietCorp's per-trial TTA lives
here, generalised to N "sleep" consolidation passes (Track G2/G3).
"""
