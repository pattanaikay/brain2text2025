# Preprint — two framings

This study has two standalone draft variants. They share an **identical technical core** (§3 Method,
§4 Setup, §5 Results, §7 Limitations); only the framing sections (title, Abstract, Intro, Related Work
emphasis, Discussion/Significance) and the target venue differ.

| Variant | File | Framing | Target |
|---------|------|---------|--------|
| Methods / TTA | [`preprint_ttu.md`](preprint_ttu.md) | iterative test-time adaptation under real *sequential* drift; BCI as testbed; the label-quality safe-regime + memory stabilizer + constant-latency result | ICLR 2026 — Test-Time Updates (TTU) / "Catch, Adapt, and Operate: Monitoring ML Models Under Drift" |
| Clinical / Health | [`preprint_health.md`](preprint_health.md) | at-home speech neuroprosthesis usability; recalibration-free, constant-latency, reliable-from-sentence-one via episodic memory | NeurIPS-style health venue — Learning from Time Series for Health (TS4H) / ML4H |

Shared bibliography: [`references.bib`](references.bib).

**Maintenance note:** §3–§5 and §7 are duplicated verbatim across the two files — edit both (or refactor
to a shared include) when changing the method, setup, results, or limitations.
