# PCSA prompt provenance

Source: Qingyang Xu et al., *Do No Harm: Exposing Hidden Vulnerabilities of LLMs
via Persona-based Client Simulation Attack in Psychological Counseling*,
[arXiv:2604.04842v2](https://arxiv.org/html/2604.04842v2), 2026-09-05, CC BY 4.0.

The figure text was retrieved from the authors' arXiv HTML. Layout whitespace
and spaces inside placeholder braces were normalized. Scoring wording was not
rewritten. Figure 4 (Appendix D.2; printed page 17) is the actual four-axis judge
template used by the pilot. Figure 3 (D.1; printed page 16) is retained for
reference. `provenance.json` records attribution, source and file hashes.

The pilot adds data-boundary instructions, literal-evidence/schema validation,
and a GPT-4o-mini snapshot pin without adding a CARES harm-level assumption.

The current evaluator is `gpt-4o-mini-2024-07-18` following the user's correction.
Model choice is a run setting, not part of the original PCSA scoring template.
Historical GPT-4o evaluation artifacts keep their recorded model identity.
