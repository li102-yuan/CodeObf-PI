# Data and figure licence

The dataset files under `data/frozen/CodeObf-PI_v1.0/` and the figures under
`PAPER_PACKAGE/figures/` are released under the **Creative Commons Attribution 4.0
International licence (CC BY 4.0)**.

- Licence summary and full legal text: https://creativecommons.org/licenses/by/4.0/
- You may share and adapt the material for any purpose, including commercially,
  provided you give appropriate credit, link to the licence, and indicate whether
  changes were made.

## Attribution

> CodeObf-PI v1.0, Peiyuan Li, Nanjing University of Information Science and
> Technology, 2026. Licensed under CC BY 4.0.

## What is not covered

| item | status |
|---|---|
| the sealed evaluation split | **not distributed**; withheld to preserve the evaluation protocol |
| `test.csv` copies inside `data/frozen/**/splits/` | **not distributed** (same data as the sealed split) |
| LoRA adapters | distributed separately; their use is additionally bound by the **Llama 3.2 Community License** |
| third-party model weights referenced in `models/commit_hashes/` | not redistributed; obtain from the original providers |

## Dual-use statement

This corpus contains adversarial prompts. It exists so that defensive detectors can
be **measured**, and every malicious sample is paired with a benign counterpart built
from the same prefix base so that false-positive behaviour is observable. Using the
material to construct or strengthen attacks is outside the intent of the release.
Where local rules require review of dual-use research, users are responsible for
obtaining it.

## Synthetic credential-like strings

Some benign business-form samples (notably `data/frozen/.../benign/G3.csv`) contain
text that resembles an HTTP authorisation header, e.g. `Authorization: Bearer eyJ...`.
These strings are **synthetic**: the host names use the reserved `.example` TLD, the
subject fields are placeholders such as `svc-ingest`, and the signature segments are
fixed literals rather than valid HMACs. They were generated as realistic-but-inert
context for the benign group and do not correspond to any account or service.
