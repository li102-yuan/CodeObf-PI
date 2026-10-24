# RELEASE MANIFEST — CodeObf-PI v1.0 reproducibility package

- built: 2026-10-24 16:23:13
- builder: `build_release_v1.0_20261024-152101.py`
- files: **241**
- size: **105.4 MiB**

## Contents by top-level path

| path | files | size |
|---|---:|---:|
| `(root)` | 7 | 0.0 MiB |
| `PAPER_PACKAGE` | 63 | 2.1 MiB |
| `configs` | 46 | 0.6 MiB |
| `data` | 15 | 99.6 MiB |
| `paper` | 1 | 0.0 MiB |
| `scripts` | 49 | 2.3 MiB |
| `tests` | 60 | 0.9 MiB |

## Deliberately withheld

| item | why |
|---|---|
| `data/sealed_test/**` | sealed evaluation split; releasing it would invalidate the benchmark |
| `data/frozen/CodeObf-PI_v1.0/splits/*/test.csv` | duplicate of the sealed split inside the frozen tree |
| `checkpoints/**` | LoRA adapters are distributed separately (Llama 3.2 Community License) |
| `models/**` (weights) | third-party weights, not redistributable here |
| `logs/**`, `reports/**` | internal run logs and working notes |
| `.workbuddy/**`, `PAPER_PACKAGE/submission/**` | local workspace state and manuscript drafts |

## Safety assertions (must all hold)

1. no path contains `sealed_test` — PASS
2. no `test.csv` under `data/frozen/**/splits/` — PASS
3. no `checkpoints/`, `models/`, `logs/`, `.workbuddy/` — PASS
4. no file above 95 MiB — PASS

## Verify

```bash
sha256sum -c SHA256SUMS
```

> Note: sample payloads in `data/frozen/.../benign/G3.csv` contain **synthetic**
> credential-like strings (`Authorization: Bearer ...`). Their hosts use the
> reserved `.example` TLD, the subjects are placeholders (`svc-*`), and the
> signature segments are fixed literals, not valid HMACs. They are not real secrets.
