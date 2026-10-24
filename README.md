# CodeObf-PI v1.0 — reproducibility package

Code and data for the paper *Measuring and Repairing the Encoding Shortcut in
Lightweight Prompt-Injection Detectors for Code*.

The study asks a narrow question: when a prompt-injection payload is **encoded**
(base64/hex, Morse, ASCII art, homoglyphs, …) and delivered inside **code**, does a
lightweight guardrail still detect the *intent*, or does it merely detect the
*encoding*? A controlled corpus crosses nine encodings with two carriers (code and
text) and pairs every malicious prompt with a benign prompt built from the same
prefix base, so the two factors can be separated. The audit finds a shortcut in the
first-generation detector and the release also contains the repaired
(de-confounded) variant and the scripts used to measure it.

## What is in this package

| path | contents |
|---|---|
| `scripts/` | data construction, baselines, QLoRA fine-tuning, evaluation, ASR chain, audits |
| `configs/` | YAML configuration for data, training, evaluation, ASR and environment |
| `tests/` | unit and interface tests (`pytest`) |
| `requirements/` | pinned environment (`requirements-locked.txt`) |
| `Makefile` | the canonical entry points (`make check`, `make test`, `make smoke`, …) |
| `data/frozen/CodeObf-PI_v1.0/` | frozen dataset: train/val splits, benign control groups, data card, manifest, checksums |
| `PAPER_PACKAGE/tables/` | the twelve result tables (`.md` for reading, `.csv` for machine use) |
| `PAPER_PACKAGE/figures/` | the eight figures (PNG 300 dpi, PDF, SVG) and their index |
| `paper/materials/01_numbers_manifest.md` | single source of truth for every number quoted in the paper |

## What is deliberately NOT in this package

| withheld | reason |
|---|---|
| the sealed test split | releasing it would invalidate the benchmark; it is withheld to preserve the evaluation protocol |
| `test.csv` copies inside the frozen tree | same data as the sealed split |
| LoRA adapters (`checkpoints/`) | distributed separately under the Llama 3.2 Community License |
| third-party model weights | not redistributable here |

## Requirements

- Python 3.10 or newer (the recorded environment is 3.12)
- a single GPU with 8 GB of VRAM; 4-bit NF4 quantisation is used for both training and inference
- the two API judges need credentials supplied through **environment variables**
  (`DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`). No key is stored in this repository.

```bash
python -m pip install -r requirements/requirements-locked.txt
make check          # governance / environment probe
make test           # unit + interface tests (does not touch the sealed split)
```

## Data layout

```
data/frozen/CodeObf-PI_v1.0/
  DATA_CARD_CodeObf-PI_v1.0.md   # sampling, splits, licence and known limitations
  manifest.json  md5sums.txt     # integrity anchors
  seed_groups.csv                # family-level grouping used by the split
  benign/G1.csv G2.csv G3.csv    # benign control groups (G3 = business-form hard negatives)
  splits/regime_id/{train,val}.csv
  splits/regime_ood/{train,val}.csv
  reports/*.csv                  # QC artefacts (length, n-gram overlap, split assertions)
```

`md5sums.txt` covers the **complete** frozen set, including the two test splits that
are withheld from this release. To verify the shipped subset, use the filtered
manifest instead:

```bash
cd data/frozen/CodeObf-PI_v1.0 && md5sum -c md5sums_release.txt
```

Nine encodings × two carriers; five encodings are used for training in the
out-of-distribution regime and four are held out for testing. Seeds 42/123/456 are
run independently and never averaged across regimes.

## Reproducing the results

| step | entry point |
|---|---|
| environment probe | `make check` |
| dataset QC | `scripts/data/` |
| detector baselines | `scripts/baseline/` |
| QLoRA fine-tuning | `scripts/train/04_01_train_qlora_ours.py` |
| three-seed runs | `scripts/train/04_03_run_three_seeds.py` |
| detection evaluation | `scripts/eval/` |
| ASR chain and its audit | `scripts/asr/` |

Script names follow `scripts/<layer>/<stage>_<seq>_<name>.py`; the stage numbers run
01–05 and the ASR sub-module uses 05_20–05_29.

## Licences

- code: MIT (see `LICENSE`)
- data and figures: CC BY 4.0 (see `DATA_LICENSE.md`)

## Dual use and ethics

The corpus contains adversarial prompts. It was built to *evaluate defensive
detectors*, and every malicious sample is paired with a benign counterpart so that
false positives can be measured. Reuse for building or improving attacks is outside
the intent of this release. The sealed split is withheld so that future work can be
compared on data that has not been tuned against.

## A note on the data

Sample payloads in `data/frozen/.../benign/G3.csv` contain strings that look like
credentials, for example `Authorization: Bearer eyJ...`. These are **synthetic
placeholders**: hosts use the reserved `.example` TLD, subjects are placeholders such
as `svc-ingest`, expiry timestamps are in the future by construction, and the
signature segment is a fixed literal rather than a valid HMAC. They are not secrets
and were generated for the benign business-form group.

## Citation

See `CITATION.cff`. If you use the dataset, please cite the paper and the archived
release (DOI to be inserted once the archive is minted).

## Contact

Peiyuan Li — Nanjing University of Information Science and Technology
`202483280093@nuist.edu.cn` · ORCID [0009-0008-7982-9597](https://orcid.org/0009-0008-7982-9597)
