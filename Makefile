.RECIPEPREFIX = >
# ---------------------------------------------------------------------------
# POSIX shell pin.
# Recipes below use POSIX constructs (test -f, case/esac, $(...), &&/||).
# From cmd.exe / PowerShell, make cannot find sh.exe on PATH and silently falls
# back to cmd.exe, which then fails on every recipe. Pinning SHELL makes the
# Makefile behave identically from Git Bash, cmd.exe and PowerShell.
# On non-Windows hosts this block is skipped and make uses its default /bin/sh.
# Change the path below if Git for Windows is installed elsewhere.
# ---------------------------------------------------------------------------
ifeq ($(OS),Windows_NT)
SHELL := D:/Program Files/Git/usr/bin/sh.exe
endif

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest
STAGE ?= env
DATASET ?= CodeObf-PI_v1.0
GATE ?= G3

.PHONY: help check test smoke freeze audit

help:
> @echo "make check                                      governance files and seal status"
> @echo "make test                                       unit and interface tests"
> @echo "make smoke STAGE=env|data|baseline|train|eval"
> @echo "make freeze DATASET=CodeObf-PI_v1.0"
> @echo "make audit GATE=G3|ASR_KAPPA"

# Frozen probe uses -s (non-empty), not -d. The placeholder directories exist from day
# one with a 0-byte md5sums.txt, so "test -d" reported "frozen OK" before any freeze had
# happened. Only a non-empty md5sums.txt, written by 02_08_freeze_dataset.py, is evidence
# that the dataset was actually frozen. Raised by W1 in WINDOW_LOG.md 2026-09-28 (I-3).
check:
> @echo "== check: governance files, frozen data, seal status =="
> @test -f PREREG_v1.md && echo "PREREG OK" || echo "PREREG MISSING"
> @test -f SOURCE_LOCK.md && echo "SOURCE_LOCK OK" || echo "SOURCE_LOCK MISSING"
> @test -f ENV_LOCK.md && echo "ENV_LOCK OK" || echo "ENV_LOCK MISSING"
> @test -s data/frozen/CodeObf-PI_v1.0/md5sums.txt && echo "frozen OK" || echo "frozen NOT YET"
> @test -s data/sealed_test/CodeObf-PI_v1.0/md5sums.txt && echo "sealed_test OK" || echo "sealed_test NOT YET"
> @echo "== check: script probes (skipped gracefully while not implemented) =="
> @if [ -f scripts/env/01_01_check_env.py ]; then $(PYTHON) scripts/env/01_01_check_env.py; else echo "01_01_check_env.py not generated, skip"; fi
> @if [ -f scripts/data/02_07_assert_splits.py ]; then $(PYTHON) scripts/data/02_07_assert_splits.py --check-only; else echo "02_07_assert_splits.py not generated, skip"; fi

test:
> @echo "== test: unit and interface tests =="
> $(PYTEST) -q tests

smoke:
> @echo "== smoke: STAGE=$(STAGE) =="
> @case "$(STAGE)" in \
>   env) $(PYTHON) scripts/env/01_00_smoke_env.py ;; \
>   data) $(PYTHON) scripts/data/02_00_smoke_data.py ;; \
>   baseline) $(PYTHON) scripts/baseline/03_00_smoke_baseline.py ;; \
>   train) $(PYTHON) scripts/train/04_00_smoke_train.py ;; \
>   eval) $(PYTHON) scripts/eval/05_00_smoke_eval.py ;; \
>   *) echo "unknown STAGE=$(STAGE), expected: env/data/baseline/train/eval"; exit 1 ;; \
> esac

freeze:
> @echo "== freeze: DATASET=$(DATASET) =="
> @if [ -z "$(DATASET)" ]; then echo "DATASET must not be empty"; exit 1; fi
> $(PYTHON) scripts/data/02_08_freeze_dataset.py --dataset $(DATASET)

audit:
> @echo "== audit: GATE=$(GATE) =="
> @case "$(GATE)" in \
>   G3) $(PYTHON) scripts/data/02_09_audit_g3.py ;; \
>   ASR_KAPPA) $(PYTHON) scripts/asr/05_25_audit_kappa.py ;; \
>   *) echo "unknown GATE=$(GATE), expected: G3/ASR_KAPPA"; exit 1 ;; \
> esac
