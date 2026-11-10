# ALCHEMI v0.3.0: CPU distillation workflow audit

Hands-on checks of the **actual NVIDIA ALCHEMI Toolkit v0.3.0 release**, pinned to `77d3c9e930a3eab5f44b61a7ffaf161609e13be2`. Run date: November 9, 2026.

This is a workflow and reproducibility audit, **not a materials-accuracy benchmark**. The example uses randomly initialized toy neural networks and random atomic configurations. There are no DFT labels, pretrained foundation-model runs, or GPU speed claims here.

## What ran

| Check | Observed result |
|---|---|
| Unmodified offline example | 48 six-atom systems, 60 steps over 5 epochs; mean first/last epoch training loss 0.341538 -> 0.003355 |
| Instrumented offline example, repeated twice | Identical numeric losses and diagnostics in this CPU environment |
| Teacher calls during offline training | 0, checked with a real forward hook after pre-labeling |
| Parameter snapshots | Teacher unchanged, student changed |
| Separate synthetic holdout | Teacher-consistency MSE decreased for all three requested quantities; details below |
| Unmodified on-policy example | 12 steps over 3 segments, 60 replay frames, generated/reference composition (2, 2); first/last segment training loss 0.2493 -> 0.0675 |
| Bounded upstream test selection | 1,089 passed, 70 skipped, **1 failed** |

The training losses are composed objectives, not energy or force accuracy in physical units. In the offline example the three loss weights are normalized: energy 1/2.2, forces 1/2.2, per-atom energy 0.2/2.2. A falling training objective alone would be insufficient evidence that distillation generalizes.

## Separate holdout check

The audit adds 24 new six-atom systems with seed 133, distinct from the example's 48-system training set (seed 33). It evaluates the same toy teacher and student before and after training. The holdout is not passed to the optimizer, and it is not used to select a model or change hyperparameters.

| Teacher-consistency MSE | Before | After |
|---|---:|---:|
| Total energy, mean over systems | 2.572971 | 0.00549235 |
| Forces, mean over all Cartesian components | 0.03890484 | 0.00123325 |
| Atomic energies, mean over atoms | 0.07421641 | 0.000986252 |

These are errors against a synthetic teacher, in its arbitrary output scale, not reference chemistry. Teacher force predictions come from an independent force head rather than the negative energy gradient. These results do not establish a conservative force field, MD stability, physical correctness, extrapolation, or DFT accuracy. The holdout is now observed; future tuning must not call it untouched.

## Reproduce

Python 3.11 is required for this recorded environment. No GPU, account, paid service, external model, or external dataset is required.

```bash
git clone https://github.com/NVIDIA/nvalchemi-toolkit.git upstream
cd upstream
git checkout 77d3c9e930a3eab5f44b61a7ffaf161609e13be2
cd ..
uv venv --python 3.11 .venv
uv pip install --index-strategy unsafe-best-match --python .venv/bin/python -r requirements.lock.txt
.venv/bin/python audit_distillation.py --source upstream --output results/rerun.json
.venv/bin/python upstream/examples/intermediate/10_onpolicy_distillation.py
```

The script verifies a clean checkout of the exact release commit. It parses the original offline example and inserts an observation callback immediately before its original training call. It does not replace the model, loss, labeling implementation, data pipeline, optimizer, or training call. It checks complete teacher/student parameter snapshots, counts teacher calls during training, evaluates the separate holdout, and repeats the numeric experiment. Standard output can differ because Warp initializes once; numeric diagnostics and the complete loss trace must match exactly within the run. Cross-hardware or cross-version bitwise identity is not claimed.

Recorded versions: Python 3.11.17, PyTorch 2.14.1+cpu, nvalchemi-toolkit 0.3.0, toolkit-ops 0.5.0, PhysicsNeMo 2.2.2, Warp 1.19.0, pytest 8.4.2. Full dependency snapshot: `requirements.lock.txt`. The install command allows uv to choose pinned versions across the two explicitly listed trusted package indexes (PyPI and the PyTorch CPU wheel index); its default first-index strategy otherwise cannot find some recorded versions. A second fresh environment installed from this snapshot reproduced every numeric diagnostic exactly. The audit also byte-compares all 316 installed Python source files against the pinned release checkout. CUDA availability was false and the CUDA device count was zero; Warp also reported no CUDA driver. The installed version is built from the release source, not the moving default branch.

### Upstream tests and non-green results

```bash
cd upstream
../.venv/bin/python -m pytest test/training/distillation/ \
  test/training/test_peft_lora.py \
  --ignore=test/training/distillation/test_multi_gpu.py \
  -q -o addopts='' --junitxml=../results/upstream-suite.xml
```

This selected run finished with **1,089 passed, 70 skipped, 1 failed**. It is not the full toolkit suite. The LoRA test file includes mocks, registration tests and real PhysicsNeMo linear-adapter tests; passing it is not a pretrained MACE LoRA training result. Optional e3nn/model/GPU-dependent coverage was skipped.

The failure was `TestOnPolicyLabelingOverhead.test_labeling_overhead_stays_within_budget`: host capture plus labeling measured 5.00x bare generation against a <4.0 assertion. Three isolated repeats also failed at 5.46x, 5.82x and 5.13x. These short CPU timings are environment-dependent and are not a GPU performance comparison. They are preserved as a reproducible limitation, not silently excluded.

An earlier attempt used pytest 9.2.0. Its accuracy test rejected `pytest.approx` on a mapping containing string metadata (`TypeError` on `accuracy`). Pinning pytest 8.4.2 removed that harness incompatibility. The first broad attempt was interrupted at the execution deadline; its partial output is not counted as a completed suite. See `results/pytest9-accuracy.log`, `results/upstream-suite.log`, and `results/timing-repeat-*.log`. Warnings include missing pytest-asyncio configuration support and optional equivariant wrappers.

## Takeaway

The most interesting part for me is the move beyond a frozen training dataset. The on-policy loop can generate configurations visited by the student, label them with the teacher, and mix them with a reference set through replay. The release's CPU examples demonstrate that plumbing working end to end. The next scientific question is whether a distilled student preserves energy/force accuracy and trajectory stability on a meaningful out-of-distribution materials task, at a measured cost advantage. This audit does not answer that question.

Not tested: pretrained MACE fine-tuning, NVIDIA GPU kernels or throughput, batched NEB, enhanced sampling, multi-GPU behavior, independent DFT validation, or production robustness. The upstream project currently asks users not to submit direct public code contributions during beta; this is an independent reproducibility repository, not an upstream PR or an NVIDIA endorsement.

## Sources

- Release source and documentation: https://github.com/NVIDIA/nvalchemi-toolkit
- Exact tag and source hashes are recorded in `results/audit.json`; the audit uses its original `examples/intermediate/09_offline_distillation.py`.
- Offline example: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/intermediate/09_offline_distillation.py
- On-policy example: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/intermediate/10_onpolicy_distillation.py
- Upstream contribution policy: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/CONTRIBUTING.md

## License

Audit code: MIT, see `LICENSE`. NVIDIA release sources remain under their upstream Apache-2.0 license and are fetched separately, not copied into this repository.
