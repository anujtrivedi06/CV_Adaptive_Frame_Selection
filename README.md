# Adaptive Frame Selection for Efficient Video Understanding

Computer Vision course project. Given a video of `N` frames, a **frozen** downstream
model, and a budget of `K << N` frames, which `K` frames should we feed it — and how
much accuracy does a task-aware choice recover over uniform sampling at identical
compute cost?

The insight the whole project rests on: the `K` *most relevant* frames are usually
near-duplicates of each other, drawn from one short window. Useful selection balances
**relevance, diversity and temporal coverage** under a budget. It is a subset-selection
problem, not a ranking problem.

See [`CV_Project_Proposal_Adaptive_Frame_Selection.pdf`](CV_Project_Proposal_Adaptive_Frame_Selection.pdf)
for the full proposal.

---

## Quickstart (CPU, no downloads, ~2 minutes)

```bash
pip install -r requirements.txt
python scripts/make_toy_dataset.py --root data/toy_order --n-videos 64 --mode order
python scripts/build_features.py  --config configs/toy_order.yaml
python scripts/run_eval.py        --config configs/toy_order.yaml --sweep
python scripts/plot_pareto.py     --runs runs/toy_order
```

That runs the entire pipeline — decode, cache, select, read, score, plot — with a
synthetic dataset and a mock reader, so nothing needs a GPU, a download, or
`transformers`. Figures land in `runs/toy_order/figures/`.

Run the tests with `pytest` (28 tests, a few seconds).

---

## How it fits together

```
video ──decode(1 fps)──> N candidate frames
                            │
                      cheap encoder (SigLIP / colour)      ← cached to disk, once
                            │
                   frame features [N,d] + query vector [d]
                            │
                        SELECTOR  ← the research
                            │
                      K frames (K << N)
                            │
                   FROZEN reader (Qwen2.5-VL / mock)
                            │
                  answer + accuracy + GFLOPs + latency
```

Two rules keep the results interpretable:

1. **The selector never sees the reader**, and the reader is never fine-tuned. Every
   accuracy difference between runs is caused by selection alone.
2. **Features are cached once.** Decoding and encoding is the only slow step; after it,
   a full selection experiment takes seconds. This is what makes a wide sweep feasible
   on a student budget.

### Layout

| Path | What it is |
|---|---|
| `afs/decode.py` | video → frames at a fixed fps (PyAV, OpenCV fallback) |
| `afs/encoders/` | `color` (cheap, CPU, no deps), `hf_clip` (SigLIP/CLIP) |
| `afs/features.py` | the feature cache — decode+encode once, reuse forever |
| `afs/selectors/baselines.py` | uniform, random, stride, top-K |
| `afs/selectors/submodular.py` | **our method**, plus MMR for comparison |
| `afs/selectors/adaptive.py` | **per-video adaptive budget** |
| `afs/selectors/oracle.py` | the upper bound (exact on toy, greedy on real data) |
| `afs/readers/` | `needle`/`similarity` mocks, `hf_vlm` (Qwen2.5-VL) |
| `afs/eval.py` | the runner |
| `afs/metrics.py` | accuracy, frames, GFLOPs, latency, evidence recall |
| `configs/` | `toy.yaml`, `toy_order.yaml`, `nextqa.yaml` |

Add a selector by subclassing `Selector` and decorating with
`@SELECTORS.register("my_name")`; it is then usable as `selector.name=my_name`.

---

## The method

Over subsets `S` with `|S| <= K`, maximise

```
F(S) = α · Σ_{i∈S} r_i                           relevance   (modular)
     + β · Σ_j max_{i∈S} s_ij                    semantic coverage  (facility location)
     + γ · Σ_b max_{i∈S} k(t_i, b)               temporal coverage  (facility location)
```

Each term is monotone and submodular, so `F` is too, and greedy forward selection
attains at least `1 − 1/e ≈ 0.632` of the optimum under the cardinality constraint — a
guarantee plain top-K ranking does not have. Training-free, `O(K·N²)`, microseconds on
CPU for `N ≤ 256`.

Implementation note: all three terms are rescaled to `[0,1]` before weighting
(relevance by `K`, each coverage term by what the full frame set would cover). Without
that the terms sit on wildly different scales and α/β/γ stop being meaningful. Rescaling
by positive constants preserves submodularity, so the guarantee is unaffected.

### Per-video adaptive budget

Every published selector fixes `K`. We predict a difficulty score per question from the
relevance distribution alone — entropy, top-1 margin, temporal spread, all free from the
cached features — and spend 2 frames where 2 suffice and 24 where they do not, subject to
`mean(K_i) = K_target`. Total compute is unchanged, so any win is a pure reallocation win.

```bash
python scripts/run_eval.py --config configs/toy_order.yaml \
    budget.adaptive=true budget.K=8 budget.K_min=2 budget.K_max=24
```

---

## The two toy datasets

`mode=single` — one coloured square in one window, "what colour is it?". Any selector
that can score relevance solves this at K=1. It tests the harness, not the method.

`mode=order` — **two** squares of different colours in two well-separated windows,
"which appears first?". Answering needs evidence from *both* windows, so a budget spent
on near-duplicate frames from one window is wasted. This is the setting that actually
tests our claim, and it doubles as the proposal's synthetic needle probe.

Because the ground-truth evidence windows are known, we also report **evidence recall**
(did the selection cover every required window?) alongside accuracy — the metric that
isolates selection quality from reader quality.

---

## What the toy already shows

Accuracy on `toy_order`, 64 videos, 120 candidate frames each, mock reader
(reproduced by the Quickstart above):

| selector | K=2 | K=4 | K=8 | K=16 | K=32 |
|---|---|---|---|---|---|
| uniform | 0.266 | 0.266 | 0.312 | 0.484 | **1.000** |
| random | 0.250 | 0.250 | 0.266 | 0.406 | 0.656 |
| top-K (relevance only) | 0.312 | 0.484 | **1.000** | 1.000 | 1.000 |
| MMR | 0.562 | 0.562 | **1.000** | 1.000 | 1.000 |
| **submodular (ours)** | **0.562** | **0.562** | **1.000** | 1.000 | 1.000 |
| *oracle (ceiling)* | *1.000* | *1.000* | *1.000* | *1.000* | *1.000* |

Three things worth saying out loud:

- **Uniform needs 32 frames to do what we do with 8** — a 4× compute reduction at equal
  accuracy, which is the headline the project is aiming at.
- **Relevance alone is not enough.** At the tightest budget (K=2) top-K scores 0.312
  while diversity-aware selection scores 0.562. It collapses into one event window,
  exactly as predicted.
- **The oracle is at 1.000 from K=2**, so even our method leaves ~44 points on the table
  at K=2. That gap is the project's actual target, and it is the most informative number
  in the table.

Our objective and MMR tie here. On a toy this clean they should; separating them is a job
for real features.

### Already-tuned, and an honest caveat

The temporal weight γ matters and interacts with the budget:

| γ | K=2 | K=4 | K=8 |
|---|---|---|---|
| 0.00 | 0.484 | 0.484 | 1.000 |
| **0.25** | **0.547** | **0.578** | **1.000** |
| 0.50 | 0.547 | 0.531 | 0.828 |

γ=0.25 is the default. Note γ=0.5 actively *hurts* at K=8: too much temporal pressure
drags the selector back towards uniform spacing. Retune on real features before trusting
these values.

The adaptive budget allocator works — at `K_target=8` it spends between 6 and 12 frames
per question (std 2.5) — but it changes nothing on the toy, because every toy question is
equally hard by construction. There is no difficulty variation to exploit. Its real test
is NExT-QA.

---

## Moving to real data

1. Download NExT-QA into `data/nextqa/` as `videos/<id>.mp4` plus `val.csv`.
2. `pip install transformers accelerate`
3. `python scripts/build_features.py --config configs/nextqa.yaml` (slow, once)
4. `python scripts/run_eval.py --config configs/nextqa.yaml --sweep`

A GPU is needed for the `hf_vlm` reader; everything else runs on CPU. Use
`reader.name=similarity` for a CPU-only sanity check on real data before the VLM is
available. Keep `dataset.limit` fixed from the first run onwards — the question subset
must not move between experiments.

**Before the report:** `reader.gflops_per_frame` is currently a placeholder (600). Measure
it properly (e.g. with `fvcore`) or the compute axis of every Pareto plot is fiction.

---

## Status

- [x] **Phase 1** — decoding, feature caching, dataset loaders, configs, tests
- [x] **Phase 2** — baselines, oracle, evaluation harness, Pareto plots
- [x] **Phase 3** — submodular selector (tuned on toy; needs retuning on real features)
- [x] **Phase 4** — adaptive budget allocator (implemented; untested on real difficulty variation)
- [ ] **Phase 5** — ablations and per-category analysis on NExT-QA
- [ ] **Phase 6** — CVPR-format report and demo

Immediate next step is the NExT-QA download and feature cache; nothing else is blocked on it.
