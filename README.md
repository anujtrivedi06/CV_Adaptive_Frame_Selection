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
| `afs/data/nextgqa.py` | NExT-GQA loader — questions **plus** ground-truth grounding |
| `afs/data/toy.py` | the two synthetic datasets |
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

## Getting NExT-GQA (the real dataset)

We use **NExT-GQA**, not plain NExT-QA. Same multiple-choice questions, but each one
also carries the *time spans that actually contain the answer*. For a frame-selection
project that is a large win:

- the **oracle is exact**, not a greedy approximation that burns reader calls;
- we can measure **evidence recall** — did the selector find the right moment? — which
  isolates selection quality from the reader's reasoning;
- so selection can be evaluated **with no GPU at all**, via the grounding-based reader.

The val split is 3,358 questions over 567 videos, across five question types
(causal-why, causal-how, temporal-next/current/previous) — the per-category breakdown
the report needs comes for free.

```bash
# 1. pick a data directory OUTSIDE any synced folder (the videos are ~6.6 GB)
#    PowerShell:  $env:AFS_DATA = "$HOME\Documents\cv_data"
#    bash:        export AFS_DATA=~/cv_data

# 2. annotations (~2 MB)
python -c "from huggingface_hub import snapshot_download as d; import os; \
d(repo_id='shuzhig/nextgqa', repo_type='dataset', allow_patterns=['annotations/*'], \
  local_dir=os.path.join(os.environ['AFS_DATA'],'nextgqa'))"

# 3. videos (~6.6 GB, resumable — just re-run if it drops)
python -c "from huggingface_hub import snapshot_download as d; import os; \
d(repo_id='shuzhig/nextgqa', repo_type='dataset', allow_patterns=['videos/*'], \
  local_dir=os.path.join(os.environ['AFS_DATA'],'nextgqa'), max_workers=8)"

# 4. cache features once (CPU: ~90 min for the full val split; minutes on a GPU)
python scripts/build_features.py --config configs/nextgqa.yaml

# 5. the sweep — CPU only, no VLM needed
python scripts/run_eval.py   --config configs/nextgqa.yaml --sweep
python scripts/plot_pareto.py --runs runs/nextgqa
```

`${AFS_DATA}` is expanded inside configs, so nobody's absolute path is committed.

### Then add the real reader (needs a GPU)

```bash
pip install accelerate
python scripts/run_eval.py --config configs/nextgqa.yaml --sweep reader.name=hf_vlm
```

Keep `dataset.limit` fixed from the first run onwards — the question subset must not
move between experiments.

**Before the report:** `reader.gflops_per_frame` is a placeholder (600). Measure it
properly (e.g. with `fvcore`) or the compute axis of every Pareto plot is fiction.

### Notes from setting this up

- The SigLIP tokenizer needs `sentencepiece`; it is not pulled in automatically.
- `transformers` v5 changed `get_image_features` to return an output object instead of
  a tensor, and renamed `torch_dtype` to `dtype`. `afs/encoders/hf_clip.py` handles both
  v4 and v5, so either works.
- Measured SigLIP-base throughput on 4 CPU threads: **0.219 s/frame**.

---

## Results on NExT-GQA (real data)

Held-out reporting split, n=250 questions, SigLIP-base features at 2 fps, median 70
candidate frames, grounding-based reader. γ was tuned on the **disjoint** dev split.
`±` is one binomial standard error.

| selector | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| random | 0.420 | 0.500 | 0.616 | 0.736 | 0.860 |
| top-K (relevance only) | 0.440 | 0.524 | 0.624 | 0.756 | 0.876 |
| MMR | 0.440 | 0.532 | 0.652 | 0.776 | 0.916 |
| **uniform** | 0.416 | **0.532** | **0.696** | 0.872 | **0.968** |
| **submodular (ours)** | **0.444** | **0.556** | **0.708** | **0.892** | 0.952 |
| *oracle (ceiling)* | *1.000* | *1.000* | *1.000* | *1.000* | *1.000* |

Standard error is ≈0.03 throughout, so read the table with that in mind.

**What we can claim.** Against the other query-conditioned selectors our objective wins
clearly and consistently — at K=8 it beats random by +0.156 (4.6 s.e.), top-K by +0.136
(4.1 s.e.) and MMR by +0.116 (3.5 s.e.). The diversity-plus-coverage formulation is doing
real work relative to ranking by relevance.

**What we cannot claim.** Submodular selection is **not** distinguishable from plain
uniform sampling: the gap is +0.028, +0.024, +0.012, +0.020, −0.016 at K=1…16, every one
inside one standard error. On this dataset, uniform is a genuinely strong baseline and we
do not beat it.

**Why — measured, not guessed.**

1. *The query signal is weak.* Ranking evidence frames above non-evidence frames by
   SigLIP similarity gives **AUC 0.590** (0.5 would be useless). SigLIP aligns images with
   object nouns; NExT-GQA asks causal and temporal questions ("*why* did the boy pick up
   the present"), which it largely cannot represent. Setting α=0 — ignoring the question
   entirely — costs only ~5 points, which says the same thing from the other side.
2. *Evidence is not sparse.* The grounded spans cover ~23% of each video, so a single
   random frame lands in evidence ~23% of the time and uniform coverage is already a
   near-optimal strategy. There is little for a smarter selector to exploit.
3. *Videos are short.* Median 35 s. Frame selection is a long-video problem, and this
   benchmark is not a long-video benchmark.

**The oracle sits at 1.000 at every budget, including K=1.** One perfectly chosen frame
answers every question, while the best real selector gets 0.444 there. The entire project
lives in that gap, and it is not a modelling-capacity gap — it is a *retrieval* gap caused
by (1).

### Per question type, K=4

| type | uniform | top-K | MMR | ours |
|---|---|---|---|---|
| causal-why | 0.682 | 0.598 | 0.626 | 0.682 |
| causal-how | 0.676 | 0.730 | 0.676 | 0.703 |
| temporal-current | 0.667 | 0.700 | 0.733 | 0.767 |
| temporal-next | 0.732 | 0.549 | 0.620 | 0.704 |
| temporal-previous | 0.800 | 1.000 | 1.000 | 1.000 |

Note `temporal-previous` has only a handful of questions in this split — that row is noise
and should not be quoted.

### What this implies for the next phase

The bottleneck is the relevance signal, not the selection algorithm. Priorities:

1. **A stronger query encoder.** Caption each frame and match in text space, or use a
   VLM-scored relevance. Raising AUC 0.590 → 0.75 would matter far more than any further
   objective tuning.
2. **A genuinely long-video benchmark** (EgoSchema, LongVideoBench) where N/K is 10–100×
   larger and uniform is no longer near-optimal.
3. Only then revisit the adaptive budget — with difficulty estimates this noisy there is
   nothing reliable to reallocate on.

Reproduce with:

```bash
python scripts/run_eval.py   --config configs/nextgqa.yaml --sweep   # report split
python scripts/run_eval.py   --config configs/nextgqa.yaml --sweep dataset.offset=0   # dev
```

## Status

- [x] **Phase 1** — decoding, feature caching, dataset loaders, configs, tests
- [x] **Phase 2** — baselines, oracle, evaluation harness, Pareto plots
- [x] **Phase 3** — submodular selector (tuned on toy; needs retuning on real features)
- [x] **Phase 4** — adaptive budget allocator (implemented; untested on real difficulty variation)
- [ ] **Phase 5** — ablations and per-category analysis on NExT-QA
- [ ] **Phase 6** — CVPR-format report and demo

Immediate next step is the NExT-QA download and feature cache; nothing else is blocked on it.
