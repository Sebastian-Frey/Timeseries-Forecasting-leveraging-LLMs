# Stage 6 — LLM-native forecasters

LLM-native forecasters (Prophet, TimeGPT, Moirai) for the paper's RQ1:
*"What is the incremental value of semantic-competition information across
model families?"* The notebooks consume the feature tables produced by
Stage 5 and write metrics CSVs into sibling folders under `PAPER_DATA_ROOT`.

## Scope

This stage contains a **representative, not exhaustive**, subset of the
LLM-forecaster code from the underlying thesis work (see the top-level
`README.md` for the paper / thesis relationship). Only the canonical version
of each experiment is ported; superseded iterative drafts are deliberately
excluded. See the top-level README's *What this repository contains*
section for the full include/exclude list.

## Notebooks

| Target | Upstream source (thesis) | Purpose |
| --- | --- | --- |
| `prophet/01_prophet_univariate.ipynb` | `Univariate_horizon.ipynb` | Prophet univariate baseline, h ∈ {1, 6, 12} |
| `prophet/02_prophet_dtw_multiN.ipynb` | `Dtw_prophet_N.ipynb` | Prophet + DTW neighbours, N ∈ {2, 3, 5, 7, 10, 20} |
| `prophet/03_prophet_semantic_exog.ipynb` | `sem_prophet.ipynb` | Prophet + semantic neighbours (univariate / exog_small / exog_full × h) |
| `timegpt/01_timegpt_dtw_onecall.ipynb` | `DTW_TimeGPT_improved_v2.ipynb` | TimeGPT + DTW (global one-call finetune, log1p, h=6) |
| `timegpt/02_timegpt_semantic_exog.ipynb` | `sem_timegpt_exog_all&small.ipynb` | TimeGPT + semantic neighbours (exog_small & exog_all × h ∈ {1, 6, 12}) |
| `moirai/01_moirai_zero_shot.ipynb` | `Morai_working_zero_shot.ipynb` | Moirai zero-shot + DTW |
| `moirai/02_moirai_finetune_h6.ipynb` | `Morai_improvement_setup.ipynb` | Moirai partial-unfreeze fine-tune, h=6 |
| `moirai/03_moirai_fourier_semantic_ensemble.ipynb` | `Morai_sem_improv_collab.ipynb` | Moirai + Fourier + semantic + ensemble |
| `moirai/04_moirai_variants_ensemble.ipynb` | `Morai_improv2.ipynb` | Moirai variants (neighborlag / auto-adaptive / leakage-free / meta-ensemble) |

> **Chronos-2 — coming soon.** The paper evaluates Chronos-2 alongside
> TimeGPT and Moirai under RQ1. The notebook is being cleaned up and will
> land in `chronos/` in a subsequent update.

### Reading the Moirai notebooks

Each Moirai notebook contains the iterative history that led to the paper's
reported numbers. Every notebook starts with a **Canonical cell for the
paper** banner that names the specific cell whose output appears in the
paper; an inline marker repeats the pointer directly above that cell. The
earlier cells are retained for transparency into the methodological
evolution and are not themselves the reported result.

The paper distinguishes two Moirai configurations:

- **DTW-based Moirai** — canonical cell in
  `moirai/02_moirai_finetune_h6.ipynb` (h = 6 partial-unfreeze fine-tune
  over DTW-neighbour covariates).
- **Semantic-based Moirai** — canonical cell in
  `moirai/03_moirai_fourier_semantic_ensemble.ipynb` (PROD v4.5 over
  top-20 semantic-neighbour panels).

`01_moirai_zero_shot.ipynb` provides the zero-shot baseline reference.
`04_moirai_variants_ensemble.ipynb` is the leakage-free robustness
companion.

## Setup

1. Copy `.env.example` (at the repo root) to `.env` and fill in:
   - `PAPER_DATA_ROOT` — absolute path to the directory holding the paper
     input/output tables (Drive mount path if running in Colab).
   - `NIXTLA_API_KEY` — required for any `timegpt/` notebook.
   - `HF_TOKEN` — required for any `moirai/` notebook.

2. Install the notebook dependencies. Each notebook keeps its original
   `!pip install …` cells inline so it runs on Colab without extra setup.

3. Run notebooks in any order; they are independent given the Stage-5
   feature tables.

## Shared helpers

- `_shared/paths.py` — `ensure_env()` mounts Drive on Colab or loads `.env`
  locally; exposes `DATA_ROOT`, `KEYWORDS_DIR_5`, `KEYWORDS_DIR_20`,
  `DTW_DFS_DIR` and result-dir helpers.
- `_shared/api_keys.py` — `get_nixtla_key()` and `get_hf_token()` resolve
  secrets from `os.environ` first, then fall back to Colab `userdata`.

The module is imported by adding `_shared/` to `sys.path` from each
notebook's bootstrap cell.

## Data requirements (not distributed)

The notebooks expect these artefacts under `PAPER_DATA_ROOT`:

- `keywords_dfs_full_5/` — per-keyword parquet panel with top-5 semantic
  neighbours (semantic exog experiments).
- `keywords_dfs_full_20/` — per-keyword parquet panel with top-20 DTW
  neighbours (Moirai zero-shot / fine-tune).
- `dtw_neighbour_dfs/dtw_{N}_dfs/` — per-keyword DTW-N neighbour panels
  (Prophet / TimeGPT DTW experiments).
- `weekly_aggregated_by_week_keyword.parquet` — Stage-5 base table.
- `weekly_semantic_metrics_weighted_110.parquet` — semantic neighbour CPC
  exog matrix.
- `blacklist.txt` — optional keyword blacklist used by the TimeGPT
  semantic notebook.

The dataset was provided under NDA for the car-rental vertical and is
not redistributable.
