# Competition-Aware CPC Forecasting with Near-Market Coverage — code companion

This repository is the code companion to the paper *"Competition-Aware CPC
Forecasting with Near-Market Coverage"* (arXiv:2603.13059v1, Frey, Beccari,
Kranz, Pellizzari, Karaman, Han, Kaiser, Mar 2026). The paper builds on, and
reframes, an earlier master's thesis completed at Nova SBE in January 2025.

The notebooks here reproduce the paper's data pipeline, feature engineering,
classical-ML baselines, LLM-native forecasters (Prophet, TimeGPT, Moirai), and
spatio-temporal graph neural networks.

## Paper and thesis

- **Paper** — *Competition-Aware CPC Forecasting with Near-Market Coverage*,
  arXiv:2603.13059v1 (13 Mar 2026). This repository is the code companion to
  the paper. The paper describes the final methodology, evaluation design,
  and results in reduced form.
- **Thesis** — *LLM-Augmented Time Series Forecasting for Car Rental Cost per
  Click*, Nova School of Business and Economics, 21 Jan 2025 (Frey, Kranz,
  Beccari, Pellizzari, Karaman; supervised by Prof. Qiwei Han). The thesis is
  the groundwork the paper builds on and contains the full literature review,
  extended methodology, and broader experimental results.

Neither PDF is distributed in this repository. The paper is available on
arXiv; the thesis is archived with Nova SBE.

## What this repository contains

This repo is a **curated, representative subset** of the code base behind the
paper and the thesis. It is designed to make the paper's experiments
reproducible; it is **not** a complete archive of the thesis work, which was
developed in a much larger private workbench with ~50 notebooks of iterative
drafts, exploratory analyses, and variants.

### Included (representative)

| Stage | Purpose | Representative notebooks |
| --- | --- | --- |
| 1 | Raw-log ingestion and weekly aggregation | `create_weekly_data_collab.ipynb` |
| 2 | EDA + keyword-unification / fill proofs | `EDA.ipynb`, `Collapse_proof.ipynb`, `backward_forward_fill_proof.ipynb` |
| 3 | Classical baselines (RF, XGB, LGBM, MLP, LSTM, GRU, SARIMAX, TabPFN-TS) | `benchmark_models.ipynb` |
| 4 | Spatio-temporal GNNs on the LLM-derived keyword graph | `model_training_and_tuning.ipynb` + `knn_calculation/*.py` |
| 5 | Combined feature pipeline (semantic + DTW neighbours, domain share, geo augmentation, feature engineering) | `01_data_creation_base.ipynb` → `feature engineering.ipynb` |
| 6 | LLM-native forecasters (Prophet, TimeGPT, Moirai) | `prophet/*`, `timegpt/*`, `moirai/*` |

### Not included (intentionally)

- Iterative drafts superseded by the canonical versions
  (`sem_prophet_improved.ipynb`, `dtw_prophet_improved.ipynb`,
  `DTW_timegpt.ipynb`, `morai_improvement.ipynb`, `morai_sem_improved.ipynb`,
  `sem_timgpt_improv.ipynb`, and several others).
- Chronos / Chronos-2 baseline notebooks — **coming soon.** The paper
  evaluates Chronos-2 alongside TimeGPT and Moirai; the corresponding
  notebook is being cleaned up and will be added to
  `6_LLM_forecasters/chronos/` in a subsequent update.
- BasicTS GNN experiments (`basicts.ipynb`, `tryts.ipynb`, `basicTS_4models.ipynb`)
  that informed the thesis but are superseded in the paper by the
  `torch_geometric_temporal` stack used in Stage 4.
- Results-aggregation notebooks used to assemble the thesis and paper figures
  (`results_sd.ipynb`). The raw metrics CSVs needed to regenerate the figures
  are proprietary.
- Raw or intermediate data artefacts. The dataset was provided by Grips
  Intelligence for the car-rental vertical (2021-2023, ~1.66 billion log-level
  observations, 1,811 keyword panel after filtering) and is not
  redistributable.

## Research questions

- **RQ1** — What is the incremental value of *semantic-competition*
  information (neighbour CPC signals) and recency-sensitive weighting for
  CPC-forecast performance, and how does that value differ across model
  families (Prophet, TimeGPT, Moirai)?
- **RQ2** — To what extent can Dynamic-Time-Warping-based neighbour selection
  improve keyword-level CPC forecasting by integrating CPC signals from
  temporally similar keywords, compared to models that rely solely on
  autoregressive CPC histories?
- **RQ3** — Can keyword-level CPC forecasting be improved by modelling
  *localised competitive dynamics* via LLM-extracted geographical intent?
- **RQ4** — Does incorporating an LLM-derived semantic keyword graph into
  spatio-temporal graph neural networks improve CPC forecasting accuracy and
  robustness across multiple horizons, compared to non-graph baselines and
  purely data-driven graph-learning approaches?
- **RQ5** — To what extent does explicitly modelling *cross-domain competition*
  through a fixed binary graph improve forecasting accuracy when using
  spatio-temporal GNNs, compared to models without an explicit competitive
  structure?

## Pipeline

| Stage | Folder | Purpose | Addresses |
| --- | --- | --- | --- |
| 1 | `1_Data_processing/` | Raw Google Ads parquet → weekly, keyword-normalised table (FX-adjusted, device/search-type pivots). | all |
| 2 | `2_EDA/` | Descriptive statistics and methodological "proof" notebooks for keyword unification and backward/forward fill. | all |
| 3 | `3_Benchmarks/` | Classical baselines: RandomForest, XGBoost, LightGBM, MLP, LSTM, GRU, SARIMAX(1,1,0), TabPFN-TS zero-shot. Horizons 1/6/12 weeks × three exogenous modes. | RQ1, RQ2 |
| 4 | `4_GNN benchmarking/` | Spatio-temporal GNNs (DCRNN, STGCN, A3TGCN, GConvLSTM, STConv, MTGNN, AGCRN, GraphWaveNet) on an LLM-derived KNN keyword graph. Includes the graph-construction scripts. | RQ4, RQ5 |
| 5 | `5_Combined_approach/` | Full feature pipeline used by the combined LLM-augmented model: semantic + DTW neighbours, domain market shares, geo augmentation, feature engineering with explicit leakage controls. | RQ1, RQ2, RQ3, RQ5 |
| 6 | `6_LLM_forecasters/` | LLM-native forecasters (Prophet, TimeGPT, Moirai) consuming Stage-5 feature tables. Ported from the thesis workbench with unified data paths and `.env`-based secret handling. | RQ1, RQ2 |

### Data flow

```
 raw daily parquet shards  (Google Drive, ~4,410 files)
            │
            ▼
 1_Data_processing/create_weekly_data_collab.ipynb
            │    ISO-week aggregation, keyword unification, FX-adjusted cost
            ▼
 weekly_processed_kn.parquet
            │
            ├─► 5_Combined_approach/01_data_creation_base.ipynb
            │        → weekly_aggregated_by_week_keyword.parquet
            │
            ├─► 5_Combined_approach/03_data_creation__domainshare.ipynb
            │        → weekly_aggregated_by_week_domain_keyword.parquet
            │
            └─► 5_Combined_approach/02_data_creation_dtw_sem_neighbour.ipynb
                     sentence-transformers (all-MiniLM-L6-v2) + DTW (tslearn)
                     → domain_sem_dtw.parquet  (+40 neighbour-CPC features)

 domain_sem_dtw.parquet
            │
            ▼
 5_Combined_approach/geo_augmentation.ipynb
            │    LLM / regex extraction of city, country, continent
            ▼
 weekly_aggregated_geo_enriched.parquet
            │
            ▼
 5_Combined_approach/feature engineering.ipynb
            │    lag-1 of every research feature, AR lags (1,2,4,12),
            │    GPU-SVD PCA on neighbour-CPC, rolling-mean SMAPE stabiliser
            ▼
 final_forecast_ready.parquet    (76 columns, forecast-ready)
            │
            ├─► 3_Benchmarks/benchmark_models.ipynb
            ├─► 4_GNN benchmarking/model_training_and_tuning.ipynb
            │        consumes KNN graph from
            │        4_GNN benchmarking/knn_calculation/
            └─► 6_LLM_forecasters/{prophet,timegpt,moirai}/*.ipynb
```

## Keyword graph construction (Stage 4)

Two builders live in `4_GNN benchmarking/knn_calculation/`:

- `build_kw_graph_nw_v3_knn.py` — top-K semantic neighbours per keyword
  (K ∈ {3, 5, 7, 10, 20}). **This is the variant used by the GNN notebook.**
- `build_kw_graph_nw_v3_multi_threshold_weighted.py` — similarity-threshold
  variant (τ ∈ {0.6, 0.7, 0.8}) with edge weights, kept for comparison in the
  paper.

Both emit `edge_index.npy`, `edge_weight.npy`, `keyword_map.json` and a
`.gexf` graph for inspection.

## Running the code

Every notebook is authored for **Google Colab** and expects its data under
`/content/drive/MyDrive/colab_data/cleaned_cpu/…`. Stages 1–5 still mount
Drive and read hardcoded paths directly. Stage 6 (`6_LLM_forecasters/`) adds a
`.env`-based indirection: copy `.env.example` to `.env` at the repo root and
set `PAPER_DATA_ROOT` to the directory holding the input tables, then the
notebooks resolve paths through the helpers in `6_LLM_forecasters/_shared/`.

Dependencies are installed inline with `!pip install` in the first cells of
each notebook; there is no `requirements.txt`. Minimum stack:

```
polars, pyarrow, pandas, numpy, scipy, scikit-learn,
xgboost, lightgbm, statsmodels, torch, tabpfn-ts,
sentence-transformers, tslearn, networkx, tqdm, dask,
torch_geometric, torch_geometric_temporal,
nixtla, prophet, cmdstanpy, uni2ts (for Stage 6).
```

Recommended execution order: `1 → 2 → 5.1 → 5.3 → 5.2 → geo_augmentation →
feature engineering → 3 → knn_calculation → 4 → 6`.

### Running the LLM forecasters (Stage 6)

The Prophet / TimeGPT / Moirai notebooks in `6_LLM_forecasters/` require API
keys:

- `NIXTLA_API_KEY` — for any `timegpt/` notebook
- `HF_TOKEN` — for any `moirai/` notebook (to pull the pretrained Moirai
  weights from Hugging Face)

Copy `.env.example` to `.env` at the repo root and fill in your keys. On
Colab the helpers also fall back to `userdata.get(...)` if the env var is
not set. Raw proprietary data is still required.

## Data availability

The raw keyword-level Google Ads data is **not** distributed with this
repository. Intermediate parquet files referenced above are the artefacts a
reviewer would need to re-run the modelling stages; please contact the
author for access under the terms permitted by the data provider.

## Repository status

Stages 1, 2, 4, 5 and 6 are functional. Stage 3 contains a representative
classical-baselines notebook; some placeholder files (`*comingsoon.txt`) in
earlier stages flag sections still being expanded. See `CLAUDE.md` (local
only) for a more detailed internal map.

## Citation

If you use this code or the methodology in academic work, please cite the
paper (arXiv:2603.13059v1) as the primary reference and the master's thesis
as the groundwork reference.
