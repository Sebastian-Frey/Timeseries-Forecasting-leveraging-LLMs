# =============================================================================
# 1. SETUP & IMPORTS
# =============================================================================
import os
import json
import torch
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from google.colab import drive

# Mount Drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# =============================================================================
# 2. CONFIGURATION
# =============================================================================

# UPDATE THIS PATH if your files are elsewhere
TENSOR_FOLDER = "/content/drive/MyDrive/Colab Notebooks/paper_gdrive/by-week-keyword/exp2_tensors"

# Path to original data (needed to recreate scaler)
GRAPH_FOLDER_PATH = "/content/drive/MyDrive/Colab Notebooks/master_thesis_gdrive/sebs_keyword_graph_knn"
TIME_SERIES_CSV_PATH = "/content/drive/MyDrive/Colab Notebooks/paper_gdrive/by-week-keyword/final_forecast_ready.parquet"

# Features used in training
FEATURE_COLS = [

    # === Core Operational Metrics ===
    'impressions_sum',
    'cpc_week',

    # === Semantic Similarity Features ===
    'avg_sim_top25_this_week',
    'avg_sim_top25_last_week',
    'n_sim_this_week',
    'n_sim_last_week',

    # === Domain Share (Competitors) ===
    'dom_share_avis',
    'dom_share_avisautonoleggio',
    'dom_share_aviscarsales',
    'dom_share_budget',
    'dom_share_budgetautonoleggio',
    'dom_share_dollar',
    'dom_share_economybookings',
    'dom_share_hertz',
    'dom_share_letsdrive',
    'dom_share_sixt',
    'dom_share_thrifty',

    # === Device Distribution ===
    'n_dev_desktop',
    'n_dev_mobile',
    'n_dev_tablet',

    # === Search Type Distribution ===
    'n_st_branded_search',
    'n_st_generic_search',

    # === DTW Neighbour Features (Graph-based) ===
    'dtw_neighbour_1',
    'dtw_neighbour_2',
    'dtw_neighbour_3',
    'dtw_neighbour_4',
    'dtw_neighbour_5',
    'dtw_neighbour_6',
    'dtw_neighbour_7',
    'dtw_neighbour_8',
    'dtw_neighbour_9',
    'dtw_neighbour_10',
    'dtw_neighbour_11',
    'dtw_neighbour_12',
    'dtw_neighbour_13',
    'dtw_neighbour_14',
    'dtw_neighbour_15',
    'dtw_neighbour_16',
    'dtw_neighbour_17',
    'dtw_neighbour_18',
    'dtw_neighbour_19',
    'dtw_neighbour_20',

    # # === Geographic Features ===
    # 'detected_city',
    # 'detected_country',
    # 'detected_continent',
    'detected_city_id',
    'detected_country_id',
    'detected_continent_id',

    # === Autoregressive / Time-Series Features ===
    'target_lag_1',
    'target_lag_2',
    'target_lag_4',
    'target_lag_12',
    'target_roll_mean_4_lag1',

    # === Semantic PCA Components ===
    'sem_pc_0',
    'sem_pc_1',
    'sem_pc_2',
    'sem_pc_4',
]
TARGET_COL = 'cpc_week'
TEST_WEEKS_LAST = 12

# =============================================================================
# 3. HELPER FUNCTIONS (Scaler & Metrics)
# =============================================================================

def get_fitted_scaler(graph_folder, time_series_path):
    """Recreates the scaler used in training."""
    print("Recreating Scaler...")

    # Load Data
    with open(os.path.join(graph_folder, 'keyword_map.json'), 'r') as f:
        keyword_map = json.load(f)
    df = pd.read_parquet(time_series_path)

    # --- MUST MATCH TRAINING: Geo-Categorical Conversion ---
    df["detected_city"] = df["detected_city"].astype(str).replace('None', 'Unknown').fillna('Unknown')
    df["detected_country"] = df["detected_country"].astype(str).replace('None', 'Unknown').fillna('Unknown')
    df["detected_continent"] = df["detected_continent"].astype(str).replace('None', 'Unknown').fillna('Unknown')

    city_to_id = {name: i for i, name in enumerate(sorted(df["detected_city"].unique()), start=1)}
    country_to_id = {name: i for i, name in enumerate(sorted(df["detected_country"].unique()), start=1)}
    continent_to_id = {name: i for i, name in enumerate(sorted(df["detected_continent"].unique()), start=1)}

    df["detected_city_id"] = df["detected_city"].map(city_to_id).astype("int32")
    df["detected_country_id"] = df["detected_country"].map(country_to_id).astype("int32")
    df["detected_continent_id"] = df["detected_continent"].map(continent_to_id).astype("int32")
    # ------------------------------------------------------

    # Parse Weeks
    if df['week'].dtype == object and df['week'].astype(str).str.contains('-').any():
        parts = df['week'].astype(str).str.split('-', expand=True)
        week_nums = pd.to_numeric(parts[0], errors='coerce')
        years = pd.to_numeric(parts[1], errors='coerce')
        df['week'] = years * 100 + week_nums
    else:
        df['week'] = pd.to_numeric(df['week'], errors='coerce')

    df = df.dropna(subset=['week'])

    # Filter Training Weeks
    weeks = np.array(sorted(df['week'].unique()))
    trainval_weeks = weeks[:-TEST_WEEKS_LAST]
    val_ratio = 0.25
    split_idx = int(len(trainval_weeks) * (1 - val_ratio))
    train_weeks = trainval_weeks[:split_idx]

    print(f"  Fitting on {len(train_weeks)} training weeks...")

    # Build Matrix for Scaler
    df_train = df[df['week'].isin(train_weeks)]
    X_flat = df_train[FEATURE_COLS].values

    # Handle Log1p for Target (Index 1)
    target_idx = FEATURE_COLS.index(TARGET_COL)
    X_flat[:, target_idx] = np.log1p(X_flat[:, target_idx])

    scaler = StandardScaler()
    scaler.fit(X_flat)

    print("  Scaler ready.")
    return scaler, target_idx


def calculate_advanced_metrics(preds, targets, scaler, target_col_idx):
    """
    Unscales tensors and computes per-node (keyword) metrics:
    - avg SMAPE across keywords
    - median SMAPE across keywords
    - std SMAPE across keywords
    - avg RMSE across keywords
    - median RMSE across keywords
    - std RMSE across keywords
    """
    # Ensure CPU numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    preds_flat = preds.reshape(-1, 1)
    targets_flat = targets.reshape(-1, 1)

    # Inverse Scale only on target column
    num_features = scaler.mean_.shape[0]
    dummy_preds = np.zeros((len(preds_flat), num_features))
    dummy_targets = np.zeros((len(targets_flat), num_features))

    dummy_preds[:, target_col_idx] = preds_flat[:, 0]
    dummy_targets[:, target_col_idx] = targets_flat[:, 0]

    real_preds = scaler.inverse_transform(dummy_preds)[:, target_col_idx]
    real_targets = scaler.inverse_transform(dummy_targets)[:, target_col_idx]

    # Reverse Log1p
    real_preds = np.expm1(real_preds)
    real_targets = np.expm1(real_targets)
    real_preds = np.maximum(real_preds, 0.0)

    # Try to reshape into [Time, Nodes] to get per-node metrics
    try:
        P = torch.from_numpy(real_preds.reshape(-1, 1811))  # [T, N]
        A = torch.from_numpy(real_targets.reshape(-1, 1811))  # [T, N]

        # Per-node metrics (average over time)
        node_mse = torch.mean((P - A) ** 2, dim=0)      # [N]
        node_rmse = torch.sqrt(node_mse)                # [N]

        numerator = torch.abs(P - A)
        denominator = (torch.abs(P) + torch.abs(A)) / 2.0
        node_smape = 100 * torch.mean(numerator / (denominator + 1e-8), dim=0)  # [N]

        # Average, median, and standard deviation across keywords
        avg_rmse = torch.mean(node_rmse).item()
        med_rmse = torch.median(node_rmse).item()
        std_rmse = torch.std(node_rmse).item()

        avg_smape = torch.mean(node_smape).item()
        med_smape = torch.median(node_smape).item()
        std_smape = torch.std(node_smape).item()

    except Exception as e:
        print(f"Warning: could not reshape into [T, N] for per-node metrics: {e}")
        # Fallback: treat everything as a flat list (no per-keyword metrics)
        P = torch.from_numpy(real_preds)
        A = torch.from_numpy(real_targets)
        # Global metrics as a fallback; set node-based ones to NaN
        mse = torch.mean((P - A) ** 2).item()
        rmse = np.sqrt(mse)
        numerator = torch.abs(P - A)
        denominator = (torch.abs(P) + torch.abs(A)) / 2.0
        smape = 100 * torch.mean(numerator / (denominator + 1e-8)).item()

        avg_rmse = rmse
        med_rmse = np.nan
        std_rmse = np.nan
        avg_smape = smape
        med_smape = np.nan
        std_smape = np.nan

    return avg_smape, med_smape, std_smape, avg_rmse, med_rmse, std_rmse


# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================

# A. Init Scaler
scaler, target_col_idx = get_fitted_scaler(GRAPH_FOLDER_PATH, TIME_SERIES_CSV_PATH)

# B. File Discovery
print("\n" + "="*80)
print(f"SEARCHING FOR TENSORS IN: {TENSOR_FOLDER}")
print("="*80)

if not os.path.exists(TENSOR_FOLDER):
    print(f"❌ ERROR: Folder not found: {TENSOR_FOLDER}")
    print("Please check the path and try again.")
else:
    all_files = os.listdir(TENSOR_FOLDER)
    pred_files = [f for f in all_files if f.endswith('_predictions.pt')]
    print(f"Found {len(all_files)} total files.")
    print(f"Found {len(pred_files)} prediction files to process.")

    if len(pred_files) == 0:
        print("  (No '_predictions.pt' files found. Did you save them?)")
        print("  First 5 files in folder:", all_files[:5])

    results_list = []

    print(f"\n{'Horizon':<8} | {'Model':<20} | {'Avg SMAPE':<10} | {'Med SMAPE':<10} | {'Std SMAPE':<10} | {'Avg RMSE':<10} | {'Med RMSE':<10} | {'Std RMSE':<10}")
    print("-" * 110)

    for p_file in sorted(pred_files):
        # Infer target filename
        t_file = p_file.replace('_predictions.pt', '_targets.pt')

        if t_file not in all_files:
            print(f"⚠ Missing targets for {p_file}, skipping.")
            continue

        # Parse Model/Horizon from filename
        # Expected format: "{Model}_h{Horizon}_predictions.pt"
        try:
            match = re.search(r'(.+)_h(\d+)_predictions\.pt', p_file)
            if match:
                model_name = match.group(1)
                horizon = int(match.group(2))
            else:
                model_name = p_file
                horizon = 0
        except:
            model_name = p_file
            horizon = 0

        # Load & Calculate
        try:
            p_path = os.path.join(TENSOR_FOLDER, p_file)
            t_path = os.path.join(TENSOR_FOLDER, t_file)

            preds = torch.load(p_path, map_location='cpu')
            targets = torch.load(t_path, map_location='cpu')

            avg_smape, med_smape, std_smape, avg_rmse, med_rmse, std_rmse = calculate_advanced_metrics(
                preds, targets, scaler, target_col_idx
            )

            print(f"{horizon:<8} | {model_name:<20} | {avg_smape:<10.2f} | {med_smape:<10.2f} | {std_smape:<10.2f} | {avg_rmse:<10.4f} | {med_rmse:<10.4f} | {std_rmse:<10.4f}")

            results_list.append({
                'Horizon': horizon,
                'Model': model_name,
                'Avg_Node_SMAPE': avg_smape,
                'Median_Node_SMAPE': med_smape,
                'Std_Node_SMAPE': std_smape,
                'Avg_Node_RMSE': avg_rmse,
                'Median_Node_RMSE': med_rmse,
                'Std_Node_RMSE': std_rmse
            })

        except Exception as e:
            print(f"Error processing {p_file}: {e}")

    # Save
    if results_list:
        df_res = pd.DataFrame(results_list).sort_values(['Horizon', 'Avg_Node_SMAPE'])
        df_res.to_csv("final_detailed_metrics.csv", index=False)
        print("\n✓ Saved metrics to final_detailed_metrics.csv")