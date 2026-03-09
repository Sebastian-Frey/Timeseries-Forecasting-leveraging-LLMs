# =============================================================================
# Ablation Study: Tensors to Detailed Results
# =============================================================================
import os
import json
import torch
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# ROOT directory containing the exp_XXX folders
TENSOR_ROOT = "/content/drive/MyDrive/Colab Notebooks/paper_gdrive/by-week-keyword/tensors"

# Data paths to recreate scalers
GRAPH_FOLDER_PATH = "/content/drive/MyDrive/Colab Notebooks/master_thesis_gdrive/sebs_keyword_graph_knn"
TIME_SERIES_CSV_PATH = "/content/drive/MyDrive/Colab Notebooks/paper_gdrive/by-week-keyword/final_forecast_ready.parquet"

TARGET_COL = 'cpc_week'
TEST_WEEKS_LAST = 12
VAL_RATIO = 0.25
NUM_NODES = 1811

# =============================================================================
# Feature Groups (MUST MATCH ABLATION NOTEBOOK)
# =============================================================================

CORE_FEAT = ['impressions_sum', 'cpc_week']
GEO_FEAT  = ['detected_city_id', 'detected_country_id', 'detected_continent_id']
DTW_FEAT  = [f'dtw_neighbour_{i}' for i in range(1, 21)]
LAG_FEAT  = ['target_lag_1', 'target_lag_2', 'target_lag_4', 'target_lag_12', 'target_roll_mean_4_lag1']

SIM25_FEAT = ['avg_sim_top25_this_week', 'avg_sim_top25_last_week', 'n_sim_this_week', 'n_sim_last_week']

DOM_FEAT = [
    'dom_share_avis', 'dom_share_avisautonoleggio', 'dom_share_aviscarsales', 'dom_share_budget',
    'dom_share_budgetautonoleggio', 'dom_share_dollar', 'dom_share_economybookings', 'dom_share_hertz',
    'dom_share_letsdrive', 'dom_share_sixt', 'dom_share_thrifty',
]

SEARCH_FEAT = ['n_dev_desktop', 'n_dev_mobile', 'n_dev_tablet',
    'n_st_branded_search', 'n_st_generic_search']

SEM_PC_FEAT = [
    'sem_pc_0', 'sem_pc_1', 'sem_pc_2', 'sem_pc_4'
]

EXPERIMENTS = {
    "core_only": CORE_FEAT,
    "core_geo": CORE_FEAT + GEO_FEAT,
    "core_dtw": CORE_FEAT + DTW_FEAT,
    "core_lags": CORE_FEAT + LAG_FEAT,
    "core_sim25": CORE_FEAT + SIM25_FEAT,
    "core_dom": CORE_FEAT + DOM_FEAT,
    "core_search": CORE_FEAT + SEARCH_FEAT,
    "core_sem_pc": CORE_FEAT + SEM_PC_FEAT,
    "all_features": CORE_FEAT + GEO_FEAT + DTW_FEAT + LAG_FEAT + SIM25_FEAT + DOM_FEAT + SEARCH_FEAT + SEM_PC_FEAT
}

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def get_scaler_for_experiment(exp_name: str, graph_folder: str, time_series_path: str):
    """Fits a scaler specifically for the feature set of a given experiment."""
    if exp_name not in EXPERIMENTS:
        raise ValueError(f"Experiment {exp_name} not defined in EXPERIMENTS config.")
        
    features = EXPERIMENTS[exp_name]
    print(f"  → Loading data and fitting scaler for {len(features)} features...")
    
    # Load Data
    df = pd.read_parquet(time_series_path)

    # Preprocessing (Geo IDs)
    df["detected_city"] = df["detected_city"].astype(str).replace('None', 'Unknown').fillna('Unknown')
    df["detected_country"] = df["detected_country"].astype(str).replace('None', 'Unknown').fillna('Unknown')
    df["detected_continent"] = df["detected_continent"].astype(str).replace('None', 'Unknown').fillna('Unknown')

    city_to_id = {name: i for i, name in enumerate(sorted(df["detected_city"].unique()), start=1)}
    country_to_id = {name: i for i, name in enumerate(sorted(df["detected_country"].unique()), start=1)}
    continent_to_id = {name: i for i, name in enumerate(sorted(df["detected_continent"].unique()), start=1)}

    df["detected_city_id"] = df["detected_city"].map(city_to_id).astype("int32")
    df["detected_country_id"] = df["detected_country"].map(country_to_id).astype("int32")
    df["detected_continent_id"] = df["detected_continent"].map(continent_to_id).astype("int32")

    # Parse Weeks
    if df['week'].dtype == object and df['week'].astype(str).str.contains('-').any():
        parts = df['week'].astype(str).str.split('-', expand=True)
        df['week'] = pd.to_numeric(parts[1]) * 100 + pd.to_numeric(parts[0])
    else:
        df['week'] = pd.to_numeric(df['week'])

    # Identify Training Weeks
    weeks = np.array(sorted(df['week'].unique()))
    trainval_weeks = weeks[:-TEST_WEEKS_LAST]
    split_idx = int(len(trainval_weeks) * (1 - VAL_RATIO))
    train_weeks = trainval_weeks[:split_idx]

    # Filter and Apply Log1p
    df_train = df[df['week'].isin(train_weeks)].copy()
    target_idx = features.index(TARGET_COL)
    
    X_flat = df_train[features].values
    X_flat[:, target_idx] = np.log1p(X_flat[:, target_idx])

    scaler = StandardScaler()
    scaler.fit(X_flat)
    
    return scaler, target_idx

def calculate_advanced_metrics(preds, targets, scaler, target_col_idx):
    """Unscales tensors and computes per-node (keyword) metrics."""
    if isinstance(preds, torch.Tensor): preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor): targets = targets.cpu().numpy()

    # Reshape to flat for scaler
    preds_flat = preds.reshape(-1, 1)
    targets_flat = targets.reshape(-1, 1)

    num_features = scaler.mean_.shape[0]
    dummy_preds = np.zeros((len(preds_flat), num_features))
    dummy_targets = np.zeros((len(targets_flat), num_features))

    dummy_preds[:, target_col_idx] = preds_flat[:, 0]
    dummy_targets[:, target_col_idx] = targets_flat[:, 0]

    # Inverse transform
    real_preds = np.expm1(scaler.inverse_transform(dummy_preds)[:, target_col_idx])
    real_targets = np.expm1(scaler.inverse_transform(dummy_targets)[:, target_col_idx])
    real_preds = np.maximum(real_preds, 0.0)

    try:
        # Reshape to [Time, Nodes]
        P = real_preds.reshape(-1, NUM_NODES)
        A = real_targets.reshape(-1, NUM_NODES)

        # Per-node RMSE
        node_rmse = np.sqrt(np.mean((P - A) ** 2, axis=0))

        # Per-node SMAPE
        numerator = np.abs(P - A)
        denominator = (np.abs(P) + np.abs(A)) / 2.0
        node_smape = 100 * np.mean(numerator / (denominator + 1e-8), axis=0)

        return {
            'avg_smape': np.mean(node_smape), 
            'med_smape': np.median(node_smape), 
            'std_smape': np.std(node_smape),
            'avg_rmse': np.mean(node_rmse), 
            'med_rmse': np.median(node_rmse), 
            'std_rmse': np.std(node_rmse)
        }
    except Exception as e:
        print(f"    ! Reshape error: {e}")
        return None

# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================

def main():
    print("="*80)
    print("ABLATION STUDY: POST-PROCESSING TENSORS")
    print("="*80)
    
    results_list = []

    # Iterate over experiment folders
    for exp_name in sorted(EXPERIMENTS.keys()):
        exp_dir = os.path.join(TENSOR_ROOT, f"exp_{exp_name}")
        if not os.path.exists(exp_dir):
            print(f"Skipping {exp_name}: Directory not found at {exp_dir}")
            continue
            
        print(f">>> Processing Experiment: {exp_name}")
        
        try:
            scaler, target_idx = get_scaler_for_experiment(exp_name, GRAPH_FOLDER_PATH, TIME_SERIES_CSV_PATH)
        except Exception as e:
            print(f"  FAILED to initialize scaler for {exp_name}: {e}")
            continue
        
        all_files = os.listdir(exp_dir)
        pred_files = [f for f in all_files if f.endswith('_predictions.pt')]
        
        if not pred_files:
            print("  No prediction tensors found in this directory.")
            continue

        print(f"  Found {len(pred_files)} prediction files.")
        
        for p_file in sorted(pred_files):
            t_file = p_file.replace('_predictions.pt', '_targets.pt')
            if t_file not in all_files:
                print(f"  ! Missing targets for {p_file}, skipping.")
                continue
            
            # Extract model and horizon using regex
            # Expected format: {model}_{exp}_h{H}_predictions.pt
            match = re.search(r'(.+)_' + re.escape(exp_name) + r'_h(\d+)_predictions\.pt', p_file)
            if match:
                model_name = match.group(1)
                horizon = int(match.group(2))
            else:
                # Fallback for unexpected naming
                model_name, horizon = p_file.replace("_predictions.pt", ""), "unknown"

            print(f"    Processing {model_name} (H={horizon})...")
            
            try:
                preds = torch.load(os.path.join(exp_dir, p_file), map_location='cpu')
                targets = torch.load(os.path.join(exp_dir, t_file), map_location='cpu')
                
                metrics = calculate_advanced_metrics(preds, targets, scaler, target_idx)
                
                if metrics:
                    results_list.append({
                        'Experiment': exp_name,
                        'Horizon': horizon,
                        'Model': model_name,
                        **metrics
                    })
            except Exception as e:
                print(f"    ! Error processing {p_file}: {e}")

    # Export Final Results
    if results_list:
        output_file = "ablation_detailed_results.csv"
        df_final = pd.DataFrame(results_list)
        
        # Sort for readability
        df_final = df_final.sort_values(['Experiment', 'Horizon', 'avg_smape'])
        
        df_final.to_csv(output_file, index=False)
        print("" + "="*80)
        print(f"SUCCESS: Results saved to {output_file}")
        print("="*80)
        print(df_final.head(10).to_string(index=False))
    else:
        print("No valid results were processed.")

if __name__ == "__main__":
    main()
