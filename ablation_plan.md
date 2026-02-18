# Experiment 2: Feature Ablation Study

### **1. Objective**

The goal of this experiment is to isolate and quantify the marginal contribution of different feature groups to the forecasting performance (SMAPE/RMSE). By systematically training models on restricted feature sets, we can determine if complex covariates (like Graph-based DTW neighbors or Geographic embeddings) provide actual signal or just introduce noise compared to a simple baseline.

### **2. Methodology**

We define a "Base" feature set (Core metrics) and incrementally add specific feature groups. All models are trained from scratch for each configuration.

**Feature Configurations:**

* **`core_only`**: Baseline. Only includes `impressions_sum` and the target `cpc_week`.
* **`core_geo`**: Adds static geographic embeddings (`city_id`, `country_id`, `continent_id`). Tests if explicit location data improves spatial learning.
* **`core_dtw`**: Adds 20 Dynamic Time Warping (DTW) neighbor features. Tests if the graph structure's pre-computed similarities help the GNN.
* **`core_lags`**: Adds autoregressive targets (`lag_1`, `lag_4`, etc.). Tests how much performance is driven by pure time-series memory vs. spatial mixing.
* **`all_features`**: The complete dataset (65+ features) including semantic scores, device distribution, and domain shares.

### **3. Selected Models**

Based on previous benchmarks, we restrict this study to the top-4 performing architectures to save compute time:

1. **DCRNN:** (Diffusion Convolutional RNN) - consistently stable.
2. **GraphWaveNet:** (Dilated Convolution) - best capture of long-range dependencies.
3. **GConvLSTM:** (Graph Convolutional LSTM) - strong temporal performance.
4. **AGCRN:** (Adaptive Graph CRN) - learns node-specific patterns effectively.

### **4. Experiment Settings**

* **Epochs:** Reduced to **50** (with Early Stopping patience=10) for efficiency.
* **Horizons:** Testing **H=1** (Immediate) and **H=12** (Long-term).
* **Output:** Tensors and metrics are saved to separate subfolders (e.g., `tensors/exp_core_geo/`) to prevent overwriting and allow for granular error analysis.

---

### **Step-by-Step Implementation Plan for the Coding Agent**

Below is the checklist for the Coding Agent to execute the changes in the notebook.

#### **Phase 1: Data Pipeline Updates**

* [ ] **Modify `load_graph_and_data**`:
* Insert code to fill `NaN`s in `detected_city`, `detected_country`, `detected_continent` with "Unknown".
* Create `_id` mappings (dictionaries) for these columns.
* Map the string columns to integer IDs in the DataFrame.


* [ ] **Modify `scale_features**`:
* Ensure the function dynamically locates the `target_col_idx` (index of 'cpc_week') instead of using a hardcoded integer, as column positions will shift between experiments.



#### **Phase 2: Configuration & Definitions**

* [ ] **Define Feature Groups**:
* Create lists: `CORE_FEAT`, `GEO_FEAT`, `DTW_FEAT`, `LAG_FEAT`, `REST_FEAT` (Semantic, Device, etc.).


* [ ] **Define Experiment Map**:
* Create the `EXPERIMENTS` dictionary mapping names (e.g., `"core_geo"`) to their respective list concatenations.


* [ ] **Define Models List**:
* Set `MODELS_TO_TEST = ["DCRNN", "GraphWaveNet", "GConvLSTM", "AGCRN"]`.



#### **Phase 3: The Experiment Loop**

* [ ] **Create Main Loop**:
* Iterate through `for exp_name, feature_list in EXPERIMENTS.items():`.


* [ ] **Dynamic Tensor Building**:
* Inside the loop, call `build_feature_tensor` and `scale_features` using *only* the current `feature_list`.


* [ ] **Dynamic Model Config**:
* Update `BASE_MODEL_CONFIGS[model]['params']['in_channels']` (or `input_size`) to match `len(feature_list)`.


* [ ] **Execution**:
* Create a specific output directory: `os.path.join(DRIVE_PATH, f"exp_{exp_name}")`.
* Run `train_one_model_optionA` with `epochs=50`.
* Save tensors to the specific experiment folder.



#### **Phase 4: Aggregation**

* [ ] **CSV Logging**:
* Ensure `export_results_to_csv` includes the `exp_name` in the `model_id` or a new column to differentiate runs in the final report.
### **Phase 1: Data Pipeline Upgrades**

**Goal:** Enable the processing of geographic covariates and ensure the scaler handles dynamic feature sets correctly.

* **Task 1.1: Update `load_graph_and_data` function**
* **Location:** "Data Utilities" cell.
* **Action:** Insert the geographic processing logic immediately after `df = pd.read_parquet(...)`.
* **Code Requirement:**
* Clean "None" strings and NaNs in `detected_city/country/continent`.
* Create sorted dictionaries for `city_to_id`, `country_to_id`, `continent_to_id`.
* Map string values to integer IDs in new columns: `detected_city_id`, `detected_country_id`, `detected_continent_id`.


* **Validation:** Ensure no `TypeError` occurs during sorting (handle mixed types).


* **Task 1.2: Verify `scale_features` flexibility**
* **Location:** "Data Utilities" cell.
* **Action:** Ensure `target_col_idx` is calculated dynamically inside the experiment loop (not hardcoded globally), as the index of `cpc_week` will shift depending on which features are active.



---

### **Phase 2: Configuration Restructuring**

**Goal:** Modularize the feature definitions so they can be mixed and matched programmatically.

* **Task 2.1: Split `FEATURE_COLS**`
* **Location:** "Configuration" cell.
* **Action:** Replace the single large `FEATURE_COLS` list with granular sub-lists:
* `CORE_FEAT` (impressions, cpc)
* `GEO_FEAT` (city_id, country_id, continent_id)
* `DTW_FEAT` (dtw_neighbor_1...20)
* `LAG_FEAT` (lags, rolling means)
* `REST_FEAT` (similarity, domain share, device, search type, semantic PCA)




* **Task 2.2: Define `EXPERIMENTS` Dictionary**
* **Location:** New cell before "Run Experiment" or within the main loop.
* **Action:** Create a map defining the ablation roadmap:
```python
EXPERIMENTS = {
    "core_only": CORE_FEAT,
    "core_geo": CORE_FEAT + GEO_FEAT,
    "core_dtw": CORE_FEAT + DTW_FEAT,
    "core_lags": CORE_FEAT + LAG_FEAT,
    "all_features": CORE_FEAT + GEO_FEAT + DTW_FEAT + LAG_FEAT + REST_FEAT
}

```




* **Task 2.3: Define Target Models**
* **Action:** Set a constant for the models to be tested to avoid running all of them.
* `TARGET_MODELS = ["DCRNN", "GraphWaveNet", "GConvLSTM", "AGCRN"]`



---

### **Phase 3: The Experiment Loop (Main Logic Rewrite)**

**Goal:** Automate the training, evaluation, and saving process for each feature set.

* **Task 3.1: Initialize the Main Loop**
* **Location:** "Run Experiment" cell.
* **Action:** Wrap the existing logic in a loop: `for exp_name, features in EXPERIMENTS.items():`
* **Sub-task:** Inside the loop, create a specific output directory:
`current_exp_dir = os.path.join(TENSOR_FOLDER, f"exp_{exp_name}")`


* **Task 3.2: Dynamic Data Reconstruction**
* **Action:** Inside the loop, update `FEATURE_COLS` to `features`.
* **Action:** Call `build_feature_tensor` and `scale_features` *inside the loop* to generate tensors (`X_train`, `X_test`, etc.) that strictly match the current feature set dimensions.


* **Task 3.3: Dynamic Model Configuration**
* **Action:** Before training, update `BASE_MODEL_CONFIGS` parameters for the current `num_features`.
* **Logic:**
```python
current_input_dim = len(features)
BASE_MODEL_CONFIGS['DCRNN']['params']['in_channels'] = current_input_dim
BASE_MODEL_CONFIGS['GraphWaveNet']['params']['input_size'] = current_input_dim
# ... repeat for GConvLSTM and AGCRN

```




* **Task 3.4: Execute Training & Saving**
* **Action:** Call `train_one_model_optionA` with `epochs=50`.
* **Action:** Call `evaluate_model_optionA`.
* **Action:** Call `save_model_tensors` pointing to `current_exp_dir`.



---

### **Phase 4: Output & Analysis**

**Goal:** Ensure results are stored in a way that enables easy comparison later.

* **Task 4.1: CSV Aggregation**
* **Action:** Ensure the `export_results_to_csv` function captures the `exp_name` (e.g., add a column for "experiment_id" or append it to the model name like `DCRNN_core_geo`).


* **Task 4.2: Verification**
* **Action:** Print a summary table at the end of each experiment loop showing the SMAPE for that specific feature set.