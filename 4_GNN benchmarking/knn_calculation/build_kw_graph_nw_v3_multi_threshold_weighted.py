import dask.dataframe as dd
import pandas as pd
import numpy as np
import json
import os
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dask.diagnostics import ProgressBar

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

# --- Input Source ---
# OPTION 1: Provide a path to a CSV with pre-selected keywords.
#           The script will ONLY use these keywords. Set to None to disable.
PRESELECTED_KEYWORDS_CSV_PATH = 'data/sebs_unique_keywords_ddf.csv' # Example: "data/my_keywords.csv"

# If using a CSV, specify the column name that contains the keywords.
KEYWORDS_CSV_COLUMN_NAME = "keyword"

# OPTION 2: If the CSV path is None, the script will discover keywords here.
INPUT_PATH = "data/final_cleaned_dataset"

# --- Keyword Discovery Config (Only used if PRESELECTED_KEYWORDS_CSV_PATH is None) ---
TRUNCATE_KEYWORDS_TO_WORDS = 3
MIN_KEYWORD_FREQUENCY = 100

# --- Graph & Output Config ---
BASE_OUTPUT_DIR = "data/sebs_keyword_graph_threshold_weighted"
NLP_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Multiple Thresholds ---
SIMILARITY_THRESHOLDS = [0.6, 0.7, 0.8]

# --- Advanced: Batch Processing Config ---
USE_BATCHING = False
BATCH_SIZE = 2048

# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------
def truncate_keyword_string(s: str, num_words: int) -> str:
    if not isinstance(s, str): return ""
    return " ".join(s.split()[:num_words])

def calculate_edges_in_batches(embeddings, threshold, batch_size):
    from tqdm import tqdm
    num_nodes = embeddings.shape[0]
    source_nodes, target_nodes, weights = [], [], []
    for i in tqdm(range(0, num_nodes, batch_size), desc="Calculating edges in batches"):
        end = i + batch_size
        sim_chunk = cosine_similarity(embeddings[i:end], embeddings)
        adj_chunk = sim_chunk > threshold
        if end > i:
            adj_chunk[:, i:end] = np.where(np.eye(end-i, M=num_nodes, k=i, dtype=bool), False, adj_chunk[:, i:end])
        batch_sources, batch_targets = np.where(adj_chunk)
        source_nodes.extend(batch_sources + i)
        target_nodes.extend(batch_targets)
        # Extract weights for the edges
        for src, tgt in zip(batch_sources, batch_targets):
            weights.append(sim_chunk[src, tgt])
    return np.array([source_nodes, target_nodes], dtype=np.int64), np.array(weights, dtype=np.float32)

def build_and_save_graph(edge_index, edge_weights, idx_to_keyword, output_dir, threshold):
    """Build and save graph for a specific threshold with weights."""
    num_edges = edge_index.shape[1]

    print(f"   - Threshold {threshold}: Created graph with {num_edges} edges.")

    if num_edges == 0:
        print(f"   - WARNING: No edges for threshold {threshold}. Skipping...")
        return

    # Build NetworkX graph
    G = nx.Graph()
    for i, keyword in idx_to_keyword.items():
        G.add_node(i, label=keyword)

    # Add edges with weights
    for i in range(num_edges):
        source, target = edge_index[0, i], edge_index[1, i]
        weight = float(edge_weights[i])
        G.add_edge(source, target, weight=weight)

    # Save graph
    os.makedirs(output_dir, exist_ok=True)
    graph_output_path = os.path.join(output_dir, "keyword_graph.gexf")
    nx.write_gexf(G, graph_output_path)

    # Save machine-readable artifacts (including weights)
    np.save(os.path.join(output_dir, "edge_index.npy"), edge_index)
    np.save(os.path.join(output_dir, "edge_weights.npy"), edge_weights)

    print(f"   - Saved graph with weights to: {output_dir}")

# --------------------------------------------------------------------------
# Main Script
# --------------------------------------------------------------------------
def main():
    print("Starting keyword graph construction with multiple thresholds (with weights)...")
    print(f"Thresholds to process: {SIMILARITY_THRESHOLDS}")

    # --- Step 1: Get the list of unique keywords ---
    if PRESELECTED_KEYWORDS_CSV_PATH:
        print(f"\n[Step 1] Loading pre-selected keywords from: {PRESELECTED_KEYWORDS_CSV_PATH}")
        try:
            df_keywords = pd.read_csv(PRESELECTED_KEYWORDS_CSV_PATH)
            if KEYWORDS_CSV_COLUMN_NAME not in df_keywords.columns:
                raise KeyError(f"Column '{KEYWORDS_CSV_COLUMN_NAME}' not found in the CSV file.")

            # Use keywords directly from the CSV
            unique_keywords = sorted(df_keywords[KEYWORDS_CSV_COLUMN_NAME].dropna().unique().tolist())
            print(f"   - Loaded {len(unique_keywords)} unique keywords from the file.")

        except FileNotFoundError:
            print(f"ERROR: The file specified was not found at '{PRESELECTED_KEYWORDS_CSV_PATH}'")
            return
        except KeyError as e:
            print(f"ERROR: {e}")
            return
    else:
        print(f"\n[Step 1] Discovering keywords from the main dataset at '{INPUT_PATH}'...")
        ddf = dd.read_parquet(INPUT_PATH)

        print(f"   - Truncating keywords to their first {TRUNCATE_KEYWORDS_TO_WORDS} words...")
        ddf['processed_keyword'] = ddf['keyword'].apply(
            truncate_keyword_string,
            meta=('processed_keyword', 'object'),
            num_words=TRUNCATE_KEYWORDS_TO_WORDS
        )

        print(f"   - Filtering keywords with frequency < {MIN_KEYWORD_FREQUENCY}...")
        with ProgressBar():
            keyword_counts = ddf['processed_keyword'].value_counts().compute()

        frequent_keywords = keyword_counts[keyword_counts >= MIN_KEYWORD_FREQUENCY].index.tolist()

        if not frequent_keywords:
            raise ValueError(f"No keywords met the frequency criteria. Try lowering MIN_KEYWORD_FREQUENCY.")

        unique_keywords = sorted(frequent_keywords)
        print(f"   - Found {len(unique_keywords)} unique keywords to use as nodes.")

    # --- Create keyword mappings (same for all thresholds) ---
    keyword_map = {keyword: i for i, keyword in enumerate(unique_keywords)}
    idx_to_keyword = {i: keyword for keyword, i in keyword_map.items()}

    # --- Step 2: Compute embeddings (ONCE for all thresholds) ---
    print(f"\n[Step 2] Computing embeddings (Model: {NLP_MODEL_NAME})...")
    model = SentenceTransformer(NLP_MODEL_NAME)

    print("   - Encoding keywords into vector embeddings...")
    embeddings = model.encode(unique_keywords, show_progress_bar=True)

    # Compute full similarity matrix (needed for weights)
    print("   - Calculating the full cosine similarity matrix...")
    sim_matrix = cosine_similarity(embeddings)

    # --- Step 3: Process each threshold ---
    print(f"\n[Step 3] Building graphs for each threshold...")

    # Save keyword_map once in the base directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(BASE_OUTPUT_DIR, "keyword_map.json"), 'w') as f:
        json.dump(keyword_map, f)
    print(f"   - Saved shared 'keyword_map.json' to '{BASE_OUTPUT_DIR}'")

    for threshold in SIMILARITY_THRESHOLDS:
        print(f"\n   Processing threshold: {threshold}")

        # Create threshold-specific output directory
        threshold_dir = os.path.join(BASE_OUTPUT_DIR, f"t{threshold}")

        # Calculate edges for this threshold
        if USE_BATCHING:
            edge_index, edge_weights = calculate_edges_in_batches(embeddings, threshold, BATCH_SIZE)
        else:
            adj_matrix = sim_matrix > threshold
            np.fill_diagonal(adj_matrix, False)
            edge_index = np.array(np.where(adj_matrix), dtype=np.int64)

            # Extract weights for the edges
            edge_weights = sim_matrix[edge_index[0], edge_index[1]].astype(np.float32)

        # Build and save graph
        build_and_save_graph(edge_index, edge_weights, idx_to_keyword, threshold_dir, threshold)

    print("\n-----------------------------------------")
    print("Graph construction complete!")
    print(f"Output directory: {BASE_OUTPUT_DIR}")
    print(f"Processed {len(SIMILARITY_THRESHOLDS)} thresholds: {SIMILARITY_THRESHOLDS}")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()
