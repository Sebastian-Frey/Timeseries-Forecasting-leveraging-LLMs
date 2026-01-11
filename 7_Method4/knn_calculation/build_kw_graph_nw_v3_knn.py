import pandas as pd
import numpy as np
import json
import os
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

# --- Input Source ---
# *** IMPORTANT: Update this path to your new unique keyword list ***
PRESELECTED_KEYWORDS_CSV_PATH = 'data/sebs_unique_keywords_ddf.csv' 
KEYWORDS_CSV_COLUMN_NAME = "keyword"

# --- Graph & Output Config ---
OUTPUT_BASE_DIR = "data/sebs_keyword_graph_knn_multiple"
NLP_MODEL_NAME = 'all-MiniLM-L6-v2'
K_VALUES = [3, 5, 7, 10, 20]  # Generate graphs for multiple K values

# --------------------------------------------------------------------------
# Helper Function
# --------------------------------------------------------------------------
def generate_graph_for_k(k, sim_matrix, num_nodes, idx_to_keyword, output_dir):
    """
    Generate a KNN graph for a specific K value using pre-computed similarity matrix.

    Args:
        k: Number of neighbors to connect for each node
        sim_matrix: Pre-computed cosine similarity matrix (num_nodes x num_nodes)
        num_nodes: Total number of nodes in the graph
        idx_to_keyword: Dictionary mapping node indices to keyword strings
        output_dir: Directory to save graph files

    Returns:
        num_edges: Total number of edges created
    """
    # Find top K neighbors for each node
    top_k_indices = np.argsort(sim_matrix, axis=1)[:, -k:]

    # Build edge lists
    source_nodes = []
    target_nodes = []
    edge_weights = []

    for i in tqdm(range(num_nodes), desc=f"Building edge list for K={k}"):
        for j in top_k_indices[i]:
            source_nodes.append(i)
            target_nodes.append(j)
            edge_weights.append(sim_matrix[i, j])

    edge_index = np.array([source_nodes, target_nodes], dtype=np.int64)
    edge_weight = np.array(edge_weights, dtype=np.float32)
    num_edges = edge_index.shape[1]

    # Build NetworkX graph
    G = nx.Graph()
    for i, keyword in idx_to_keyword.items():
        G.add_node(i, label=keyword)

    for i in range(num_edges):
        source, target = edge_index[0, i], edge_index[1, i]
        G.add_edge(source, target, weight=float(edge_weight[i]))

    # Save all outputs
    os.makedirs(output_dir, exist_ok=True)

    # Save NetworkX graph
    graph_path = os.path.join(output_dir, "keyword_graph_knn.gexf")
    nx.write_gexf(G, graph_path)

    # Save numpy arrays
    np.save(os.path.join(output_dir, "edge_index.npy"), edge_index)
    np.save(os.path.join(output_dir, "edge_weight.npy"), edge_weight)

    # Save keyword map
    keyword_map = {keyword: i for i, keyword in idx_to_keyword.items()}
    with open(os.path.join(output_dir, "keyword_map.json"), 'w') as f:
        json.dump(keyword_map, f)

    return num_edges

# --------------------------------------------------------------------------
# Main Script
# --------------------------------------------------------------------------
def main():
    print("Starting keyword graph construction (KNN Method)...")
    
    # --- Step 1: Get the list of unique keywords ---
    print(f"\n[Step 1] Loading pre-selected keywords from: {PRESELECTED_KEYWORDS_CSV_PATH}")
    try:
        df_keywords = pd.read_csv(PRESELECTED_KEYWORDS_CSV_PATH)
        if KEYWORDS_CSV_COLUMN_NAME not in df_keywords.columns:
            raise KeyError(f"Column '{KEYWORDS_CSV_COLUMN_NAME}' not found in the CSV file.")
        
        unique_keywords = sorted(df_keywords[KEYWORDS_CSV_COLUMN_NAME].dropna().unique().tolist())
        print(f"   - Loaded {len(unique_keywords)} unique keywords from the file.")
        
    except FileNotFoundError:
        print(f"ERROR: The file specified was not found at '{PRESELECTED_KEYWORDS_CSV_PATH}'")
        return
    except KeyError as e:
        print(f"ERROR: {e}")
        return

    keyword_map = {keyword: i for i, keyword in enumerate(unique_keywords)}
    idx_to_keyword = {i: keyword for keyword, i in keyword_map.items()}
    num_nodes = len(unique_keywords)

    # Validate K_VALUES configuration
    if not K_VALUES or not isinstance(K_VALUES, list):
        print("ERROR: K_VALUES must be a non-empty list")
        return

    if not all(isinstance(k, int) and k > 0 for k in K_VALUES):
        print("ERROR: All K values must be positive integers")
        return

    max_k = max(K_VALUES)
    if max_k >= num_nodes:
        print(f"WARNING: Maximum K value ({max_k}) is >= number of nodes ({num_nodes})")
        print(f"         This will result in near-fully-connected graphs")

    # --- Step 2: Encode keywords and compute similarity matrix ---
    print(f"\n[Step 2] Encoding keywords and computing similarity matrix...")
    print(f"   - Model: {NLP_MODEL_NAME}")
    model = SentenceTransformer(NLP_MODEL_NAME)
    
    print("   - Encoding keywords into vector embeddings...")
    embeddings = model.encode(unique_keywords, show_progress_bar=True)
    
    print("   - Calculating the full cosine similarity matrix...")
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -1)  # Prevent self-connections

    # --- Step 3: Generate graphs for each K value ---
    print(f"\n[Step 3] Generating graphs for K values: {K_VALUES}...")

    for k in K_VALUES:
        print(f"   - Processing K={k}...")
        subdir = os.path.join(OUTPUT_BASE_DIR, f"k{k}")
        num_edges = generate_graph_for_k(k, sim_matrix, num_nodes, idx_to_keyword, subdir)
        print(f"     Created graph with {num_edges} edges (N * K = {num_nodes * k})")
        print(f"     Saved to: {subdir}")

    print(f"\n[Step 4] Graph generation complete!")
    print(f"Generated {len(K_VALUES)} graphs in: {OUTPUT_BASE_DIR}")
    print("-" * 60)

if __name__ == "__main__":
    main()