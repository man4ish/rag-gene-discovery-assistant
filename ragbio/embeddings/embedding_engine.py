#!/usr/bin/pyhton3
"""
embedding_engine.py

Pipeline to generate vector embeddings from PubMed abstracts using Ollama,
and build a FAISS similarity index for retrieval in RAG workflows.

This module performs the following steps:
1. Load abstracts from JSON files (skips empty abstracts).
2. Generate embeddings for each abstract using the Ollama embedding model.
   - Supports checkpointing to resume progress.
   - Handles API failures gracefully by inserting zero vectors.
3. Build a FAISS index (inner product) on the embeddings.
   - Supports GPU acceleration if available and configured.
   - Normalizes embeddings to unit length for cosine similarity.
4. Save the FAISS index and corresponding PMID map for retrieval.

Configuration:
- Uses `ragbio.config` for paths, model parameters, and GPU usage:
    - ABSTRACT_FOLDER : Folder containing abstract JSON files
    - INDEX_FOLDER    : Folder to save FAISS index and mapping
    - INDEX_FILE      : File path to save FAISS index
    - ID_MAP_FILE     : JSON file mapping vector indices to PMIDs
    - MODEL_NAME      : Embedding model (Ollama)
    - USE_GPU         : Whether to use GPU for FAISS

Checkpointing:
- Optional JSON checkpoint allows resuming embedding generation.
- Checkpoints are saved every 500 abstracts to avoid API overload.

Dependencies:
- numpy, faiss
- ollama
- tqdm
- ragbio.config
- json, os, time

Usage:
    python build_faiss_index.py

Author:
    Manish Kumar
"""

import os
import json
import time
import faiss
import numpy as np
import ollama
from tqdm import tqdm
from ragbio import config


# ==========================================
# 1. Load abstracts
# ==========================================
def load_abstracts():
    """Load abstracts from JSON files, skipping empty ones."""
    abstracts, pmids = [], []

    for file in os.listdir(config.ABSTRACT_FOLDER):
        if file.endswith(".json"):
            with open(os.path.join(config.ABSTRACT_FOLDER, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                text = data.get("abstract", "").strip()
                if text:
                    abstracts.append(text)
                    pmids.append(data["pmid"])

    print(f"[INFO] Loaded {len(abstracts)} abstracts for embedding.")
    return abstracts, pmids


# ==========================================
# 2. Generate embeddings
# ==========================================
def generate_embeddings(texts, model_name, batch_size=8, checkpoint_path=None):
    """Generate embeddings using Ollama and save periodically."""
    embeddings = []
    start_idx = 0

    # Resume if checkpoint exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            saved = json.load(f)
        start_idx = len(saved.get("embeddings", []))
        embeddings = saved["embeddings"]
        print(f"[RESUME] Resuming from index {start_idx}")

    for i in tqdm(range(start_idx, len(texts)), desc="Generating embeddings"):
        text = texts[i]
        try:
            result = ollama.embeddings(model=model_name, prompt=text)
            emb = result["embedding"]
        except Exception as e:
            print(f"[WARN] Embedding failed for index {i}: {e}")
            emb = [0] * 768

        embeddings.append(emb)

        # Save checkpoint every 500 embeddings
        if checkpoint_path and (i + 1) % 500 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({"embeddings": embeddings}, f)
            print(f"[CHECKPOINT] Saved progress at {i + 1} embeddings")

        # Small delay to avoid API overload
        time.sleep(0.1)

    # Final save
    if checkpoint_path:
        with open(checkpoint_path, "w") as f:
            json.dump({"embeddings": embeddings}, f)
    return np.array(embeddings, dtype="float32")


# ==========================================
# 3. Build FAISS index
# ==========================================
def build_faiss_index(embeddings):
    """Build FAISS index (GPU if available)."""
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    if config.USE_GPU:
        try:
            print("[INFO] Using GPU FAISS index")
            res = faiss.StandardGpuResources()
            index_flat = faiss.IndexFlatIP(dim)
            index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        except Exception as e:
            print(f"[WARN] GPU FAISS failed, using CPU. Reason: {e}")
            index = faiss.IndexFlatIP(dim)
    else:
        print("[INFO] Using CPU FAISS index")
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings)
    print(f"[INFO] FAISS index built with {index.ntotal} vectors.")
    return index


# ==========================================
# 4. Save index and mapping
# ==========================================
def save_index(index, pmids):
    """Save FAISS index and PMID map."""
    os.makedirs(config.INDEX_FOLDER, exist_ok=True)
    faiss.write_index(index, config.INDEX_FILE)
    with open(config.ID_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(pmids, f, indent=2)

    print(f"[OK] Index saved → {config.INDEX_FILE}")
    print(f"[OK] PMID map saved → {config.ID_MAP_FILE}")


# ==========================================
# 5. Main
# ==========================================
def main():
    abstracts, pmids = load_abstracts()
    if not abstracts:
        print("[ERROR] No abstracts found. Run data_loader.py first.")
        return

    checkpoint = os.path.join(config.INDEX_FOLDER, "embedding_checkpoint.json")
    embeddings = generate_embeddings(abstracts, config.MODEL_NAME, checkpoint_path=checkpoint)
    index = build_faiss_index(embeddings)
    save_index(index, pmids)
    print("[DONE] Embedding pipeline complete.")


if __name__ == "__main__":
    main()
