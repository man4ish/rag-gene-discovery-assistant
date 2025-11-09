"""
embedding_engine.py
-------------------
Generates embeddings for PubMed abstracts using Ollama (DeepSeek/LLaMA3).
Stores FAISS index and PMID mapping in external storage.

Enhancements:
- GPU-aware FAISS index creation.
- Resumable embedding (skip already processed PMIDs).
- Graceful handling of Ollama timeouts/errors.
- Progress save checkpoints every N embeddings.
"""

import os
import json
import time
import faiss
import numpy as np
import ollama
from tqdm import tqdm
import config


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
