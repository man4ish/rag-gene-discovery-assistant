"""
embedding_engine.py
-------------------
Generates embeddings for PubMed abstracts using Ollama (DeepSeek/LLaMA3).
Stores FAISS index and PMID mapping in external storage.
"""

import os
import json
import faiss
import numpy as np
import ollama
from tqdm import tqdm
import config


def load_abstracts():
    """Load all abstracts into memory."""
    abstracts, pmids = [], []

    for file in os.listdir(config.ABSTRACT_FOLDER):
        if file.endswith(".json"):
            with open(os.path.join(config.ABSTRACT_FOLDER, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                text = data.get("abstract", "").strip()
                if text:
                    abstracts.append(text)
                    pmids.append(data["pmid"])

    print(f"Loaded {len(abstracts)} abstracts for embedding.")
    return abstracts, pmids


def generate_embeddings(texts, model_name, batch_size=10):
    """Generate embeddings using Ollama models."""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i + batch_size]
        for text in batch:
            try:
                result = ollama.embeddings(model=model_name, prompt=text)
                embeddings.append(result["embedding"])
            except Exception as e:
                print(f"Embedding error: {e}")
                embeddings.append([0] * 768)  # placeholder if failure
    return np.array(embeddings, dtype="float32")


def build_faiss_index(embeddings):
    """Build and normalize FAISS index."""
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index


def save_index(index, pmids):
    """Save FAISS index and PMID map."""
    os.makedirs(config.INDEX_FOLDER, exist_ok=True)
    faiss.write_index(index, config.INDEX_FILE)
    with open(config.ID_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(pmids, f, indent=2)
    print(f"Index saved to {config.INDEX_FILE}")
    print(f"PMID map saved to {config.ID_MAP_FILE}")


def main():
    abstracts, pmids = load_abstracts()
    if not abstracts:
        print("No abstracts found. Please run data_loader.py first.")
        return

    embeddings = generate_embeddings(abstracts, config.MODEL_NAME)
    index = build_faiss_index(embeddings)
    save_index(index, pmids)
    print("Embedding pipeline complete.")


if __name__ == "__main__":
    main()
