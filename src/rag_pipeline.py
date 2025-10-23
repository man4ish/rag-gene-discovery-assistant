"""
RAG Pipeline for Gene Discovery
-------------------------------
- Retrieves top PubMed abstracts using FAISS
- Generates answers using Ollama LLM (DeepSeek/LLaMA3)
- Supports optional structured extraction (drug-target-disease)
"""

import os
import json
import argparse
import numpy as np
import faiss
import ollama
from config import ABSTRACT_FOLDER, INDEX_FILE, ID_MAP_FILE, MODEL_NAME, TOP_K

# ------------------------------------------------------------
# RAG Assistant Class
# ------------------------------------------------------------
class RAGAssistant:
    def __init__(self, abstract_folder=ABSTRACT_FOLDER, model_name=MODEL_NAME):
        self.abstract_folder = abstract_folder
        self.model_name = model_name

    # ---------------------------
    # Abstract Utilities
    # ---------------------------
    def get_abstract_text(self, pmid: str) -> str:
        file_path = os.path.join(self.abstract_folder, f"{pmid}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("abstract", "")

    # ---------------------------
    # RAG Answer
    # ---------------------------
    def rag_answer(self, user_query, pmid_list):
        context = "\n\n".join([self.get_abstract_text(pmid) for pmid in pmid_list])
        prompt = (
            f"You are a biomedical research assistant. "
            f"Answer the following question using the context below.\n\n"
            f"Question: {user_query}\n\nContext:\n{context[:10000]}"
        )
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        return getattr(response, "message", str(response))

    # ---------------------------
    # Structured Extraction
    # ---------------------------
    def extract_structured_info(self, drug_name, abstract):
        prompt = f"""
        Extract structured drug-target-cancer information for '{drug_name}':
        {abstract}
        Format as JSON: [{{'target': str, 'cancer': str, 'mechanism': str}}]
        """
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        text = getattr(response, "message", str(response))
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return []

    # ---------------------------
    # FAISS Utilities
    # ---------------------------
    def load_faiss_index(self, index_file=INDEX_FILE, id_map_file=ID_MAP_FILE):
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"FAISS index not found: {index_file}")
        if not os.path.exists(id_map_file):
            raise FileNotFoundError(f"PMID map not found: {id_map_file}")

        index = faiss.read_index(index_file)
        with open(id_map_file, "r", encoding="utf-8") as f:
            pmid_map = json.load(f)
        return index, pmid_map

    def get_embedding(self, text: str):
        result = ollama.embeddings(model=self.model_name, prompt=text)
        return np.array(result["embedding"], dtype="float32")

    def retrieve_top_pmids(self, query: str, index, pmid_map, top_k: int = TOP_K):
        query_emb = self.get_embedding(query)
        faiss.normalize_L2(query_emb.reshape(1, -1))

        distances, indices = index.search(query_emb.reshape(1, -1), top_k)
        top_pmids = [pmid_map[i] for i in indices[0]]
        return top_pmids, distances[0]

    # ---------------------------
    # Full Pipeline
    # ---------------------------
    def run_pipeline(self, query: str, top_k: int = TOP_K, structured: bool = False):
        print("Loading FAISS index and PMID map...")
        index, pmid_map = self.load_faiss_index()

        print(f"Retrieving top {top_k} relevant abstracts...")
        top_pmids, scores = self.retrieve_top_pmids(query, index, pmid_map, top_k)

        print(f"Loaded {len(top_pmids)} abstracts.")
        summary = self.rag_answer(query, top_pmids)

        print("\n=== QUERY ===")
        print(query)
        print("\n=== SUMMARY ===")
        print(summary)
        print("\n=== PMIDs ===")
        print(", ".join(top_pmids))

        structured_results = []
        if structured:
            for pmid in top_pmids:
                abstract = self.get_abstract_text(pmid)
                structured_results.extend(self.extract_structured_info(query, abstract))
            print("\n=== STRUCTURED INFO ===")
            print(json.dumps(structured_results, indent=2))

        return summary, top_pmids, structured_results


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline for Gene Discovery")
    parser.add_argument("--query", type=str, required=True, help="Query to search PubMed abstracts")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Number of top abstracts to retrieve")
    parser.add_argument("--structured", action="store_true", help="Extract structured info from abstracts")
    args = parser.parse_args()

    assistant = RAGAssistant()
    assistant.run_pipeline(args.query, top_k=args.top_k, structured=args.structured)
