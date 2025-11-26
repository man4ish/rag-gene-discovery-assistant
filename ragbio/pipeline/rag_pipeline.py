#!/usr/bin/pyhton3

"""
rag_pipeline.py

RAG (Retrieval-Augmented Generation) Pipeline for Biomedical Literature:
- Retrieves relevant PubMed abstracts using a FAISS vector index.
- Generates natural language summaries using LangChain LLMs (Ollama).
- Optionally extracts structured drug-target-disease information.
- Integrates structured results into a Neo4j knowledge graph.

Features:
1. FAISSPMIDRetriever: Custom LangChain retriever to load FAISS index and return top-K documents.
2. RAGAssistant: High-level class to run the RAG pipeline:
   - Load FAISS index and PMID map
   - Retrieve top-K relevant abstracts
   - Generate summary via LangChain RAG chain
   - Optionally perform structured extraction and update KG
3. Robust JSON handling for abstracts and metadata.
4. CLI support for ad-hoc queries with top-K and structured extraction options.

Configuration:
- Uses environment and config settings from `ragbio.config`:
    - ABSTRACT_FOLDER: Folder storing abstract JSONs
    - INDEX_FILE: FAISS index file path
    - ID_MAP_FILE: Mapping of FAISS indices to PMIDs
    - MODEL_NAME: Embedding model for Ollama embeddings
    - TOP_K: Default number of abstracts to retrieve

Dependencies:
- numpy, faiss
- LangChain (llms, embeddings, documents, runnables, prompts, retrievers)
- pydantic
- ragbio.config
- ragbio.knowledge_graph.structured_drug_kg
- json, os, argparse, re, operator

Usage:
    python rag_pipeline.py --query "BRCA1 drug interactions" --top_k 10 --structured

Author:
    Manish Kumar
"""


import os
import re
import json
import numpy as np
import argparse
import faiss
from typing import List
from operator import itemgetter

# LangChain Imports
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
# New, correct path for BaseRetriever
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

# Local Imports
from ragbio.config import (
    ABSTRACT_FOLDER,
    INDEX_FILE,
    ID_MAP_FILE,
    MODEL_NAME,   # Embedding model from config
    TOP_K,
)
from ragbio.knowledge_graph.structured_drug_kg import add_structured_data_to_kg


# ------------------------------------------------------------
# Custom LangChain Retriever
# ------------------------------------------------------------
class FAISSPMIDRetriever(BaseRetriever):
    """A custom LangChain Retriever to load FAISS index and return Documents."""
    
    # Define an embeddings client and the search index/map as fields
    embeddings: OllamaEmbeddings = Field(...)
    index: faiss.Index = Field(...)
    pmid_map: List[str] = Field(...)
    abstract_folder: str = Field(...)
    k: int = Field(default=TOP_K)
    
    class Config:
        arbitrary_types_allowed = True
        
    def _get_abstract_text(self, pmid: str) -> str:
        """Utility to get abstract text from a JSON file."""
        file_path = os.path.join(self.abstract_folder, f"{pmid}.json")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("abstract", "")
        except FileNotFoundError:
            return ""

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """LangChain's required retrieval method."""
        
        # 1. Get Query Embedding (Returns a Python list: List[float])
        query_list = self.embeddings.embed_query(query)

        # 2. CONVERT TO NUMPY ARRAY and RESHAPE to (1, D)
        # The key fix: Convert to float32 AND reshape to a 2D array (1, D)
        query_emb = np.array(query_list, dtype=np.float32).reshape(1, -1) 

        # 3. Normalize the 2D array (FAISS now correctly accesses shape[0] and shape[1])
        faiss.normalize_L2(query_emb)

        # 4. Search Index
        # The array is already shaped (1, D), so you can pass it directly
        distances, indices = self.index.search(query_emb, self.k)
        
        # 5. Map to PMIDs and Create Documents
        documents = []
        for i, score in zip(indices[0], distances[0]):
            pmid = self.pmid_map[i]
            abstract_text = self._get_abstract_text(pmid)
            
            # Create a LangChain Document object
            doc = Document(
                page_content=abstract_text,
                metadata={"source": pmid, "score": float(score)}
            )
            documents.append(doc)
            
        return documents


# ------------------------------------------------------------
# RAG Assistant Class (Adapted for LangChain)
# ------------------------------------------------------------
class RAGAssistant:
    def __init__(self, abstract_folder=ABSTRACT_FOLDER, embed_model=MODEL_NAME, chat_model="deepseek-r1:latest", output_dir="output"):
        self.abstract_folder = abstract_folder
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # LangChain LLM and Embeddings setup
        self.embed_model = OllamaEmbeddings(model=embed_model)
        self.chat_model = Ollama(model=chat_model, temperature=0) # Use LangChain's Ollama LLM
        
        # Load FAISS components
        self.index, self.pmid_map = self.load_faiss_index()
        
        # Initialize the custom LangChain Retriever
        self.retriever = FAISSPMIDRetriever(
            embeddings=self.embed_model,
            index=self.index,
            pmid_map=self.pmid_map,
            abstract_folder=self.abstract_folder
        )

        # Build the RAG Chain using LCEL
        self.rag_chain = self._build_rag_chain()

    # ---------------------------
    # LangChain RAG Chain
    # ---------------------------
    def _build_rag_chain(self):
        """Builds the main RAG chain using LCEL."""
        
        # Prompt Template
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a biomedical research assistant. Answer the question only using the context provided below. If the context does not contain the answer, state that you cannot answer based on the provided documents."),
            ("user", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        
        # Document formatter function for context stuffing
        def format_docs(docs: List[Document]):
            # Join the page_content of all retrieved documents
            return "\n\n".join([doc.page_content for doc in docs])

        # LCEL Chain:
        # 1. RunnablePassthrough: Pass the 'question' to the next step
        # 2. RunnablePassthrough + RunnableLambda: Retrieve docs and format them as 'context'
        # 3. Chain the formatted context and the original question into the prompt
        # 4. Pass the prompt to the LLM
        chain = (
            {"context": itemgetter("question") | self.retriever | format_docs, 
             "question": itemgetter("question")}
            | rag_prompt
            | self.chat_model
        )
        return chain

    # ---------------------------
    # Abstract/Index Utilities (Kept for compatibility)
    # ---------------------------
    def get_abstract_text(self, pmid: str) -> str:
        # Now uses the Retriever's utility method
        return self.retriever._get_abstract_text(pmid) 

    def load_faiss_index(self, index_file=INDEX_FILE, id_map_file=ID_MAP_FILE):
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"FAISS index not found: {index_file}")
        if not os.path.exists(id_map_file):
            raise FileNotFoundError(f"PMID map not found: {id_map_file}")

        # Use faiss.read_index directly (LangChain FAISS wrapper is for building, not loading raw faiss index)
        index = faiss.read_index(index_file)
        with open(id_map_file, "r", encoding="utf-8") as f:
            pmid_map = json.load(f)
        return index, pmid_map

    # ---------------------------
    # Structured Extraction (Adapted)
    # ---------------------------
    # In RAGAssistant class

    def extract_structured_info(self, drug_name, abstract):
        """Replaced custom ollama.chat with LangChain LLM call."""
        prompt = f"""
        Extract structured drug-target-cancer information for '{drug_name}':
        {abstract}
        
        Format your *entire* response as ONLY a JSON list (e.g., [{{...}}]) 
        and include absolutely NO additional text, commentary, or markdown formatting 
        outside of the JSON structure itself.
        """
        # Call the LangChain LLM object
        text = self.chat_model.invoke(prompt)
        
        try:
            # 1. Strip Markdown fences (in case the LLM still uses them)
            if text.startswith("```json"):
                text = text.strip().replace("```json\n", "").replace("\n```", "")
            elif text.startswith("```"):
                text = text.strip().replace("```\n", "").replace("\n```", "")
            
            # 2. Aggressive Cleanup (The most important step for your error):
            # Find the first '[' and the last ']' to isolate the JSON list.
            # This handles pre-text, post-text, and commentary within the string.
            start_index = text.find('[')
            end_index = text.rfind(']')
            
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_text = text[start_index : end_index + 1]
                parsed = json.loads(json_text)
                return parsed if isinstance(parsed, list) else [parsed]
            else:
                # If we can't find a clear JSON structure, assume failure
                raise ValueError("No valid JSON list structure found.")

        except Exception as e:
            print(f"Warning: Failed to parse structured JSON: {e}")
            print(f"Raw text: {text}")
            return []

    # ---------------------------
    # Output Saving (Unchanged)
    # ---------------------------
    def save_output(self, query, summary, top_pmids, structured_results=None):
        # Save summary and PMIDs (optional, same as before)
        summary_file = os.path.join(self.output_dir, "rag_demo_results.txt")
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(f"Query: {query}\n")
            f.write(f"PMIDs: {', '.join(top_pmids)}\n")
            f.write(f"Summary: {summary}\n\n")

        # ---------------------------
        # New: Save JSON per query
        # ---------------------------
        if structured_results:
            # Derive safe filename from query
            query_var = query.lower()
            query_var = re.sub(r'\W+', '_', query_var)  # Replace non-alphanumeric with underscore
            json_file = os.path.join(self.output_dir, f"{query_var}_output.json")

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(structured_results, f, indent=2)
            print(f"Structured results saved to: {json_file}")

    # ---------------------------
    # Full Pipeline (Adapted)
    # ---------------------------
    def run_pipeline(self, query: str, top_k: int = TOP_K, structured: bool = False):
        print("Loading FAISS index and PMID map...")
        # Note: The index is loaded in __init__ now.
        self.retriever.k = top_k # Update the retriever's k value
        
        # 1. LangChain RAG Call
        print(f"Running RAG chain with top {top_k} abstracts...")
        chain_result = self.rag_chain.invoke({"question": query})
        
        # 2. Get Retrieved Documents (Need to re-run retrieval to get PMIDs)
        retrieved_docs = self.retriever._get_relevant_documents(query, k=top_k)
        top_pmids = [doc.metadata.get("source") for doc in retrieved_docs]
        summary = chain_result
        
        structured_results = []
        if structured:
            print("\n--- Starting Structured Extraction ---")
            for pmid in top_pmids:
                abstract = self.get_abstract_text(pmid)
                # Pass the main query as the drug_name hint
                structured_entries = self.extract_structured_info(query, abstract)
                for entry in structured_entries:
                    entry["pmid"] = pmid
                structured_results.extend(structured_entries)

            if structured_results:
                add_structured_data_to_kg(structured_results) # Your unchanged KG function
                print("Structured data added to Neo4j KG.")

        # Print results
        print("\n=== QUERY ===")
        print(query)
        print("\n=== SUMMARY ===")
        print(summary)
        print("\n=== PMIDs ===")
        print(", ".join(top_pmids))

        if structured_results:
            print("\n=== STRUCTURED INFO ===")
            print(json.dumps(structured_results, indent=2))

        self.save_output(query, summary, top_pmids, structured_results)
        return summary, top_pmids, structured_results


# ------------------------------------------------------------
# CLI (Unchanged)
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline for Gene Discovery + Structured KG")
    parser.add_argument("--query", type=str, required=True, help="Query to search PubMed abstracts")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Number of top abstracts to retrieve")
    parser.add_argument("--structured", action="store_true", help="Extract structured info and populate Neo4j KG")
    args = parser.parse_args()

    assistant = RAGAssistant()
    assistant.run_pipeline(args.query, top_k=args.top_k, structured=args.structured)