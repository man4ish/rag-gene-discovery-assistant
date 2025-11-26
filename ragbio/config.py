#!/usr/bin/pyhton3

"""
config.py

Configuration module for the RAG (Retrieval-Augmented Generation) gene discovery pipeline.

This file defines all key paths, folders, index files, model parameters, and flags 
used across the pipeline, including:

1. Workspace and Base Paths:
   - WORKSPACE_DIR : Main workspace directory (adjusts automatically for Docker or local paths)
   - BASE_PATH     : Base data folder containing abstracts, metadata, PDFs, and index

2. Derived Folders:
   - ABSTRACT_FOLDER : Stores downloaded PubMed abstracts in JSON format
   - METADATA_FOLDER : Stores metadata JSON files for each PubMed article
   - PDF_FOLDER      : Stores PDFs of open-access articles
   - INDEX_FOLDER    : Stores FAISS index and PMID mapping files

3. Index Files:
   - INDEX_FILE : FAISS vector index of PubMed abstracts
   - ID_MAP_FILE: JSON file mapping FAISS index positions to PubMed IDs

4. Model and Retrieval Parameters:
   - MODEL_NAME : Embedding model used for generating vector representations (from Ollama)
   - TOP_K      : Default number of top abstracts to retrieve

5. GPU Usage:
   - USE_GPU : Boolean flag indicating whether to use GPU for embeddings

6. Directory Setup:
   - Ensures all key folders exist at startup

Environment Variables:
- OLLAMA_MODEL : Optional override for MODEL_NAME

Author:
    Manish Kumar
"""


import os

# Use /workspace only if it exists (Docker)
if os.path.exists("/workspace"):
    WORKSPACE_DIR = "/workspace"
else:
    WORKSPACE_DIR = "/home/manish/Desktop/machine/rag-gene-discovery-assistant"

os.makedirs(WORKSPACE_DIR, exist_ok=True)

# Base path – can also use environment variable
BASE_PATH = r"/home/manish/Desktop/machine/rag-gene-discovery-assistant/data/PubMed"

# Derived folders
ABSTRACT_FOLDER = os.path.join(BASE_PATH, "Abstracts")
METADATA_FOLDER = os.path.join(BASE_PATH, "Metadata")
PDF_FOLDER = os.path.join(BASE_PATH, "PDFs")
INDEX_FOLDER = os.path.join(BASE_PATH, "Index")

# Index files
INDEX_FILE = os.path.join(INDEX_FOLDER, "pubmed_index.faiss")
ID_MAP_FILE = os.path.join(INDEX_FOLDER, "pmid_map.json")

# Model and search parameters
MODEL_NAME = os.getenv("OLLAMA_MODEL", "mxbai-embed-large")
TOP_K = 15

# GPU usage flag
USE_GPU = True

# Ensure all directories exist
for folder in [ABSTRACT_FOLDER, METADATA_FOLDER, PDF_FOLDER, INDEX_FOLDER]:
    os.makedirs(folder, exist_ok=True)

print("Workspace directory:", WORKSPACE_DIR)
print("Base path:", BASE_PATH)
