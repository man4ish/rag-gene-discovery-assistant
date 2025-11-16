import os

# Base path – use environment variable if available (for Docker mounting)
BASE_PATH = os.getenv("PUBMED_BASE_PATH", "/workspace/data/PubMed")

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
TOP_K = int(os.getenv("TOP_K", 10))

# GPU usage flag for FAISS or embeddings
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"

# Ensure all directories exist
for folder in [ABSTRACT_FOLDER, METADATA_FOLDER, PDF_FOLDER, INDEX_FOLDER]:
    os.makedirs(folder, exist_ok=True)
