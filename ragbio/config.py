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
