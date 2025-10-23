import os

# Base configuration
BASE_PATH = "/Volumes/Seagate2TB/PubMed_v2"

# Derived folders
ABSTRACT_FOLDER = os.path.join(BASE_PATH, "Abstracts")
METADATA_FOLDER = os.path.join(BASE_PATH, "Metadata")
PDF_FOLDER = os.path.join(BASE_PATH, "PDFs")
INDEX_FOLDER = os.path.join(BASE_PATH, "Index")

# Index files
INDEX_FILE = os.path.join(INDEX_FOLDER, "pubmed_index.faiss")
ID_MAP_FILE = os.path.join(INDEX_FOLDER, "pmid_map.json")

# Model and search parameters
MODEL_NAME = "deepseek-r1:latest"
TOP_K = 10

# Ensure all directories exist
for folder in [ABSTRACT_FOLDER, METADATA_FOLDER, PDF_FOLDER, INDEX_FOLDER]:
    os.makedirs(folder, exist_ok=True)
