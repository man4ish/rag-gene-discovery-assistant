# RAG-Powered Gene Discovery Assistant

A generative AI tool for biomedical knowledge discovery using **Hugging Face embeddings** and **Ollama LLMs (DeepSeek / LLaMA3)**.  
This project integrates **retrieval-augmented generation (RAG)** with PubMed literature and gene annotation data to summarize gene–disease relationships.

---

## Overview
The RAG-powered assistant enables:

- Semantic search over PubMed abstracts and gene annotations.
- Summarization of complex biomedical information.
- Citation tracking with PubMed IDs.
- Modular and reusable pipeline for gene-disease exploration.

**Example queries:**
- "Which genes are linked to oxidative stress in Alzheimer’s disease?"
- "Summarize recent findings about TP53 variants in cancer."

---

## Architecture

```

User Query
│
▼
Vector Retrieval (BioBERT / BioSentVec Embeddings)
│
▼
Top Abstracts + Gene Annotations
│
▼
Ollama LLM (DeepSeek / LLaMA3)
│
▼
Summarized Biomedical Answer + Citations

````

---

## Installation

```bash
git clone https://github.com/<your-username>/rag-gene-discovery-assistant.git
cd rag-gene-discovery-assistant
pip install -r requirements.txt
````

**Example `requirements.txt`**

```
langchain
faiss-cpu
sentence-transformers
biopython
pymed
requests
sqlite-utils
ollama
pandas
```

---

## Usage

#### 1. Load Data and Build Embeddings

```bash
python src/data_loader.py
python src/embedding_engine.py
```

#### 2. Run RAG Pipeline

```bash
python src/rag_pipeline.py --query "genes linked to Parkinson's disease"
```

#### 3. Explore in Notebook

Open `notebooks/RAG_GeneDiscovery_Assistant.ipynb` to see example queries and outputs.

---

## Technologies Used

| Category     | Tool                                        |
| ------------ | ------------------------------------------- |
| Embeddings   | BioBERT, BioSentVec (Hugging Face)          |
| LLM Backend  | DeepSeek / LLaMA3 (Ollama)                  |
| Retrieval    | FAISS                                       |
| Data Sources | PubMed, UniProt, NCBI Gene                  |
| Language     | Python 3.10+                                |
| Frameworks   | LangChain (optional), Sentence Transformers |

---

## Future Enhancements

* Compare **DeepSeek/LLaMA3** with **BioGPT** outputs.
* Integrate **Neo4j** for gene–disease–drug knowledge graph visualization.
* Fine-tune LLMs on curated variant interpretation reports for improved clinical relevance.

---

## Author

**Manish Kumar**
Senior Bioinformatics Software Developer | AI Researcher | Data Science Enthusiast
