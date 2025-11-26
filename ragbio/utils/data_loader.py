#!/usr/bin/env python3

"""
pubmed_fetcher.py

Fetch abstracts, metadata, and open-access PDFs from PubMed for a given search term.
The script uses Biopython's Entrez API and optionally scrapes PMC for PDFs. 
It organizes downloaded content into structured folders and JSON files for downstream processing.

Features:
- Configurable search term, email, batch size, and starting index via environment variables.
- Automatic creation of folders for abstracts, metadata, and PDFs.
- Handles retries for PubMed API requests and resumes from previously processed PMIDs.
- Fetches article title, abstract, authors, and optionally downloads open-access PDFs from PMC.
- Saves each record as JSON for easy downstream integration (e.g., RAG pipelines).

Configuration:
- Environment Variables:
    - NCBI_EMAIL      : Email used for NCBI Entrez API (default: "mandecent.gupta@gmail.com")
    - PUBMED_SEARCH   : PubMed search term (default: "cancer AND (drug OR therapy OR treatment)")
    - PUBMED_RETMAX   : Number of records per batch (default: 1000)
    - PUBMED_RETSTART : Starting index for fetching records (default: 0)
- Uses folder paths defined in `ragbio.config`:
    - ABSTRACT_FOLDER : stores JSON abstracts
    - METADATA_FOLDER : stores JSON metadata
    - PDF_FOLDER      : stores PDFs (if available)

Dependencies:
- biopython (Entrez)
- requests
- beautifulsoup4
- ragbio.config
- json, os, time, urllib

Usage:
    python pubmed_fetcher.py

Author:
    Manish Kumar
"""


import os
import json
import time
import requests
from bs4 import BeautifulSoup
from Bio import Entrez
from urllib.error import HTTPError, URLError
from ragbio import config


# ==========================================
# Configuration
# ==========================================
Entrez.api_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
Entrez.email = os.getenv("NCBI_EMAIL", "mandecent.gupta@gmail.com")
SEARCH_TERM = os.getenv("PUBMED_SEARCH", "cancer AND (drug OR therapy OR treatment)")
RETMAX = int(os.getenv("PUBMED_RETMAX", 1000))
RETSTART = int(os.getenv("PUBMED_RETSTART", 0))

# ==========================================
# Setup
# ==========================================
for folder in [config.ABSTRACT_FOLDER, config.METADATA_FOLDER, config.PDF_FOLDER]:
    os.makedirs(folder, exist_ok=True)

print(f"\n[INFO] Data folders initialized:")
print(f"  Abstracts → {config.ABSTRACT_FOLDER}")
print(f"  Metadata  → {config.METADATA_FOLDER}")
print(f"  PDFs      → {config.PDF_FOLDER}\n")


# ==========================================
# Helper functions
# ==========================================
def get_total_records(term):
    """Get total record count for a search term."""
    handle = Entrez.esearch(db="pubmed", term=term, retmax=1)
    record = Entrez.read(handle)
    return int(record.get("Count", 0))


def fetch_pubmed_ids(term, retmax=1000, retstart=0, retries=3, delay=5):
    """Fetch PubMed IDs with retry and pagination support."""
    for attempt in range(1, retries + 1):
        try:
            handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax, retstart=retstart)
            record = Entrez.read(handle)
            return record.get("IdList", [])
        except (HTTPError, URLError, RuntimeError) as e:
            print(f"[WARN] Attempt {attempt} failed: {e}")
            time.sleep(delay)
    print("[ERROR] Failed to fetch PubMed IDs after multiple attempts.")
    return []


def fetch_abstract_and_metadata(pmid):
    """Fetch abstract, metadata, and optionally PDF for a PubMed ID."""
    handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
    records = Entrez.read(handle)
    article = records["PubmedArticle"][0]["MedlineCitation"]["Article"]

    title = article.get("ArticleTitle", "")
    abstract = " ".join(article.get("Abstract", {}).get("AbstractText", [])) if "Abstract" in article else ""

    authors = []
    if "AuthorList" in article:
        for a in article["AuthorList"]:
            name = f"{a.get('LastName', '')} {a.get('ForeName', '')}".strip()
            if name:
                authors.append(name)

    return {"pmid": pmid, "title": title, "abstract": abstract, "authors": authors}


def save_to_json(data, folder, filename):
    """Save JSON file safely."""
    filepath = os.path.join(folder, f"{filename}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filepath


def try_download_pdf(pmid):
    """Try downloading an open-access PDF (if available on PMC)."""
    try:
        pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/?term={pmid}"
        response = requests.get(pmc_url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        pdf_link = soup.find("a", string="PDF")

        if pdf_link:
            pdf_href = pdf_link.get("href")
            if pdf_href.startswith("/"):
                pdf_href = "https://www.ncbi.nlm.nih.gov" + pdf_href

            pdf_response = requests.get(pdf_href, timeout=10)
            pdf_path = os.path.join(config.PDF_FOLDER, f"{pmid}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(pdf_response.content)
            print(f"[PDF] Downloaded for PMID {pmid}")
            return True
    except Exception as e:
        print(f"[WARN] PDF not available for PMID {pmid}: {e}")
    return False


def get_processed_pmids(folder):
    """Get PMIDs already downloaded to resume from last batch."""
    pmids = []
    for fname in os.listdir(folder):
        if fname.endswith(".json"):
            pmids.append(fname.replace(".json", ""))
    return set(pmids)


# ==========================================
# Main
# ==========================================
def main():
    total = get_total_records(SEARCH_TERM)
    print(f"[INFO] Total available records for '{SEARCH_TERM}': {total}")

    processed = get_processed_pmids(config.ABSTRACT_FOLDER)
    print(f"[INFO] Already processed: {len(processed)} PMIDs.")

    retstart = RETSTART
    while retstart < total:
        pmids = fetch_pubmed_ids(SEARCH_TERM, RETMAX, retstart)
        if not pmids:
            break

        print(f"\n[INFO] Fetching batch starting at {retstart}, size={len(pmids)}")

        for pmid in pmids:
            if pmid in processed:
                print(f"[SKIP] PMID {pmid} already exists.")
                continue

            try:
                record = fetch_abstract_and_metadata(pmid)
                save_to_json({"pmid": pmid, "title": record["title"], "abstract": record["abstract"]},
                             config.ABSTRACT_FOLDER, pmid)
                save_to_json(record, config.METADATA_FOLDER, f"{pmid}_meta")

                try_download_pdf(pmid)
                processed.add(pmid)
                print(f"[OK] PMID {pmid} - {record['title'][:80]}")

            except Exception as e:
                print(f"[ERROR] Problem with PMID {pmid}: {e}")

        retstart += RETMAX
        print(f"[INFO] Completed batch. Next start index: {retstart}")
        time.sleep(2)

    print(f"\n[INFO] All done. Total processed: {len(processed)}")


if __name__ == "__main__":
    main()
