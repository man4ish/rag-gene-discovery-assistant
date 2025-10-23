"""
data_loader.py
---------------
Fetch PubMed abstracts, metadata, and PDFs.
Stores data in structured folders on external storage (configured via config.py).
"""

import os
import json
import requests
from bs4 import BeautifulSoup
from Bio import Entrez
import config


Entrez.email = "mandecent.gupta@gmail.com"  # Replace with your email
SEARCH_TERM = "cancer AND (drug OR therapy OR treatment)"
RETMAX = 1000  # Number of articles to fetch for testing (adjust as needed)


# Create output directories if they do not exist
for folder in [
    config.ABSTRACT_FOLDER,
    config.METADATA_FOLDER,
    config.PDF_FOLDER,
]:
    os.makedirs(folder, exist_ok=True)


def fetch_pubmed_ids(term, retmax=1000):
    """Fetch PubMed IDs based on search term."""
    handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax)
    record = Entrez.read(handle)
    return record.get("IdList", [])


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
    """Save JSON file."""
    filepath = os.path.join(folder, f"{filename}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filepath


def try_download_pdf(pmid):
    """Try downloading an open-access PDF (if available on PMC)."""
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
        print(f"PDF downloaded for PMID {pmid}")
        return True
    return False


def main():
    print(f"Searching PubMed for: {SEARCH_TERM}")
    pmids = fetch_pubmed_ids(SEARCH_TERM, RETMAX)
    print(f"Found {len(pmids)} records.\n")

    for pmid in pmids:
        try:
            record = fetch_abstract_and_metadata(pmid)

            # Save abstract and metadata
            save_to_json(
                {"pmid": pmid, "title": record["title"], "abstract": record["abstract"]},
                config.ABSTRACT_FOLDER,
                pmid,
            )
            save_to_json(record, config.METADATA_FOLDER, f"{pmid}_meta")

            # Attempt PDF download
            try_download_pdf(pmid)
            print(f"Saved PMID {pmid} - {record['title'][:80]}")

        except Exception as e:
            print(f"Error with PMID {pmid}: {e}")

    print("\nData loading complete.")


if __name__ == "__main__":
    main()
