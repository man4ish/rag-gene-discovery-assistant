import pytest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
from ragbio import pubmed_fetcher as pf

# -----------------------------
# 1. Test get_total_records
# -----------------------------
@patch("ragbio.pubmed_fetcher.Entrez.esearch")
@patch("ragbio.pubmed_fetcher.Entrez.read")
def test_get_total_records(mock_read, mock_esearch):
    mock_read.return_value = {"Count": "42"}
    total = pf.get_total_records("cancer")
    assert total == 42
    mock_esearch.assert_called_once()

# -----------------------------
# 2. Test fetch_pubmed_ids with retries
# -----------------------------
@patch("ragbio.pubmed_fetcher.Entrez.esearch")
@patch("ragbio.pubmed_fetcher.Entrez.read")
def test_fetch_pubmed_ids_success(mock_read, mock_esearch):
    mock_read.return_value = {"IdList": ["123", "456"]}
    ids = pf.fetch_pubmed_ids("cancer", retmax=2)
    assert ids == ["123", "456"]

# -----------------------------
# 3. Test fetch_abstract_and_metadata
# -----------------------------
@patch("ragbio.pubmed_fetcher.Entrez.efetch")
@patch("ragbio.pubmed_fetcher.Entrez.read")
def test_fetch_abstract_and_metadata(mock_read, mock_efetch):
    mock_read.return_value = {
        "PubmedArticle": [
            {"MedlineCitation": {"Article": {
                "ArticleTitle": "Test Title",
                "Abstract": {"AbstractText": ["This is an abstract."]},
                "AuthorList": [{"LastName":"Doe","ForeName":"John"}]
            }}}
        ]
    }
    record = pf.fetch_abstract_and_metadata("123")
    assert record["pmid"] == "123"
    assert record["title"] == "Test Title"
    assert record["abstract"] == "This is an abstract."
    assert record["authors"] == ["Doe John"]

# -----------------------------
# 4. Test save_to_json
# -----------------------------
def test_save_to_json(tmp_path):
    data = {"pmid":"123", "abstract":"Test"}
    filepath = pf.save_to_json(data, tmp_path, "123")
    assert os.path.exists(filepath)
    with open(filepath) as f:
        loaded = json.load(f)
    assert loaded == data

# -----------------------------
# 5. Test try_download_pdf
# -----------------------------
@patch("ragbio.pubmed_fetcher.requests.get")
def test_try_download_pdf(mock_get, tmp_path):
    # Mock HTML response with PDF link
    html = '<a href="/pmc/articles/PMC12345/pdf" >PDF</a>'
    mock_get.return_value.text = html.encode('utf-8') if hasattr(html, 'encode') else html
    mock_get.return_value.content = b"%PDF-1.4"
    
    with patch("ragbio.pubmed_fetcher.config.PDF_FOLDER", tmp_path):
        result = pf.try_download_pdf("123")
        assert result  # PDF should be "downloaded"
        pdf_file = tmp_path / "123.pdf"
        assert pdf_file.exists()
        with open(pdf_file, "rb") as f:
            content = f.read()
        assert content.startswith(b"%PDF")

# -----------------------------
# 6. Test get_processed_pmids
# -----------------------------
def test_get_processed_pmids(tmp_path):
    (tmp_path / "123.json").write_text("{}")
    (tmp_path / "456.json").write_text("{}")
    pmids = pf.get_processed_pmids(tmp_path)
    assert pmids == {"123", "456"}
