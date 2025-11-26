import pytest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import json
from ragbio import rag_pipeline as rp
from ragbio.rag_pipeline import RAGAssistant, FAISSPMIDRetriever
from langchain_core.documents import Document

# -----------------------------
# 1. Test _get_abstract_text
# -----------------------------
def test_get_abstract_text(tmp_path):
    pmid_file = tmp_path / "123.json"
    pmid_file.write_text(json.dumps({"pmid": "123", "abstract": "Test abstract"}))

    retriever = FAISSPMIDRetriever(
        embeddings=MagicMock(),
        index=MagicMock(),
        pmid_map=["123"],
        abstract_folder=str(tmp_path),
        k=1
    )
    text = retriever._get_abstract_text("123")
    assert text == "Test abstract"

# -----------------------------
# 2. Test _get_relevant_documents
# -----------------------------
def test_get_relevant_documents():
    fake_embedding = [0.1]*768
    mock_index = MagicMock()
    mock_index.search.return_value = (np.array([[0.1]]), np.array([[0]]))
    retriever = FAISSPMIDRetriever(
        embeddings=MagicMock(embed_query=lambda q: fake_embedding),
        index=mock_index,
        pmid_map=["123"],
        abstract_folder=".",
        k=1
    )

    with patch.object(retriever, "_get_abstract_text", return_value="Abstract text"):
        docs = retriever._get_relevant_documents("query")
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "Abstract text"

# -----------------------------
# 3. Test extract_structured_info
# -----------------------------
def test_extract_structured_info():
    assistant = RAGAssistant()
    mock_response = '[{"drug":"D","targets":[{"target":"T","cancer":"C"}]}]'
    assistant.chat_model.invoke = MagicMock(return_value=mock_response)
    res = assistant.extract_structured_info("D", "Abstract")
    assert isinstance(res, list)
    assert res[0]["drug"] == "D"

# -----------------------------
# 4. Test run_pipeline (simplified)
# -----------------------------
def test_run_pipeline():
    assistant = RAGAssistant()
    # Mock all external calls
    assistant.rag_chain.invoke = MagicMock(return_value="Summary text")
    assistant.retriever._get_relevant_documents = MagicMock(return_value=[
        Document(page_content="A", metadata={"source":"123", "score":1.0})
    ])
    assistant.get_abstract_text = MagicMock(return_value="Abstract A")
    assistant.extract_structured_info = MagicMock(return_value=[{"drug":"D","targets":[{"target":"T","cancer":"C"}]}])
    with patch("ragbio.rag_pipeline.add_structured_data_to_kg") as mock_add_kg:
        summary, pmids, structured = assistant.run_pipeline("query", top_k=1, structured=True)
        assert summary == "Summary text"
        assert pmids == ["123"]
        assert structured[0]["drug"] == "D"
        mock_add_kg.assert_called_once()

# -----------------------------
# 5. Test save_output JSON writing
# -----------------------------
def test_save_output(tmp_path):
    assistant = RAGAssistant(output_dir=str(tmp_path))
    structured_data = [{"drug":"D","targets":[{"target":"T","cancer":"C"}]}]
    assistant.save_output("query", "Summary", ["123"], structured_data)
    files = list(tmp_path.glob("*_output.json"))
    assert len(files) == 1
    with open(files[0]) as f:
        data = json.load(f)
        assert data == structured_data
