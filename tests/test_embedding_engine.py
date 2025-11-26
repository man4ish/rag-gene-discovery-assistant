
import os
import json
import pytest
import numpy as np
from unittest.mock import patch, mock_open, MagicMock

import ragbio.embedding_engine as ee
from ragbio import config

# -----------------------------
# 1. Test load_abstracts
# -----------------------------
def test_load_abstracts(tmp_path):
    # Create fake JSON abstract files
    abstracts_dir = tmp_path / "abstracts"
    abstracts_dir.mkdir()
    file1 = abstracts_dir / "1.json"
    file1.write_text(json.dumps({"pmid": "123", "abstract": "Test abstract 1"}))
    file2 = abstracts_dir / "2.json"
    file2.write_text(json.dumps({"pmid": "124", "abstract": "Test abstract 2"}))
    # Patch config.ABSTRACT_FOLDER
    with patch.object(config, "ABSTRACT_FOLDER", str(abstracts_dir)):
        abstracts, pmids = ee.load_abstracts()
        assert len(abstracts) == 2
        assert "Test abstract 1" in abstracts
        assert pmids == ["123", "124"]

# -----------------------------
# 2. Test generate_embeddings
# -----------------------------
@patch("ragbio.embedding_engine.ollama.embeddings")
@patch("builtins.open", new_callable=mock_open)
def test_generate_embeddings(mock_file, mock_ollama):
    texts = ["text1", "text2"]
    mock_ollama.return_value = {"embedding": [0.1]*768}
    emb_array = ee.generate_embeddings(texts, "dummy_model")
    assert emb_array.shape == (2, 768)
    np.testing.assert_almost_equal(emb_array[0], np.array([0.1]*768, dtype="float32"))

# -----------------------------
# 3. Test build_faiss_index
# -----------------------------
def test_build_faiss_index_cpu():
    # Disable GPU for test
    with patch.object(config, "USE_GPU", False):
        data = np.random.rand(5, 768).astype("float32")
        index = ee.build_faiss_index(data)
        assert index.ntotal == 5

# -----------------------------
# 4. Test save_index
# -----------------------------
def test_save_index(tmp_path):
    pmids = ["123", "124"]
    index = ee.build_faiss_index(np.random.rand(2, 768).astype("float32"))
    with patch.object(config, "INDEX_FOLDER", tmp_path):
        with patch.object(config, "INDEX_FILE", tmp_path / "index.faiss"), \
             patch.object(config, "ID_MAP_FILE", tmp_path / "id_map.json"):
            ee.save_index(index, pmids)
            # Check files exist
            assert (tmp_path / "index.faiss").exists()
            assert (tmp_path / "id_map.json").exists()
            with open(tmp_path / "id_map.json") as f:
                data = json.load(f)
                assert data == pmids
