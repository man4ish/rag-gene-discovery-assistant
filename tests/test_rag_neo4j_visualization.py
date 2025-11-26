import pytest
import json
import os
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

# -----------------------------
# Import your modules
# -----------------------------
import ragbio.embedding_engine as ee
import ragbio.structured_drug_kg as kg
import ragbio.rag_neo4j_streamlit as rn

# ==========================================
# 1. embedding_engine.py tests
# ==========================================

def test_load_abstracts(tmp_path):
    # Create fake JSON abstract files
    abstracts_dir = tmp_path / "abstracts"
    abstracts_dir.mkdir()
    file1 = abstracts_dir / "1.json"
    file1.write_text(json.dumps({"pmid": "123", "abstract": "Test abstract 1"}))
    file2 = abstracts_dir / "2.json"
    file2.write_text(json.dumps({"pmid": "124", "abstract": "Test abstract 2"}))

    with patch.object(ee.config, "ABSTRACT_FOLDER", str(abstracts_dir)):
        abstracts, pmids = ee.load_abstracts()
        assert len(abstracts) == 2
        assert "Test abstract 1" in abstracts
        assert pmids == ["123", "124"]

@patch("ragbio.embedding_engine.ollama.embeddings")
@patch("builtins.open", new_callable=mock_open)
def test_generate_embeddings(mock_file, mock_ollama):
    texts = ["text1", "text2"]
    mock_ollama.return_value = {"embedding": [0.1]*768}
    emb_array = ee.generate_embeddings(texts, "dummy_model")
    assert emb_array.shape == (2, 768)
    np.testing.assert_almost_equal(emb_array[0], np.array([0.1]*768, dtype="float32"))

def test_build_faiss_index_cpu():
    with patch.object(ee.config, "USE_GPU", False):
        data = np.random.rand(5, 768).astype("float32")
        index = ee.build_faiss_index(data)
        assert index.ntotal == 5

def test_save_index(tmp_path):
    pmids = ["123", "124"]
    index = ee.build_faiss_index(np.random.rand(2, 768).astype("float32"))
    with patch.object(ee.config, "INDEX_FOLDER", tmp_path):
        with patch.object(ee.config, "INDEX_FILE", tmp_path / "index.faiss"), \
             patch.object(ee.config, "ID_MAP_FILE", tmp_path / "id_map.json"):
            ee.save_index(index, pmids)
            assert (tmp_path / "index.faiss").exists()
            assert (tmp_path / "id_map.json").exists()
            with open(tmp_path / "id_map.json") as f:
                data = json.load(f)
                assert data == pmids

# ==========================================
# 2. structured_drug_kg.py tests
# ==========================================

def test_add_structured_data_to_kg():
    data = [
        {"drug": "DrugA", "targets": [{"target": "Target1", "cancer": "Cancer1", "mechanism": "M1"}], "pmid": "111"}
    ]
    mock_session = MagicMock()
    mock_session.execute_write = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch.object(kg, "driver", mock_driver):
        kg.add_structured_data_to_kg(data)

    mock_session.execute_write.assert_called_once()

def test_query_drugs_by_target():
    mock_record = [{"drug": "DrugA"}, {"drug": "DrugB"}]
    mock_tx = MagicMock()
    mock_tx.run.return_value = mock_record
    mock_session = MagicMock()
    mock_session.execute_read.side_effect = lambda func: func(mock_tx)
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch.object(kg, "driver", mock_driver):
        drugs = kg.query_drugs_by_target("Target1", "Cancer1")

    assert drugs == ["DrugA", "DrugB"]
    mock_tx.run.assert_called_once()

# ==========================================
# 3. rag_neo4j_streamlit.py tests
# ==========================================

def test_load_json_to_neo4j():
    sample_data = [{"target": "T1", "drug": "D1", "disease": "C1", "pmid": "123"}]
    mock_file = mock_open(read_data=json.dumps(sample_data))
    with patch("builtins.open", mock_file), \
         patch("os.path.exists", return_value=True), \
         patch.object(rn, "graph") as mock_graph, \
         patch("streamlit.warning"), patch("streamlit.success"):
        mock_graph.merge = MagicMock()
        rn.load_json_to_neo4j("dummy.json")
        assert mock_graph.merge.call_count >= 4

def test_load_all_jsons_to_neo4j(tmp_path):
    file1 = tmp_path / "a.json"
    file1.write_text(json.dumps([{"target": "T1"}]))
    file2 = tmp_path / "b.json"
    file2.write_text(json.dumps([{"target": "T2"}]))
    with patch.object(rn, "load_json_to_neo4j") as mock_load_json, \
         patch("glob.glob", return_value=[str(file1), str(file2)]), \
         patch("streamlit.warning"), patch("streamlit.success"):
        rn.load_all_jsons_to_neo4j(tmp_path)
        assert mock_load_json.call_count == 2

def test_fetch_cy_elements():
    mock_record = {
        'n': MagicMock(labels={"Target"}, name="T1"),
        'm': MagicMock(labels={"Drug"}, name="D1"),
        'r': MagicMock(__class__=type("Rel", (), {}))
    }
    with patch.object(rn, "graph") as mock_graph:
        mock_graph.run.return_value.data.return_value = [mock_record]
        elements = rn.fetch_cy_elements(node_types=["target", "drug"])
        assert any(e['data']['id'] == "T1" for e in elements)
        assert any(e['data']['id'] == "D1" for e in elements)
