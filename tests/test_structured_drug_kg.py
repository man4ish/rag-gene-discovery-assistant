import pytest
from unittest.mock import patch, MagicMock
import ragbio.structured_drug_kg as kg

# -----------------------------
# Test add_structured_data_to_kg
# -----------------------------
def test_add_structured_data_to_kg():
    # Sample structured data
    data = [
        {
            "drug": "DrugA",
            "targets": [
                {"target": "Target1", "cancer": "Cancer1", "mechanism": "Mechanism1"}
            ],
            "pmid": "111"
        }
    ]

    # Mock driver.session
    mock_session = MagicMock()
    mock_session.execute_write = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch.object(kg, "driver", mock_driver):
        kg.add_structured_data_to_kg(data)

    # Check that execute_write was called once for the target
    mock_session.execute_write.assert_called_once()
    called_args = mock_session.execute_write.call_args[0]
    assert called_args[1] == "Target1" or "DrugA" in called_args  # target and drug passed

# -----------------------------
# Test query_drugs_by_target
# -----------------------------
def test_query_drugs_by_target():
    # Mock result of tx.run
    mock_record = [{"drug": "DrugA"}, {"drug": "DrugB"}]
    mock_tx = MagicMock()
    mock_tx.run.return_value = mock_record
    mock_session = MagicMock()
    mock_session.execute_read.side_effect = lambda func: func(mock_tx)

    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch.object(kg, "driver", mock_driver):
        drugs = kg.query_drugs_by_target("Target1", "Cancer1")

    # Check that returned list matches
    assert drugs == ["DrugA", "DrugB"]
    mock_tx.run.assert_called_once()
