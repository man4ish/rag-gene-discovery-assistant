import pytest
from unittest.mock import patch, mock_open
import update_config

# Mock content of config.py
CONFIG_CONTENT = """WORKSPACE_DIR = "/old/path"
BASE_PATH = "/old/path/data/PubMed"
TOP_K = 5
USE_GPU = True
OTHER_SETTING = "keep"
"""

@pytest.fixture
def mock_file():
    m = mock_open(read_data=CONFIG_CONTENT)
    with patch("builtins.open", m):
        yield m

@patch("os.path.exists")
def test_update_all_parameters(mock_exists, mock_file):
    mock_exists.return_value = True
    workspace = "/new/workspace"
    top_k = 20
    use_gpu = False

    update_config.update_config_file(workspace_dir=workspace, top_k=top_k, use_gpu=use_gpu)

    handle = mock_file()
    written_lines = [call.args[0] for call in handle.write.call_args_list] if handle.write.call_args_list else handle.writelines.call_args[0][0]

    # Check that WORKSPACE_DIR and BASE_PATH were updated
    assert f'WORKSPACE_DIR = r"{workspace}"\n' in written_lines
    assert f'BASE_PATH = r"{workspace}/data/PubMed"\n' in written_lines
    assert f"TOP_K = {top_k}\n" in written_lines
    assert f"USE_GPU = {use_gpu}\n" in written_lines
    # Ensure OTHER_SETTING remains
    assert 'OTHER_SETTING = "keep"\n' in written_lines

@patch("os.path.exists")
def test_missing_config_file(mock_exists):
    mock_exists.return_value = False
    with pytest.raises(FileNotFoundError):
        update_config.update_config_file(workspace_dir="/x")

@patch("os.path.exists")
def test_update_partial_parameters(mock_exists, mock_file):
    mock_exists.return_value = True
    workspace = "/workspace/path"
    update_config.update_config_file(workspace_dir=workspace)

    handle = mock_file()
    written_lines = handle.writelines.call_args[0][0]
    assert f'WORKSPACE_DIR = r"{workspace}"\n' in written_lines
    assert f'BASE_PATH = r"{workspace}/data/PubMed"\n' in written_lines
    # TOP_K and USE_GPU remain unchanged
    assert "TOP_K = 5\n" in written_lines
    assert "USE_GPU = True\n" in written_lines
