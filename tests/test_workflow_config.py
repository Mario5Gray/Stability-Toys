"""
Unit tests for WorkflowConfigManager.

Tests configuration loading, validation, saving, and workflow retrieval.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

import yaml

# Import after setting up path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.workflow_config import (
    WorkflowConfigManager,
    WorkflowConfig,
    WorkflowsYAML,
    get_workflow_config,
    reload_workflow_config,
)


@pytest.fixture
def valid_workflow_yaml():
    """Create valid workflow YAML content."""
    return {
        "default_workflow": "txt2img-basic",
        "workflows": {
            "txt2img-basic": {
                "display_name": "Text to Image (Basic)",
                "description": "Simple txt2img workflow",
                "default_size": "512x512",
                "default_steps": 20,
                "default_cfg": 7.0,
                "tags": ["txt2img", "basic"],
                "workflow": {
                    "3": {"class_type": "KSampler", "inputs": {}},
                    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
                },
            },
            "txt2img-lcm": {
                "display_name": "Text to Image (LCM)",
                "description": "Fast LCM workflow",
                "default_size": "512x512",
                "default_steps": 4,
                "default_cfg": 1.0,
                "tags": ["txt2img", "lcm", "fast"],
                "workflow": {
                    "filepath": "/app/workflows/LCM_WORKFLOW.json"
                },
            },
        },
    }


@pytest.fixture
def temp_workflow_file(valid_workflow_yaml):
    """Create a temporary workflows.yml file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(valid_workflow_yaml, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)
    # Also cleanup .tmp file if it exists
    tmp_path = temp_path + ".tmp"
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def workflow_manager(temp_workflow_file):
    """Create a WorkflowConfigManager with test config."""
    return WorkflowConfigManager(temp_workflow_file)


class TestWorkflowConfigManagerInit:
    """Test WorkflowConfigManager initialization."""

    def test_init_with_valid_config(self, temp_workflow_file):
        """Test initialization with valid config file."""
        manager = WorkflowConfigManager(temp_workflow_file)

        assert manager.config is not None
        assert manager.config.default_workflow == "txt2img-basic"
        assert len(manager.config.workflows) == 2

    def test_init_file_not_found(self):
        """Test initialization with non-existent file raises error."""
        with pytest.raises(FileNotFoundError) as exc_info:
            WorkflowConfigManager("/nonexistent/path/workflows.yml")

        assert "workflows.yml not found" in str(exc_info.value)

    def test_init_empty_path_raises_error(self):
        """Test initialization with empty path raises error."""
        with pytest.raises(ValueError) as exc_info:
            WorkflowConfigManager("")

        assert "not specified" in str(exc_info.value)

    def test_init_empty_file_raises_error(self):
        """Test initialization with empty YAML file raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                WorkflowConfigManager(temp_path)
            assert "empty" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_init_missing_default_workflow(self):
        """Test initialization without default_workflow field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump({"workflows": {"test": {"display_name": "Test"}}}, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                WorkflowConfigManager(temp_path)
            assert "default_workflow" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_init_missing_workflows(self):
        """Test initialization without workflows field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump({"default_workflow": "test"}, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                WorkflowConfigManager(temp_path)
            assert "workflows" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_init_default_workflow_not_in_workflows(self):
        """Test initialization with default_workflow not in workflows dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump({
                "default_workflow": "nonexistent",
                "workflows": {"test": {"display_name": "Test", "workflow": {}}}
            }, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                WorkflowConfigManager(temp_path)
            assert "not found" in str(exc_info.value)
        finally:
            os.unlink(temp_path)


class TestWorkflowConfigParsing:
    """Test workflow configuration parsing."""

    def test_parse_workflow_with_all_fields(self, workflow_manager):
        """Test parsing workflow with all fields present."""
        wf = workflow_manager.get_workflow("txt2img-basic")

        assert wf.name == "txt2img-basic"
        assert wf.display_name == "Text to Image (Basic)"
        assert wf.description == "Simple txt2img workflow"
        assert wf.default_size == "512x512"
        assert wf.default_steps == 20
        assert wf.default_cfg == 7.0
        assert wf.tags == ["txt2img", "basic"]
        assert "3" in wf.workflow
        assert wf.workflow["3"]["class_type"] == "KSampler"

    def test_parse_workflow_with_filepath(self, workflow_manager):
        """Test parsing workflow with filepath reference."""
        wf = workflow_manager.get_workflow("txt2img-lcm")

        assert wf.workflow["filepath"] == "/app/workflows/LCM_WORKFLOW.json"

    def test_parse_workflow_defaults(self):
        """Test that missing optional fields use defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump({
                "default_workflow": "minimal",
                "workflows": {
                    "minimal": {
                        "workflow": {}
                    }
                }
            }, f)
            temp_path = f.name

        try:
            manager = WorkflowConfigManager(temp_path)
            wf = manager.get_workflow("minimal")

            assert wf.display_name == "minimal"  # Falls back to name
            assert wf.description == ""
            assert wf.default_size == "512x512"
            assert wf.default_steps == 20
            assert wf.default_cfg == 7.0
            assert wf.tags == []
        finally:
            os.unlink(temp_path)

    def test_parse_tags_as_string(self):
        """Test parsing tags when provided as comma-separated string."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump({
                "default_workflow": "test",
                "workflows": {
                    "test": {
                        "workflow": {},
                        "tags": "tag1, tag2, tag3"
                    }
                }
            }, f)
            temp_path = f.name

        try:
            manager = WorkflowConfigManager(temp_path)
            wf = manager.get_workflow("test")

            assert wf.tags == ["tag1", "tag2", "tag3"]
        finally:
            os.unlink(temp_path)

    def test_parse_invalid_workflow_json(self):
        """Test that invalid workflow JSON raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump({
                "default_workflow": "test",
                "workflows": {
                    "test": {
                        "workflow": "not a dict"
                    }
                }
            }, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                WorkflowConfigManager(temp_path)
            assert "invalid workflow JSON" in str(exc_info.value)
        finally:
            os.unlink(temp_path)


class TestWorkflowRetrieval:
    """Test workflow retrieval methods."""

    def test_get_workflow_exists(self, workflow_manager):
        """Test getting existing workflow."""
        wf = workflow_manager.get_workflow("txt2img-basic")

        assert wf is not None
        assert wf.name == "txt2img-basic"

    def test_get_workflow_not_exists(self, workflow_manager):
        """Test getting non-existent workflow raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            workflow_manager.get_workflow("nonexistent")

        assert "not found" in str(exc_info.value)
        assert "txt2img-basic" in str(exc_info.value)  # Shows available

    def test_list_workflows(self, workflow_manager):
        """Test listing all workflow names."""
        workflows = workflow_manager.list_workflows()

        assert len(workflows) == 2
        assert "txt2img-basic" in workflows
        assert "txt2img-lcm" in workflows

    def test_get_default_workflow(self, workflow_manager):
        """Test getting default workflow name."""
        default = workflow_manager.get_default_workflow()

        assert default == "txt2img-basic"


class TestWorkflowSaving:
    """Test workflow configuration saving."""

    def test_save_config(self, workflow_manager, temp_workflow_file):
        """Test saving configuration to file."""
        # Modify and save
        data = workflow_manager.to_dict()
        data["workflows"]["new-workflow"] = {
            "display_name": "New Workflow",
            "description": "A new workflow",
            "default_size": "768x768",
            "default_steps": 30,
            "default_cfg": 8.0,
            "tags": ["new"],
            "workflow": {"1": {"class_type": "Test"}},
        }

        workflow_manager.save_config(data)

        # Verify it was saved and reloaded
        assert "new-workflow" in workflow_manager.list_workflows()
        wf = workflow_manager.get_workflow("new-workflow")
        assert wf.display_name == "New Workflow"
        assert wf.default_size == "768x768"

    def test_save_config_changes_default(self, workflow_manager):
        """Test changing default workflow via save."""
        data = workflow_manager.to_dict()
        data["default_workflow"] = "txt2img-lcm"

        workflow_manager.save_config(data)

        assert workflow_manager.get_default_workflow() == "txt2img-lcm"

    def test_save_config_removes_workflow(self, workflow_manager):
        """Test removing a workflow via save."""
        data = workflow_manager.to_dict()
        del data["workflows"]["txt2img-lcm"]

        workflow_manager.save_config(data)

        assert "txt2img-lcm" not in workflow_manager.list_workflows()
        assert len(workflow_manager.list_workflows()) == 1

    def test_save_config_persists_to_disk(self, workflow_manager, temp_workflow_file):
        """Test that save actually writes to disk."""
        data = workflow_manager.to_dict()
        data["workflows"]["persisted"] = {
            "display_name": "Persisted",
            "workflow": {},
        }

        workflow_manager.save_config(data)

        # Read file directly
        with open(temp_workflow_file, 'r') as f:
            saved_data = yaml.safe_load(f)

        assert "persisted" in saved_data["workflows"]


class TestToDict:
    """Test to_dict method."""

    def test_to_dict_includes_workflow(self, workflow_manager):
        """Test to_dict includes workflow JSON by default."""
        data = workflow_manager.to_dict(include_workflow=True)

        assert "workflow" in data["workflows"]["txt2img-basic"]
        assert "3" in data["workflows"]["txt2img-basic"]["workflow"]

    def test_to_dict_excludes_workflow(self, workflow_manager):
        """Test to_dict can exclude workflow JSON."""
        data = workflow_manager.to_dict(include_workflow=False)

        assert "workflow" not in data["workflows"]["txt2img-basic"]

    def test_to_dict_structure(self, workflow_manager):
        """Test to_dict returns correct structure."""
        data = workflow_manager.to_dict()

        assert "default_workflow" in data
        assert "workflows" in data
        assert data["default_workflow"] == "txt2img-basic"

        wf = data["workflows"]["txt2img-basic"]
        assert "display_name" in wf
        assert "description" in wf
        assert "default_size" in wf
        assert "default_steps" in wf
        assert "default_cfg" in wf
        assert "tags" in wf


class TestReload:
    """Test configuration reload."""

    def test_reload_picks_up_changes(self, workflow_manager, temp_workflow_file):
        """Test that reload picks up external file changes."""
        # Externally modify the file
        with open(temp_workflow_file, 'r') as f:
            data = yaml.safe_load(f)

        data["workflows"]["txt2img-basic"]["description"] = "Modified description"

        with open(temp_workflow_file, 'w') as f:
            yaml.dump(data, f)

        # Reload
        workflow_manager.reload()

        wf = workflow_manager.get_workflow("txt2img-basic")
        assert wf.description == "Modified description"


class TestGlobalInstance:
    """Test global instance management."""

    def test_get_workflow_config_returns_same_instance(self, temp_workflow_file):
        """Test that get_workflow_config returns singleton."""
        # Reset global
        import server.workflow_config as wc
        wc._config_manager = None

        manager1 = get_workflow_config(temp_workflow_file)
        manager2 = get_workflow_config()  # Should return same instance

        assert manager1 is manager2

        # Cleanup
        wc._config_manager = None

    def test_reload_workflow_config(self, temp_workflow_file):
        """Test global reload function."""
        import server.workflow_config as wc
        wc._config_manager = None

        manager = get_workflow_config(temp_workflow_file)
        original_default = manager.get_default_workflow()

        # Modify file
        with open(temp_workflow_file, 'r') as f:
            data = yaml.safe_load(f)
        data["default_workflow"] = "txt2img-lcm"
        with open(temp_workflow_file, 'w') as f:
            yaml.dump(data, f)

        reload_workflow_config()

        assert manager.get_default_workflow() == "txt2img-lcm"

        # Cleanup
        wc._config_manager = None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_workflow_with_empty_workflow_dict(self):
        """Test workflow with empty workflow dict is valid."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump({
                "default_workflow": "empty",
                "workflows": {
                    "empty": {
                        "display_name": "Empty Workflow",
                        "workflow": {}
                    }
                }
            }, f)
            temp_path = f.name

        try:
            manager = WorkflowConfigManager(temp_path)
            wf = manager.get_workflow("empty")
            assert wf.workflow == {}
        finally:
            os.unlink(temp_path)

    def test_workflow_with_complex_nested_json(self):
        """Test workflow with deeply nested workflow JSON."""
        complex_workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "nested": {
                        "deep": {
                            "value": [1, 2, 3]
                        }
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump({
                "default_workflow": "complex",
                "workflows": {
                    "complex": {
                        "workflow": complex_workflow
                    }
                }
            }, f)
            temp_path = f.name

        try:
            manager = WorkflowConfigManager(temp_path)
            wf = manager.get_workflow("complex")
            assert wf.workflow["3"]["inputs"]["nested"]["deep"]["value"] == [1, 2, 3]
        finally:
            os.unlink(temp_path)

    def test_workflow_name_with_special_characters(self):
        """Test workflow names with hyphens and underscores."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump({
                "default_workflow": "my-workflow_v2",
                "workflows": {
                    "my-workflow_v2": {
                        "display_name": "My Workflow V2",
                        "workflow": {}
                    }
                }
            }, f)
            temp_path = f.name

        try:
            manager = WorkflowConfigManager(temp_path)
            wf = manager.get_workflow("my-workflow_v2")
            assert wf.name == "my-workflow_v2"
        finally:
            os.unlink(temp_path)

    def test_numeric_default_values(self):
        """Test that numeric values are properly typed."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump({
                "default_workflow": "test",
                "workflows": {
                    "test": {
                        "default_steps": 50,
                        "default_cfg": 12.5,
                        "workflow": {}
                    }
                }
            }, f)
            temp_path = f.name

        try:
            manager = WorkflowConfigManager(temp_path)
            wf = manager.get_workflow("test")
            assert isinstance(wf.default_steps, int)
            assert isinstance(wf.default_cfg, float)
            assert wf.default_steps == 50
            assert wf.default_cfg == 12.5
        finally:
            os.unlink(temp_path)
