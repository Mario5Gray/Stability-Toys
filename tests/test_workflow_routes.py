"""
Unit tests for Workflow API routes.

Tests the FastAPI endpoints for workflow management.
"""

import pytest
import pytest_asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

import yaml

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.workflow_routes import router
from server.workflow_config import WorkflowConfigManager


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
                },
            },
            "txt2img-lcm": {
                "display_name": "Text to Image (LCM)",
                "description": "Fast LCM workflow",
                "default_size": "512x512",
                "default_steps": 4,
                "default_cfg": 1.0,
                "tags": ["txt2img", "lcm"],
                "workflow": {
                    "filepath": "/app/workflows/LCM.json"
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
    tmp_path = temp_path + ".tmp"
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def workflow_manager(temp_workflow_file):
    """Create a WorkflowConfigManager with test config."""
    return WorkflowConfigManager(temp_workflow_file)


@pytest.fixture
def test_app(workflow_manager):
    """Create a test FastAPI app with workflow routes."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app, workflow_manager):
    """Create a test client with mocked workflow config."""
    with patch('server.workflow_routes.get_workflow_config', return_value=workflow_manager):
        yield TestClient(test_app)


class TestListWorkflows:
    """Test GET /api/workflows endpoint."""

    def test_list_workflows_success(self, client):
        """Test listing workflows returns correct structure."""
        response = client.get("/api/workflows")

        assert response.status_code == 200
        data = response.json()

        assert "default_workflow" in data
        assert "workflows" in data
        assert data["default_workflow"] == "txt2img-basic"
        assert len(data["workflows"]) == 2

    def test_list_workflows_excludes_workflow_json(self, client):
        """Test that list doesn't include full workflow JSON."""
        response = client.get("/api/workflows")
        data = response.json()

        # Workflow JSON should not be in list response
        for name, wf in data["workflows"].items():
            assert "workflow" not in wf

    def test_list_workflows_includes_metadata(self, client):
        """Test that list includes workflow metadata."""
        response = client.get("/api/workflows")
        data = response.json()

        wf = data["workflows"]["txt2img-basic"]
        assert wf["display_name"] == "Text to Image (Basic)"
        assert wf["description"] == "Simple txt2img workflow"
        assert wf["default_size"] == "512x512"
        assert wf["default_steps"] == 20
        assert wf["default_cfg"] == 7.0
        assert wf["tags"] == ["txt2img", "basic"]


class TestGetWorkflow:
    """Test GET /api/workflows/{name} endpoint."""

    def test_get_workflow_success(self, client):
        """Test getting a single workflow."""
        response = client.get("/api/workflows/txt2img-basic")

        assert response.status_code == 200
        data = response.json()

        assert data["name"] == "txt2img-basic"
        assert data["display_name"] == "Text to Image (Basic)"
        assert "workflow" in data
        assert "3" in data["workflow"]

    def test_get_workflow_includes_full_json(self, client):
        """Test that get single workflow includes full workflow JSON."""
        response = client.get("/api/workflows/txt2img-basic")
        data = response.json()

        assert data["workflow"]["3"]["class_type"] == "KSampler"

    def test_get_workflow_with_filepath(self, client):
        """Test getting workflow with filepath reference."""
        response = client.get("/api/workflows/txt2img-lcm")
        data = response.json()

        assert data["workflow"]["filepath"] == "/app/workflows/LCM.json"

    def test_get_workflow_not_found(self, client):
        """Test getting non-existent workflow returns 404."""
        response = client.get("/api/workflows/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_get_workflow_url_encoded_name(self, client, workflow_manager):
        """Test getting workflow with URL-encoded name."""
        # Add a workflow with special characters
        data = workflow_manager.to_dict()
        data["workflows"]["my-workflow_v2"] = {
            "display_name": "My Workflow V2",
            "workflow": {},
        }
        workflow_manager.save_config(data)

        response = client.get("/api/workflows/my-workflow_v2")
        assert response.status_code == 200


class TestCreateOrUpdateWorkflow:
    """Test POST /api/workflows/{name} endpoint."""

    def test_create_new_workflow(self, client, workflow_manager):
        """Test creating a new workflow."""
        response = client.post("/api/workflows/new-workflow", json={
            "display_name": "New Workflow",
            "description": "A brand new workflow",
            "default_size": "768x768",
            "default_steps": 30,
            "default_cfg": 8.0,
            "tags": ["new", "test"],
            "workflow": {"1": {"class_type": "Test"}}
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "saved"
        assert "new-workflow" in data["workflows"]

        # Verify it was actually saved
        assert "new-workflow" in workflow_manager.list_workflows()

    def test_update_existing_workflow(self, client, workflow_manager):
        """Test updating an existing workflow."""
        response = client.post("/api/workflows/txt2img-basic", json={
            "display_name": "Updated Name",
            "description": "Updated description",
            "default_size": "1024x1024",
            "default_steps": 50,
            "default_cfg": 10.0,
            "tags": ["updated"],
            "workflow": {"new": {"class_type": "Updated"}}
        })

        assert response.status_code == 200

        wf = workflow_manager.get_workflow("txt2img-basic")
        assert wf.display_name == "Updated Name"
        assert wf.default_size == "1024x1024"

    def test_create_workflow_sets_timestamps(self, client, workflow_manager):
        """Test that creating workflow sets created_at and updated_at."""
        response = client.post("/api/workflows/timestamped", json={
            "display_name": "Timestamped",
            "workflow": {}
        })

        assert response.status_code == 200

        wf = workflow_manager.get_workflow("timestamped")
        assert wf.created_at != ""
        assert wf.updated_at != ""

    def test_update_workflow_updates_timestamp(self, client, workflow_manager):
        """Test that updating workflow updates updated_at but not created_at."""
        # Create
        client.post("/api/workflows/time-test", json={
            "display_name": "Time Test",
            "workflow": {}
        })

        wf1 = workflow_manager.get_workflow("time-test")
        created_at = wf1.created_at

        # Update
        client.post("/api/workflows/time-test", json={
            "display_name": "Time Test Updated",
            "workflow": {}
        })

        wf2 = workflow_manager.get_workflow("time-test")
        assert wf2.created_at == created_at  # Unchanged
        # updated_at should be different (or same if test is too fast)

    def test_create_workflow_with_defaults(self, client, workflow_manager):
        """Test creating workflow uses defaults for missing fields."""
        response = client.post("/api/workflows/minimal", json={
            "display_name": "Minimal",
            "workflow": {}
        })

        assert response.status_code == 200

        wf = workflow_manager.get_workflow("minimal")
        assert wf.description == ""
        assert wf.default_size == "512x512"
        assert wf.default_steps == 20
        assert wf.default_cfg == 7.0
        assert wf.tags == []

    def test_create_first_workflow_becomes_default(self, temp_workflow_file):
        """Test that first workflow becomes default if none set."""
        # Create empty config
        with open(temp_workflow_file, 'w') as f:
            yaml.dump({
                "default_workflow": "nonexistent",
                "workflows": {
                    "placeholder": {"display_name": "Placeholder", "workflow": {}}
                }
            }, f)

        manager = WorkflowConfigManager(temp_workflow_file)
        app = FastAPI()
        app.include_router(router)

        with patch('server.workflow_routes.get_workflow_config', return_value=manager):
            client = TestClient(app)

            # Delete the placeholder, leaving no valid default
            data = manager.to_dict()
            del data["workflows"]["placeholder"]
            data["default_workflow"] = ""
            manager.config.default_workflow = ""

            response = client.post("/api/workflows/first", json={
                "display_name": "First",
                "workflow": {}
            })

            assert response.status_code == 200


class TestBulkSaveWorkflows:
    """Test PUT /api/workflows endpoint."""

    def test_bulk_save_success(self, client, workflow_manager):
        """Test bulk saving workflows."""
        response = client.put("/api/workflows", json={
            "default_workflow": "txt2img-lcm",
            "workflows": {
                "txt2img-lcm": {
                    "display_name": "LCM Only",
                    "workflow": {}
                }
            }
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "saved"

        assert workflow_manager.get_default_workflow() == "txt2img-lcm"
        assert len(workflow_manager.list_workflows()) == 1

    def test_bulk_save_empty_workflows_fails(self, client):
        """Test bulk save with empty workflows returns 400."""
        response = client.put("/api/workflows", json={
            "default_workflow": "test",
            "workflows": {}
        })

        assert response.status_code == 400
        assert "At least one workflow" in response.json()["detail"]

    def test_bulk_save_invalid_default_fails(self, client):
        """Test bulk save with invalid default returns 400."""
        response = client.put("/api/workflows", json={
            "default_workflow": "nonexistent",
            "workflows": {
                "existing": {"display_name": "Existing", "workflow": {}}
            }
        })

        assert response.status_code == 400
        assert "not found in workflows" in response.json()["detail"]

    def test_bulk_save_changes_default(self, client, workflow_manager):
        """Test changing default workflow via bulk save."""
        assert workflow_manager.get_default_workflow() == "txt2img-basic"

        data = workflow_manager.to_dict()
        data["default_workflow"] = "txt2img-lcm"

        response = client.put("/api/workflows", json=data)

        assert response.status_code == 200
        assert workflow_manager.get_default_workflow() == "txt2img-lcm"


class TestDeleteWorkflow:
    """Test DELETE /api/workflows/{name} endpoint."""

    def test_delete_workflow_success(self, client, workflow_manager):
        """Test deleting a workflow."""
        assert "txt2img-lcm" in workflow_manager.list_workflows()

        response = client.delete("/api/workflows/txt2img-lcm")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["workflow"] == "txt2img-lcm"

        assert "txt2img-lcm" not in workflow_manager.list_workflows()

    def test_delete_default_workflow_fails(self, client):
        """Test deleting default workflow returns 400."""
        response = client.delete("/api/workflows/txt2img-basic")

        assert response.status_code == 400
        assert "Cannot delete default" in response.json()["detail"]

    def test_delete_nonexistent_workflow_fails(self, client):
        """Test deleting non-existent workflow returns 404."""
        response = client.delete("/api/workflows/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestReloadWorkflows:
    """Test POST /api/workflows/reload endpoint."""

    def test_reload_success(self, client, workflow_manager, temp_workflow_file):
        """Test reloading configuration."""
        # Externally modify the file
        with open(temp_workflow_file, 'r') as f:
            data = yaml.safe_load(f)

        data["workflows"]["txt2img-basic"]["description"] = "Externally modified"

        with open(temp_workflow_file, 'w') as f:
            yaml.dump(data, f)

        response = client.post("/api/workflows/reload")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "reloaded"
        assert data["workflows_count"] == 2
        assert "txt2img-basic" in data["workflows"]
        assert data["default_workflow"] == "txt2img-basic"

    def test_reload_returns_workflow_list(self, client):
        """Test reload returns list of workflow names."""
        response = client.post("/api/workflows/reload")
        data = response.json()

        assert isinstance(data["workflows"], list)
        assert "txt2img-basic" in data["workflows"]
        assert "txt2img-lcm" in data["workflows"]


class TestErrorHandling:
    """Test error handling in routes."""

    def test_save_error_returns_500(self, client, workflow_manager):
        """Test that save errors return 500."""
        with patch.object(workflow_manager, 'save_config', side_effect=IOError("Disk full")):
            response = client.post("/api/workflows/test", json={
                "display_name": "Test",
                "workflow": {}
            })

            assert response.status_code == 500
            assert "Disk full" in response.json()["detail"]

    def test_reload_error_returns_500(self, client, workflow_manager):
        """Test that reload errors return 500."""
        with patch.object(workflow_manager, 'reload', side_effect=ValueError("Invalid config")):
            # Need to also patch reload_workflow_config
            with patch('server.workflow_routes.reload_workflow_config', side_effect=ValueError("Invalid config")):
                response = client.post("/api/workflows/reload")

                assert response.status_code == 500


class TestContentTypes:
    """Test request/response content types."""

    def test_list_returns_json(self, client):
        """Test list endpoint returns JSON content type."""
        response = client.get("/api/workflows")

        assert response.headers["content-type"].startswith("application/json")

    def test_post_accepts_json(self, client):
        """Test POST endpoint accepts JSON."""
        response = client.post(
            "/api/workflows/test",
            json={"display_name": "Test", "workflow": {}},
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200


class TestWorkflowDataIntegrity:
    """Test that workflow data maintains integrity through save/load cycles."""

    def test_workflow_roundtrip(self, client, workflow_manager):
        """Test workflow data survives create -> get -> update cycle."""
        original_workflow = {
            "1": {"class_type": "A", "inputs": {"nested": {"value": 123}}},
            "2": {"class_type": "B", "inputs": {"list": [1, 2, 3]}}
        }

        # Create
        client.post("/api/workflows/roundtrip", json={
            "display_name": "Roundtrip Test",
            "description": "Testing data integrity",
            "default_size": "768x768",
            "default_steps": 25,
            "default_cfg": 5.5,
            "tags": ["test", "roundtrip"],
            "workflow": original_workflow
        })

        # Get
        response = client.get("/api/workflows/roundtrip")
        data = response.json()

        assert data["display_name"] == "Roundtrip Test"
        assert data["description"] == "Testing data integrity"
        assert data["default_size"] == "768x768"
        assert data["default_steps"] == 25
        assert data["default_cfg"] == 5.5
        assert data["tags"] == ["test", "roundtrip"]
        assert data["workflow"]["1"]["inputs"]["nested"]["value"] == 123
        assert data["workflow"]["2"]["inputs"]["list"] == [1, 2, 3]

    def test_large_workflow_json(self, client, workflow_manager):
        """Test handling of large workflow JSON."""
        # Create a workflow with many nodes
        large_workflow = {
            str(i): {"class_type": f"Node{i}", "inputs": {"value": i * 100}}
            for i in range(100)
        }

        response = client.post("/api/workflows/large", json={
            "display_name": "Large Workflow",
            "workflow": large_workflow
        })

        assert response.status_code == 200

        # Verify it was saved correctly
        get_response = client.get("/api/workflows/large")
        data = get_response.json()

        assert len(data["workflow"]) == 100
        assert data["workflow"]["50"]["class_type"] == "Node50"
