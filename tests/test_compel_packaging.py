import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compel_pin_is_isolated_from_notebook_dependency_resolution():
    conditioning = (ROOT / "requirements-conditioning.txt").read_text()
    runtime = (ROOT / "requirements.txt").read_text()
    requirement_lines = [
        line.strip().lower()
        for line in conditioning.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert "compel==2.3.1" in conditioning
    assert "compel" not in runtime.lower()
    assert not any("notebook" in line for line in requirement_lines)
    assert not any("jupyter" in line for line in requirement_lines)
    assert "pyparsing~=3.0" in runtime


def test_transformers_major_is_capped_in_runtime_and_test_requirements():
    assert "transformers>=4.30.0,<5.0" in (ROOT / "requirements.txt").read_text()
    assert "transformers>=4.30.0,<5.0" in (
        ROOT / "requirements-test.txt"
    ).read_text()


def test_images_install_compel_without_declared_notebook_dependencies():
    for filename in ("Dockerfile", "Dockerfile.test"):
        text = (ROOT / filename).read_text()
        assert "requirements-conditioning.txt" in text
        assert re.search(
            r"pip install[^\n\\]*--no-deps[^\n\\]*requirements-conditioning\.txt",
            text,
        )
        assert "version('compel') == '2.3.1'" in text
