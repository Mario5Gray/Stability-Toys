# Setting Up the Yume Test Suite

## Directory Structure

Your project should have this structure:

```
yume/                          # Root project directory
├── yume/                      # Main package directory
│   ├── __init__.py           # Package init
│   ├── dream_worker.py       # DreamWorker class
│   ├── scoring.py            # Scoring classes
│   ├── backends/             # Backend implementations
│   │   ├── __init__.py
│   │   └── base.py
│   └── ... other modules
│
├── tests/                     # Test directory
│   ├── __init__.py           # Makes tests a package
│   ├── conftest.py           # Shared fixtures
│   ├── pytest.ini            # Pytest config
│   ├── test_dream_worker.py  # DreamWorker tests
│   ├── test_scoring.py       # Scoring tests
│   └── test_integration.py   # Integration tests
│
├── requirements.txt           # Main dependencies
├── requirements-test.txt      # Test dependencies
├── run_tests.py              # Test runner script (in root)
├── setup.py                  # Package setup
└── README.md
```

## Step-by-Step Setup

### 1. Create the Directory Structure

```bash
cd /path/to/your/project

# Create main package directory (if not exists)
mkdir -p yume

# Create tests directory
mkdir -p tests

# Create __init__.py files
touch yume/__init__.py
touch tests/__init__.py
```

### 2. Move Your Source Files

Make sure your Yume source files are in `yume/`:

```bash
# Your structure should be:
yume/
├── __init__.py
├── dream_worker.py
├── scoring.py
└── backends/
    ├── __init__.py
    └── base.py
```

### 3. Copy Test Files to tests/

```bash
# Copy all test files to tests/ directory
cp test_*.py tests/
cp conftest.py tests/
cp pytest.ini tests/
cp requirements-test.txt .  # Keep in root
```

### 4. Update yume/__init__.py

```python
# yume/__init__.py
"""
Yume - Latent Space Exploration Library
"""

__version__ = "0.1.0"

# Import main classes for easy access
from .dream_worker import DreamWorker, DreamCandidate
from .scoring import CLIPScorer, AestheticScorer, CompositeScorer

__all__ = [
    "DreamWorker",
    "DreamCandidate", 
    "CLIPScorer",
    "AestheticScorer",
    "CompositeScorer",
]
```

### 5. Create tests/__init__.py

```python
# tests/__init__.py
"""
Test suite for Yume library.
"""
```

### 6. Install Package in Development Mode

```bash
# From project root
pip install -e .
```

This requires a `setup.py` file (see below).

### 7. Create setup.py (in project root)

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="yume",
    version="0.1.0",
    description="Latent Space Exploration Library",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "opencv-python>=4.8.0",
        "transformers>=4.30.0",
        "redis>=4.6.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-timeout>=2.1.0",
            "pytest-mock>=3.11.0",
        ],
    },
    python_requires=">=3.9",
)
```

## Running Tests

### From Project Root

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Install package in dev mode
pip install -e .

# Run tests
pytest tests/

# Or with the test runner
python run_tests.py all
```

### From tests/ Directory

```bash
cd tests/

# Run all tests
pytest

# Run specific test file
pytest test_dream_worker.py

# Run with coverage
pytest --cov=yume --cov-report=html
```

### Common Commands

```bash
# From project root
pytest tests/ -v                    # Verbose
pytest tests/ -k "test_init"        # Run specific test pattern
pytest tests/ -m "not slow"         # Skip slow tests
pytest tests/ --cov=yume            # With coverage

# Run test runner
python run_tests.py unit            # Unit tests only
python run_tests.py integration     # Integration tests
python run_tests.py coverage        # With coverage report
```

## Import Paths in Tests

With this structure, your tests should import like:

```python
# In test files
from yume.dream_worker import DreamWorker, DreamCandidate
from yume.scoring import CLIPScorer, AestheticScorer
from yume.backends.base import GenSpec, PipelineWorker
```

The `conftest.py` adds the parent directory to `sys.path` to make this work.

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'yume'`:

**Option 1: Install package in development mode (recommended)**
```bash
pip install -e .
```

**Option 2: Add to PYTHONPATH**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

**Option 3: Run from project root**
```bash
# From project root, not from tests/
pytest tests/
```

### Can't Find conftest.py

Make sure `conftest.py` is in the `tests/` directory. Pytest automatically discovers it.

### Tests Not Found

```bash
# Check pytest can discover tests
pytest --collect-only tests/

# Make sure test files start with test_
# Make sure test functions start with test_
```

### Redis/GPU Errors

Tests use mocks by default. If you see errors:

```bash
# Skip tests requiring real Redis
pytest -m "not requires_redis"

# Skip tests requiring GPU  
pytest -m "not requires_gpu"
```

## Alternative: Running Without Installation

If you don't want to install the package, update the test imports:

```python
# At top of conftest.py (already included)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Then import as:
from yume.dream_worker import DreamWorker
```

This is already set up in the provided `conftest.py`.

## CI/CD Setup

For GitHub Actions:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install -r requirements-test.txt
    
    - name: Run tests
      run: pytest tests/ --cov=yume --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Verification

After setup, verify everything works:

```bash
# 1. Check package is importable
python -c "from yume import DreamWorker; print('✓ Package imports work')"

# 2. Check pytest discovers tests
pytest --collect-only tests/

# 3. Run a quick test
pytest tests/test_dream_worker.py::TestDreamWorkerInitialization::test_init_default_config -v

# 4. Run all tests
pytest tests/
```

If all these pass, you're good to go! 🎉
