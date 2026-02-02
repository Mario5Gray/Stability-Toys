# Yume Tests - Quick Start

## Automated Setup (Easiest)

```bash
# 1. Run the setup script
python setup_tests.py

# 2. Install package in dev mode
pip install -e .

# 3. Install test dependencies
pip install -r requirements-test.txt

# 4. Run tests
pytest tests/
```

Done! 🎉

---

## Manual Setup (If you prefer)

### 1. Create Directory Structure

```
your-project/
├── yume/                    # Your source code here
│   ├── __init__.py
│   ├── dream_worker.py
│   ├── scoring.py
│   └── backends/
│       ├── __init__.py
│       └── base.py
│
├── tests/                   # Test files here
│   ├── __init__.py
│   ├── conftest.py
│   ├── pytest.ini
│   ├── test_dream_worker.py
│   ├── test_scoring.py
│   └── test_integration.py
│
├── setup.py                 # Package setup
├── requirements-test.txt    # Test dependencies
├── run_tests.py            # Test runner
└── README.md
```

### 2. Copy Files

```bash
# Create directories
mkdir -p yume tests

# Create __init__ files
touch yume/__init__.py
touch tests/__init__.py

# Move test files
mv test_*.py conftest.py pytest.ini tests/

# Move your source files to yume/
mv dream_worker.py scoring.py yume/
```

### 3. Install and Run

```bash
# Install in development mode
pip install -e .

# Install test dependencies
pip install -r requirements-test.txt

# Run tests
pytest tests/
```

---

## Running Tests

### From Project Root (Recommended)

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=yume --cov-report=html

# Using test runner
python run_tests.py all
python run_tests.py coverage
```

### From tests/ Directory

```bash
cd tests/

# Run all tests
pytest

# Run specific test file
pytest test_dream_worker.py -v

# With coverage
pytest --cov=yume
```

---

## Common Commands

```bash
# Unit tests only
pytest tests/ -m unit

# Integration tests only  
pytest tests/ -m integration

# Fast tests (skip slow)
pytest tests/ -m "not slow"

# Specific test
pytest tests/test_dream_worker.py::TestDreamWorkerInitialization -v

# With verbose output
pytest tests/ -v

# Stop on first failure
pytest tests/ -x
```

---

## Verification

Test that everything works:

```bash
# 1. Check imports
python -c "from yume import DreamWorker; print('✓ Imports work')"

# 2. Collect tests
pytest --collect-only tests/

# 3. Run one test
pytest tests/test_dream_worker.py::TestDreamWorkerInitialization::test_init_default_config -v

# 4. Run all tests
pytest tests/
```

---

## Troubleshooting

### Can't import yume

```bash
# Option 1: Install package (recommended)
pip install -e .

# Option 2: Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Option 3: Run from project root
pytest tests/  # Not from tests/ directory
```

### Tests not found

```bash
# Check pytest can find tests
pytest --collect-only tests/

# Make sure:
# - Test files start with test_
# - Test functions start with test_
# - conftest.py is in tests/
```

### Missing dependencies

```bash
pip install -r requirements-test.txt
```

---

## Files Included

- **test_dream_worker.py** - DreamWorker unit tests
- **test_scoring.py** - Scoring classes tests  
- **test_integration.py** - Integration tests
- **conftest.py** - Shared fixtures
- **pytest.ini** - Pytest configuration
- **requirements-test.txt** - Test dependencies
- **run_tests.py** - Test runner script
- **setup.py** - Package setup
- **setup_tests.py** - Automated setup script
- **SETUP_GUIDE.md** - Detailed setup guide
- **QUICK_REFERENCE.md** - Command reference

---

## Need Help?

See **SETUP_GUIDE.md** for detailed instructions and troubleshooting.
