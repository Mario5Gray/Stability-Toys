# Yume Library Test Suite

Comprehensive test suite for the Yume latent space exploration library.

## Overview

The test suite covers:

- **Unit Tests**: Individual components (DreamWorker, CLIPScorer, AestheticScorer)
- **Integration Tests**: End-to-end workflows and component interactions
- **Mocking**: Isolated testing without external dependencies (GPU, Redis, models)
- **Coverage**: Code coverage reporting and analysis

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── pytest.ini               # Pytest configuration
├── requirements-test.txt    # Test dependencies
├── run_tests.py            # Test runner script
│
├── test_dream_worker.py    # DreamWorker unit tests
├── test_scoring.py         # Scoring classes unit tests
└── test_integration.py     # Integration tests
```

## Quick Start

### Install Test Dependencies

```bash
pip install -r requirements-test.txt
```

### Run All Tests

```bash
# Using pytest directly
pytest

# Or using the test runner
python run_tests.py all
```

### Run Specific Test Suites

```bash
# Unit tests only
python run_tests.py unit

# Integration tests only
python run_tests.py integration

# Fast tests (exclude slow tests)
python run_tests.py fast
```

## Test Categories

### Unit Tests (`test_dream_worker.py`)

Tests for the DreamWorker class:

- ✅ Initialization with different configurations
- ✅ Dream session management (start, stop, status)
- ✅ Candidate generation and scoring
- ✅ Prompt variation generation
- ✅ Exploration strategies (random, linear walk, grid)
- ✅ Latent hashing for deduplication
- ✅ Redis storage operations
- ✅ FPS tracking
- ✅ Error handling

```bash
pytest test_dream_worker.py -v
```

### Scoring Tests (`test_scoring.py`)

Tests for scoring classes:

- ✅ CLIPScorer with Hugging Face implementation
- ✅ CLIPScorer with OpenAI CLIP implementation
- ✅ Automatic CLIP type detection
- ✅ Text embedding caching
- ✅ Batch scoring
- ✅ AestheticScorer heuristic scoring
- ✅ CompositeScorer with weighted scores
- ✅ Edge cases and error handling

```bash
pytest test_scoring.py -v
```

### Integration Tests (`test_integration.py`)

End-to-end workflow tests:

- ✅ Complete dream session workflow
- ✅ Candidate pipeline (generate → score → store)
- ✅ Top candidate rendering
- ✅ Concurrent operations
- ✅ Dream persistence and retrieval
- ✅ Different exploration strategies
- ✅ Prompt variation strategies
- ✅ Error recovery
- ✅ Performance metrics

```bash
pytest test_integration.py -v
```

## Test Runner Commands

The `run_tests.py` script provides convenient commands:

### Basic Commands

```bash
# Run all tests
python run_tests.py all

# Run with coverage report
python run_tests.py coverage

# Run only fast tests
python run_tests.py fast

# Run only slow tests
python run_tests.py slow
```

### Targeted Testing

```bash
# Run specific test file
python run_tests.py specific test_dream_worker.py

# Run specific test class
python run_tests.py specific test_dream_worker.py::TestDreamWorkerInitialization

# Run specific test method
python run_tests.py specific test_dream_worker.py::TestDreamWorkerInitialization::test_init_default_config
```

### Advanced Commands

```bash
# Run tests in parallel (4 workers)
python run_tests.py parallel -w 4

# Run only previously failed tests
python run_tests.py failed

# Watch mode (re-run on file changes)
python run_tests.py watch

# Clean cache files
python run_tests.py clean
```

### Verbose Output

Add `-v` flag for detailed output:

```bash
python run_tests.py all -v
python run_tests.py unit -v
```

## Test Markers

Tests are organized with pytest markers:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Exclude slow tests
pytest -m "not slow"

# Run tests requiring GPU
pytest -m requires_gpu

# Run tests requiring Redis
pytest -m requires_redis
```

## Coverage Reports

### Generate Coverage Report

```bash
python run_tests.py coverage
```

This generates:
- Terminal report with missing lines
- HTML report in `htmlcov/` directory
- JSON report for CI integration

### View HTML Coverage Report

```bash
# After running coverage
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Coverage Goals

Target coverage levels:
- **Overall**: >85%
- **Core modules**: >90%
- **Critical paths**: 100%

## Fixtures

### Shared Fixtures (conftest.py)

Available in all tests:

**Images:**
- `test_image_64` - 64x64 RGB image
- `test_image_224` - 224x224 RGB image (CLIP size)
- `test_image_512` - 512x512 RGB image
- `test_image_bytes` - PNG image as bytes

**Latents:**
- `random_latent_tensor` - PyTorch tensor (1, 4, 8, 8)
- `random_latent_numpy` - NumPy array (1, 4, 8, 8)

**Mocks:**
- `mock_redis_client` - Async Redis client with storage
- `mock_pipeline_worker` - LCM pipeline worker
- `mock_clip_model_hf` - Hugging Face CLIP model
- `mock_clip_processor` - CLIP processor
- `mock_clip_scorer` - Complete CLIP scorer

**Data:**
- `sample_prompts` - List of test prompts
- `sample_seeds` - List of test seeds

### Using Fixtures

```python
def test_my_feature(mock_redis_client, test_image_224):
    # Fixtures are automatically injected
    # Use them directly
    pass
```

## Custom Assertions

Helper assertions for common checks:

```python
# Check score validity
pytest.assert_valid_score(0.75)

# Check image validity
pytest.assert_valid_image(image)

# Check candidate validity
pytest.assert_valid_candidate(candidate)
```

## Writing New Tests

### Test Template

```python
"""
Test module for [component].
"""

import pytest


class TestMyFeature:
    """Test [feature description]."""
    
    def test_basic_functionality(self, mock_redis_client):
        """Test basic functionality."""
        # Arrange
        worker = DreamWorker(...)
        
        # Act
        result = worker.some_method()
        
        # Assert
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_async_feature(self):
        """Test async feature."""
        result = await async_function()
        assert result
    
    @pytest.mark.slow
    def test_expensive_operation(self):
        """Test that takes a long time."""
        # Mark as slow
        pass
```

### Best Practices

1. **Use descriptive names**: `test_candidate_generation_with_random_strategy`
2. **One assertion per test**: Keep tests focused
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Use fixtures**: Don't repeat setup code
5. **Mock external deps**: Isolate unit tests
6. **Test edge cases**: Empty inputs, errors, boundaries
7. **Add docstrings**: Explain what's being tested

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements-test.txt
    
    - name: Run tests with coverage
      run: |
        python run_tests.py coverage
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Troubleshooting

### Tests Failing

**Import errors:**
```bash
# Add source to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**Async tests not running:**
```bash
# Install pytest-asyncio
pip install pytest-asyncio
```

**Redis errors:**
```bash
# Tests use mocks by default
# To test with real Redis:
pytest -m requires_redis
```

### Slow Tests

```bash
# Skip slow tests
pytest -m "not slow"

# Or run in parallel
pytest -n 4
```

### Coverage Not Working

```bash
# Reinstall pytest-cov
pip install --upgrade pytest-cov

# Check coverage config in pytest.ini
```

## Performance Testing

Run performance benchmarks:

```bash
# Install benchmark plugin
pip install pytest-benchmark

# Run with benchmarks
pytest --benchmark-only
```

## Contributing Tests

When adding new features:

1. Write tests first (TDD)
2. Ensure >85% coverage
3. Add integration tests for workflows
4. Update this README if needed
5. Run full test suite before PR

```bash
# Before committing
python run_tests.py all -v
python run_tests.py coverage
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)

## License

Same as Yume library.
