# Yume Tests - Quick Reference

## Setup

```bash
pip install -r requirements-test.txt
```

## Most Common Commands

```bash
# Run all tests
pytest

# Run with coverage
python run_tests.py coverage

# Run fast tests only
python run_tests.py fast

# Run specific test file
pytest test_dream_worker.py -v

# Run specific test
pytest test_dream_worker.py::TestDreamWorkerInitialization::test_init_default_config -v
```

## Test Categories

| Command | Description |
|---------|-------------|
| `python run_tests.py unit` | Unit tests only |
| `python run_tests.py integration` | Integration tests only |
| `python run_tests.py fast` | Fast tests (exclude slow) |
| `python run_tests.py slow` | Slow tests only |

## Markers

```bash
pytest -m unit              # Unit tests
pytest -m integration       # Integration tests  
pytest -m "not slow"        # Exclude slow tests
pytest -m requires_gpu      # Tests requiring GPU
pytest -m requires_redis    # Tests requiring Redis
```

## Coverage

```bash
# Generate HTML coverage report
python run_tests.py coverage

# View report
open htmlcov/index.html
```

## Debugging

```bash
# Verbose output
pytest -v

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Re-run only failed tests
pytest --lf
```

## Parallel Execution

```bash
# Run with 4 workers
python run_tests.py parallel -w 4
```

## Watch Mode

```bash
# Re-run on file changes (requires pytest-watch)
python run_tests.py watch
```

## Common Issues

### Import Errors
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Async Issues
```bash
pip install pytest-asyncio
```

### Slow Tests
```bash
pytest -m "not slow"  # Skip slow tests
pytest -n 4           # Run in parallel
```

## Test File Structure

- `test_dream_worker.py` - DreamWorker unit tests (200+ tests)
- `test_scoring.py` - Scoring classes tests (CLIPScorer, AestheticScorer)
- `test_integration.py` - End-to-end integration tests
- `conftest.py` - Shared fixtures and configuration
- `pytest.ini` - Pytest configuration

## Writing Tests

```python
import pytest

class TestMyFeature:
    def test_something(self, mock_redis_client):
        """Test description."""
        # Arrange
        worker = DreamWorker(...)
        
        # Act  
        result = worker.method()
        
        # Assert
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_async(self):
        """Test async code."""
        result = await async_func()
        assert result
```

## Fixtures Available

- `test_image_64`, `test_image_224`, `test_image_512` - Test images
- `mock_redis_client` - Mock Redis client
- `mock_pipeline_worker` - Mock LCM worker
- `mock_clip_scorer` - Mock CLIP scorer
- `sample_prompts`, `sample_seeds` - Test data

## CI/CD Integration

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: python run_tests.py coverage
  
- name: Upload coverage
  uses: codecov/codecov-action@v2
```
