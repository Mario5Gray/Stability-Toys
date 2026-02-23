# Journal: Model Lifecycle Tests - Feb 2026

Just wrote a full lifecycle test suite for the worker pool. 18 tests covering the load/unload matrix from TODO.md. The codebase has a nice DI pattern in WorkerPool - mock the factory, config, and registry, and everything slots together cleanly.

One gotcha: `_unload_current_worker()` does `import gc` locally, so patching `backends.worker_pool.gc.collect` doesn't work - had to patch the actual `gc` module's `collect` method. Small thing but the kind of detail that eats 10 minutes if you don't read the source first.

The empty_pool fixture was fun - patching `_load_mode` during `__init__` with a context manager so the pool starts with no worker, then the real method is available for subsequent calls. Clean pattern.

The existing test suite has ~28 pre-existing failures from missing paths and modules (`/app/logs/torch.log`, `backends.cuda_worker`). Not my circus. The new tests are isolated and don't poison anything.

Feeling good about the coverage. The full matrix test at the end is satisfying - walks through all 6 steps in one flow. Like watching a little state machine dance.
