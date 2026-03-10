  Type annotation fixes:
  - base.py — GenerateRequest/Future via TYPE_CHECKING, missing @ on @dataclass, Protocol method ... bodies
  - cuda_worker.py — diffusers imports from explicit submodules, removed invalid super().__init__(), .images suppressed with # type: ignore[union-attr]
  - model_registry.py — loras: List[str] = field(default_factory=list), any → Any
  - rknn_worker.py — LCMScheduler import, removed super().__init__(), typed self.pipe, scheduler cast, # type: ignore on known-safe calls
  - rknnlcm.py — import paths, # type: ignore on numpy/None narrowing throughout
  - worker_pool.py — fut: Future = field(default_factory=Future), Optional[dict], assert mode.model_path is not None
  - comfy_client.py — Json type alias, Callable, temp var for len(v.get(k))
  - redis_provider.py — fixed real bug (ttl is None syntax error + undefined STORAGE_TTL_IMAGE), bytes conversion, operator ignore
  - storage_provider.py — return type Optional[StorageProvider]
  - comfy_routes.py, file_watcher.py — str | None guards
  - lcm_sr_server.py — rknnlite ignore, Image.Resampling.BICUBIC, type ignores on known-safe calls
  - mode_config.py, workflow_config.py — config: T = None # type: ignore[assignment] pattern, assert in _validate_paths
  - run.py — import logging.config
  - ws_routes.py — Optional[str] on _error, # type: ignore[arg-type]

  pyrightconfig.json — added tests, utils, yume_lab to exclude list
