def test_controlnet_cache_reuses_loaded_model_for_same_model_id():
    from backends.controlnet_cache import ControlNetModelCache

    calls = []

    def loader(path: str):
        calls.append(path)
        return {"path": path}

    cache = ControlNetModelCache(max_entries=2)
    first = cache.acquire("sdxl-canny", "/models/sdxl-canny", loader=loader)
    cache.release("sdxl-canny")
    second = cache.acquire("sdxl-canny", "/models/sdxl-canny", loader=loader)

    assert first is second
    assert calls == ["/models/sdxl-canny"]


def test_controlnet_cache_does_not_evict_pinned_entries():
    from backends.controlnet_cache import ControlNetModelCache

    cache = ControlNetModelCache(max_entries=1)
    cache.acquire("sdxl-canny", "/models/sdxl-canny", loader=lambda path: {"path": path})
    cache.acquire("sdxl-depth", "/models/sdxl-depth", loader=lambda path: {"path": path})
    assert "sdxl-canny" in cache.snapshot()["entries"]
