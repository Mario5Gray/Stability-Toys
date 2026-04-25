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
    first = cache.acquire(
        "sdxl-canny", "/models/sdxl-canny", loader=lambda path: {"path": path}
    )
    second = cache.acquire(
        "sdxl-depth", "/models/sdxl-depth", loader=lambda path: {"path": path}
    )

    entries = cache.snapshot()["entries"]
    assert "sdxl-canny" in entries
    assert "sdxl-depth" in entries
    assert first == {"path": "/models/sdxl-canny"}
    assert second == {"path": "/models/sdxl-depth"}
