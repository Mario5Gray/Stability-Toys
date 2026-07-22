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


def test_controlnet_cache_clear_drops_unpinned_entries():
    # unload/free-vram must actually release ControlNet models. release() only
    # decrements pin_count; with <= max_entries loaded nothing ever evicts, so
    # the models stay resident in VRAM. clear() is what frees them.
    from backends.controlnet_cache import ControlNetModelCache

    cache = ControlNetModelCache(max_entries=4)
    cache.acquire("hunyuandit-canny", "/m/canny", loader=lambda p: {"p": p})
    cache.release("hunyuandit-canny")
    cache.acquire("sd15-depth", "/m/depth", loader=lambda p: {"p": p})
    cache.release("sd15-depth")

    assert cache.snapshot()["entries"] == ["hunyuandit-canny", "sd15-depth"]

    dropped = cache.clear()

    assert dropped == 2
    assert cache.snapshot()["entries"] == []


def test_controlnet_cache_clear_keeps_pinned_entries():
    # A pinned entry means a job is mid-flight using it; do not yank it.
    from backends.controlnet_cache import ControlNetModelCache

    cache = ControlNetModelCache(max_entries=4)
    cache.acquire("in-use", "/m/in-use", loader=lambda p: {"p": p})  # pinned
    cache.acquire("idle", "/m/idle", loader=lambda p: {"p": p})
    cache.release("idle")

    dropped = cache.clear()

    assert dropped == 1
    assert cache.snapshot()["entries"] == ["in-use"]
