from PIL import Image


def test_cuda_test_import_does_not_break_pillow_png_plugin():
    import tests.test_cuda_worker_base  # noqa: F401

    Image.init()

    assert "PNG" in Image.SAVE

    import PIL.PngImagePlugin as png_plugin

    assert getattr(png_plugin, "__file__", "").endswith("PngImagePlugin.py")
