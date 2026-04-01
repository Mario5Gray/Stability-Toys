from pathlib import Path

import pytest


def test_run_once_writes_output_and_prints_summary(tmp_path, monkeypatch, capsys):
    from server.superres_cli import run_once

    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.jpeg"
    input_path.write_bytes(b"input-bytes")

    events = []

    class FakeService:
        model_path = "/models/sr/RealESRGAN_x4plus.pth"

        def shutdown(self):
            events.append(("shutdown",))

    def fake_initialize_superres_service(**kwargs):
        events.append(("init", kwargs))
        return FakeService()

    def fake_submit_superres(**kwargs):
        events.append(("submit", kwargs))
        return b"upscaled-bytes"

    monkeypatch.setattr("server.superres_cli.initialize_superres_service", fake_initialize_superres_service)
    monkeypatch.setattr("server.superres_cli.submit_superres", fake_submit_superres)

    rc = run_once(
        input_path=input_path,
        output_path=output_path,
        magnitude=2,
        out_format="jpeg",
        quality=88,
        environ={
            "BACKEND": "cuda",
            "CUDA_SR_MODEL": "/models/sr/RealESRGAN_x4plus.pth",
        },
    )

    assert rc == 0
    assert output_path.read_bytes() == b"upscaled-bytes"
    assert events[0][0] == "init"
    assert events[1][0] == "submit"
    assert events[2] == ("shutdown",)

    out = capsys.readouterr().out
    assert "backend=cuda" in out
    assert "model=RealESRGAN_x4plus.pth" in out
    assert f"output={output_path}" in out


def test_main_rejects_missing_input(tmp_path, capsys):
    from server.superres_cli import main

    missing = tmp_path / "missing.png"
    output_path = tmp_path / "output.png"

    rc = main(
        [
            "--input",
            str(missing),
            "--output",
            str(output_path),
        ]
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "input file not found" in err


def test_run_once_shuts_down_service_on_failure(tmp_path, monkeypatch, capsys):
    from server.superres_cli import run_once

    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    input_path.write_bytes(b"input-bytes")

    events = []

    class FakeService:
        model_path = "/models/sr/RealESRGAN_x4plus.pth"

        def shutdown(self):
            events.append(("shutdown",))

    def fake_initialize_superres_service(**kwargs):
        return FakeService()

    def fake_submit_superres(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("server.superres_cli.initialize_superres_service", fake_initialize_superres_service)
    monkeypatch.setattr("server.superres_cli.submit_superres", fake_submit_superres)

    rc = run_once(
        input_path=input_path,
        output_path=output_path,
        magnitude=1,
        out_format="png",
        quality=92,
        environ={
            "BACKEND": "cuda",
            "CUDA_SR_MODEL": "/models/sr/RealESRGAN_x4plus.pth",
        },
    )

    assert rc == 1
    assert events == [("shutdown",)]
    err = capsys.readouterr().err
    assert "superres failed: boom" in err
