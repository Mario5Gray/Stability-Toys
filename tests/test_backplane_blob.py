import asyncio
import pytest
from backends.backplane.blob import InProcBlob, SharedMemBlob, encode_frame, decode_frame, SCHEMA_VERSION
from backends.backplane.frames import Ack, Progress, Result, BackplaneError, BackplaneErrorCode


def _read(blob):
    return asyncio.get_event_loop().run_until_complete(blob.read())


def test_inproc_blob_read_returns_bytes_close_noop():
    b = InProcBlob(b"hello")
    assert _read(b) == b"hello"
    assert b.read_sync() == b"hello"  # sync path used by the loop-less facade
    b.close()  # no raise
    assert b.read_sync() == b"hello"  # close() is a true no-op (spec §4.4)


def test_decode_frame_rejects_empty_input():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        decode_frame(b"")


def test_sharedmem_blob_roundtrip_then_unlink():
    src = SharedMemBlob.create(b"PNGDATA")
    assert _read(src) == b"PNGDATA"
    src.close()  # close + unlink
    # After unlink, re-attaching by name must fail.
    from multiprocessing import shared_memory
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=src.name)


def test_codec_roundtrips_ack_and_progress_with_schema_version():
    for frame in (Ack("j1", 2), Progress("j1", 5, 20, "decode")):
        raw = encode_frame(frame)
        assert raw[0] == SCHEMA_VERSION  # version is the first byte
        assert decode_frame(raw) == frame


def test_codec_roundtrips_result_with_sharedmem_blob():
    blob = SharedMemBlob.create(b"xy")
    out = decode_frame(encode_frame(Result("j1", 42, blob)))
    assert out.seed == 42
    assert isinstance(out.image, SharedMemBlob)
    assert out.image.name == blob.name
    blob.close()


def test_codec_roundtrips_error_terminal_code_only():
    err = BackplaneError(BackplaneErrorCode.OOM, "CUDA out of memory")
    out = decode_frame(encode_frame(err))
    assert out.code is BackplaneErrorCode.OOM
    assert out.message == "CUDA out of memory"
    assert out.original is None  # live instance does not cross the wire
