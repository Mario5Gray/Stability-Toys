"""
M0 prerequisite (facet-3, STABL-rgvxuedo): confirm GenerateRequest survives the
spawn pickle boundary. Gates the Task 3 job wire-form — if raw pickle is clean the
envelope carries the instance; otherwise it carries model_dump() + model_validate().

NOTE: the plan's draft used steps/width/height; the real GenerateRequest
(server/lcm_sr_server.py:136) fields are num_inference_steps + size ("512x512").
"""
import multiprocessing as mp

from server.lcm_sr_server import GenerateRequest


def _echo(conn, payload):
    # Runs in a spawn child: receive, send back — proves cross-process transport.
    conn.send(payload)
    conn.close()


def test_generaterequest_round_trips_spawn_boundary():
    req = GenerateRequest(prompt="a cat", num_inference_steps=4, size="512x512")
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    p = ctx.Process(target=_echo, args=(child, req))
    p.start()
    got = parent.recv()
    p.join(timeout=10)
    assert isinstance(got, GenerateRequest)
    assert got.prompt == "a cat"
    assert got.num_inference_steps == 4
    assert got.size == "512x512"
