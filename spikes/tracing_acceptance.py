"""Live acceptance for STABL-qnlaclof: does a real generation reach Tempo?

Run INSIDE the deployed container on enigma, then read back from node1. The unit
tests prove the span graph is built correctly in-process; this proves the built
graph actually leaves the box.

    docker exec stability-toys-dev python /app/spikes/tracing_acceptance.py

No PYTHONPATH needed — the script puts the repo root on sys.path itself.

WHAT IT CHECKS, and why it is not just "a trace exists":

  1. the trace is retrievable from Tempo by id
  2. it contains BOTH sides of the process boundary — worker.submit and
     worker.execute — with the child's parent being the boundary span
  3. the kinds are PRODUCER and CONSUMER

(2) and (3) are the acceptance. Review of PR #70 found the boundary span missing
entirely and the child mislabelled INTERNAL, and a trace-id check passes in that
state — so an acceptance that only looks for "some spans" would have signed off
on a trace with the one interesting edge absent.

Reads TEMPO_QUERY (default http://node1.lan:3200).
"""
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

# Run from anywhere, including `docker exec ... python /app/spikes/this.py`.
# Python puts the SCRIPT's directory on sys.path, not the working directory, so
# that invocation gets /app/spikes and `import server` fails — which reads as
# "the container is broken" rather than "the path is wrong". Every other spike
# has this too and has been worked around by hand with PYTHONPATH each time.
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

TEMPO = os.environ.get("TEMPO_QUERY", "http://node1.lan:3200").rstrip("/")


def _fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    if os.environ.get("TRACING_ENABLED", "").strip().lower() in ("", "0", "false", "off", "no"):
        _fail("TRACING_ENABLED is not set in this container; nothing would be exported")

    from server import tracing

    t = tracing.get_tracing()
    if not t.enabled:
        _fail(
            "the facade is disabled despite TRACING_ENABLED — check "
            "OTEL_EXPORTER_OTLP_ENDPOINT is set and the SDK is installed"
        )
    print(f"exporter endpoint : {tracing._traces_endpoint()}")

    # Build the boundary pair exactly as the two processes do, so a failure here
    # is a TRANSPORT failure rather than an instrumentation one.
    from opentelemetry.propagate import inject

    tracer = tracing.get_tracer("spikes.tracing_acceptance")
    with tracer.start_as_current_span(
            "worker.submit", **tracing.kind_kwargs("PRODUCER")) as producer:
        carrier: dict = {}
        inject(carrier)
        ctx = producer.get_span_context()
        trace_id = format(ctx.trace_id, "032x")
        producer_span_id = format(ctx.span_id, "016x")

    parent = tracing.context_from_carrier(carrier)
    with tracer.start_as_current_span(
            "worker.execute", context=parent, **tracing.kind_kwargs("CONSUMER")):
        pass

    # BatchSpanProcessor is asynchronous; without this the read-back races the
    # export and reports a missing trace that is merely late.
    t._provider.force_flush(timeout_millis=10_000)
    print(f"trace id          : {trace_id}")

    url = f"{TEMPO}/api/traces/{trace_id}"
    payload = None
    for attempt in range(10):
        time.sleep(2)
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                if resp.status == 200:
                    body = resp.read()
                    if body:
                        payload = json.loads(body)
                        break
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                _fail(f"tempo returned HTTP {exc.code} for {url}")
        except Exception as exc:                     # noqa: BLE001
            _fail(f"cannot reach tempo at {url}: {exc}")
        print(f"  not indexed yet (attempt {attempt + 1}/10)")

    if payload is None:
        _fail(
            f"trace {trace_id} never appeared in tempo. The collector accepted it "
            f"or not — check the collector's own logs; a 200 from the collector is "
            f"NOT proof of delivery to tempo."
        )

    spans = {}
    for batch in payload.get("batches", []):
        for scope in batch.get("scopeSpans", []):
            for span in scope.get("spans", []):
                spans[span["name"]] = span

    print(f"spans in trace    : {sorted(spans)}")
    for name in ("worker.submit", "worker.execute"):
        if name not in spans:
            _fail(f"{name} is missing — the boundary is not represented")

    submit, execute = spans["worker.submit"], spans["worker.execute"]
    if submit.get("kind") != "SPAN_KIND_PRODUCER":
        _fail(f"worker.submit kind is {submit.get('kind')}, not PRODUCER")
    if execute.get("kind") != "SPAN_KIND_CONSUMER":
        _fail(f"worker.execute kind is {execute.get('kind')}, not CONSUMER")

    # Tempo returns ids base64-encoded; compare on the decoded hex.
    import base64

    child_parent = base64.b64decode(execute.get("parentSpanId", "")).hex()
    if child_parent != producer_span_id:
        _fail(
            f"worker.execute's parent is {child_parent or '(none)'}, not the "
            f"boundary span {producer_span_id} — a shared trace id would have "
            f"hidden this"
        )

    print()
    print("PASS: one trace, boundary edge intact, PRODUCER -> CONSUMER")
    print(f"  {TEMPO}/api/traces/{trace_id}")


if __name__ == "__main__":
    main()
