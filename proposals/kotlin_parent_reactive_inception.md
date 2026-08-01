Genuinely viable, and better-timed than it would have been a week ago — the seam is now documented and tested rather than tribal. Let me be concrete about what lines up and what doesn't.

What Kotlin actually buys
All three libraries we just surveyed converge on the JVM. RSocket's reference implementation is rsocket-java; Aeron is JVM-native; Project Reactor gives you Flux/Mono with real operators and real backpressure. That collapses gaps #1, #2 and #3 from my readiness assessment into "use the mature library" instead of "hand-roll a Processor and a demand protocol." The asyncio-vs-blocking-CUDA marriage problem disappears too — it moves to the Python worker's edge, where the worker is a responder and can stay synchronous.

The Governor is a concurrency state machine, which is Kotlin's home turf. Job queue, epochs, admission, dispatch, cancellation, two-budget timeouts. Structured concurrency and typed channels replace _job_lock, the dispatch-thread freeze dance in tests, and concurrent.futures bridging.

Types would have caught this session's bugs. spxwqlan was a five-field record encoded as three — an exhaustive constructor makes that a compile error, and CARRIED_JOB_FIELDS/NOT_CARRIED_JOB_FIELDS stops being necessary. Sealed frame hierarchies do the same for backplane frames. That's not a small argument given how many silent-default bugs we found today.

The crux: where does model resolution live?
This is the decision the proposal forces, and it's bigger than the language choice.

Today the parent is the single resolution authority — deliberately. Facet-3's design says the child rebuilds from a codec and never re-resolves, precisely so a parent-side resolve_model patch still governs and the child touches no filesystem. But resolution is utils/model_detector introspecting safetensors and diffusers configs, family profiles, variant detection. That is deeply Python.

So a Kotlin Governor means either:

Resolution moves to the Python worker — inverting the authority placement we just settled in ltefhpkk/iuiwzthc. The epoch model assumes the parent knows the resolved model before the load. If the worker resolves, the parent is stamping epochs against something it hasn't seen.
Kotlin reimplements detection — duplicating diffusers/safetensors introspection in a second language, and permanently keeping two detectors agreeing.
Neither is fatal, but this — not RSocket, not the transport — is the load-bearing design question. My read: the Governor genuinely splits into scheduling/authority (language-agnostic, good Kotlin fit) and resolution (irreducibly Python). Today they're fused. Forcing that split is probably good architecture independent of Kotlin, and it's what I'd want designed before any rewrite.

InfiniBand doesn't earn its place yet
We move ~1MB PNGs and small job structs. On one box, shared-memory IPC is already effectively free. RDMA earns its keep for large tensors across hosts — a multi-host cluster with model-parallel workers. Today it's the least justified piece of the proposal, and I'd drop it from the design until the workload is multi-host. Aeron's IPC mode gives you the same API if you later move to UDP/RDMA, so nothing is lost by deferring.

What carries and what gets rebuilt
Carries: every design decision, because they're written down — authority reservation, the two-budget timeout with executing_since stamped after reload and barrier, dead-epoch pruning, the completeness contract, the cancel/queue interaction. Those specs are language-independent and they're the expensive part.

Rebuilt: the implementations, and each one is a chance to reintroduce a bug we already paid for. The mitigation is that most of them now have tests describing the failure, and those port as test cases even if the code doesn't.

What I'd do before committing
Not a debate — a spike with a falsifiable question:

Can a Kotlin RSocket requester drive a Python rsocket-py responder that runs a blocking CUDA generate, with request(n) and cancel working end to end, and a result blob crossing without a copy that hurts?

That's roughly a day, it exercises the exact edge that would kill the direction, and it produces a real artifact rather than an opinion. If it works, the remaining risk is scope; if it doesn't, you've saved a rewrite.

And I'd sequence it after steps 1–2 of the reactive work (Processor, WS as a real Subscriber, Progress emission) — those are useful under either outcome, they're what the client actually sees, and doing them first tells you how much of the reactive model you need before betting a rewrite on it.

One thing worth saying plainly: none of this makes generation faster on one GPU. It makes the control plane sound, the pipeline observable, and the system honest about backpressure. Those are good reasons — just not throughput reasons.

Want me to write this up as a brainstorm spec with the resolution-authority split as the central decision? That's the piece I'd want settled on paper before anyone writes Kotlin.