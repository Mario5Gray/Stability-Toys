# Plan: Language Server MCP Integration

Adds a semantic layer to the agent toolchain that can resolve types, follow
imports across packages, produce diagnostics, and perform safe renames — things
treesitter cannot do because it only understands syntax, not meaning.

Three phases. Each phase is independently useful and can stop there if the
cost/benefit doesn't justify continuing.

---

## Background: why this complements treesitter

| Question | treesitter | LSP |
|---|---|---|
| Where is `switch_mode` defined? | ✓ (name match) | ✓ (resolved) |
| All callers of *this* `switch_mode` on `WorkerPool` specifically | ✗ finds all names | ✓ type-resolved |
| What type does `fut.result()` return here? | ✗ | ✓ |
| Are there type errors in the demand-reload path? | ✗ | ✓ diagnostics |
| Find the string literal `"job:complete"` | ✗ | ✗ (use Grep) |
| Rename `_free_worker` safely across the project | ✗ | ✓ scope-aware |
| Works on a file with a syntax error | ✓ | ✗ degrades |
| Works without project indexing | ✓ | ✗ needs index |

The two tools run in parallel. treesitter for structure and speed; LSP for
semantic precision when it matters.

---

## Phase 1 — Language Server Selection, Tools, and Environments ✅ COMPLETE

### 1.1 Python backend — Pyright

**Selected:** `pyright` (Microsoft, npm-distributed)
**Status: done — `pyright .` reports 0 errors, 0 warnings.**

**Why over the alternatives:**
- `pylsp` — community-maintained, modular but slower; Jedi-based go-to-def
  works well for pure Python but lags on large dependency trees (torch,
  diffusers).
- `jedi-language-server` — thin Jedi wrapper; good for completions, weaker on
  type inference.
- `pyright` — built on TypeScript, statically typed itself, fastest indexer of
  the three. Understands Python 3.10+ type syntax (`X | Y`, `ParamSpec`,
  dataclasses with `field()`). Handles `concurrent.futures.Future[T]` generics
  correctly, which matters for `WorkerPool`.

**What it resolves in this codebase:**
- `fut: Future` → `Future[tuple[bytes, int]]` through the generation job chain
- `job.execute(self._worker)` → dispatched to the correct concrete subclass
- `self._mode_config.get_mode(name)` → `ModeConfig` with all its fields
- Diagnostics: missing type annotations, unreachable code, wrong argument types

**Caveats (known, accepted):**
- `torch` and `diffusers` ship partial stubs. Pyright shows `Unknown` for
  some tensor operations. Suppressed with targeted `# type: ignore[union-attr]`
  at `.images` access points; the value is in server-side business logic.
- `rknnlite` has no stubs — suppressed with `# type: ignore[import-untyped]`
  at each import site.
- venv resolution: pyright does **not** use a `venvPath`/`venv` key in
  `pyrightconfig.json` for this project. Python is resolved at runtime by
  passing `--pythonpath "$(which python)"` on the CLI (or through the IDE's
  Python path setting). This avoids hardcoding a venv path in the config.

**Installation:**
```bash
npm install -g pyright
# or: pip install pyright (self-contained binary)
```
Installed version: **pyright 1.1.408**

**Config file:** `pyrightconfig.json` at repo root (current actual state):
```json
{
  "include": ["backends", "server", "invokers", "persistence"],
  "exclude": [
    "lcm-sr-ui", "node_modules", "**/__pycache__", ".git", ".mypy_cache",
    "tests", "utils", "yume_lab"
  ],
  "pythonVersion": "3.12",
  "pythonPlatform": "Linux",
  "typeCheckingMode": "basic",
  "reportMissingImports": true,
  "reportMissingTypeStubs": false,
  "executionEnvironments": [{ "root": "." }]
}
```

`typeCheckingMode: "basic"` (not `strict`). The codebase has had type
annotations added incrementally during the pyright clean-up pass (2026-03).
Many third-party boundaries (diffusers pipelines, rknnlite, redis-py) are
suppressed with targeted `# type: ignore` comments rather than full
annotations, keeping signal-to-noise high. Upgrade to `standard` when the core
server modules have complete annotations.

`tests/`, `utils/`, and `yume_lab/` are excluded — they are not production
code and `pytest` / hardware-specific packages are not in the runtime pyright
checks against.

---

### 1.2 JavaScript/React frontend — typescript-language-server ✅ COMPLETE

**Selected:** `typescript-language-server` (the official tsserver-based LSP)
**Status: done — `lcm-sr-ui/jsconfig.json` in place and accepted by tsserver.**

**Why:**
- The frontend is `.jsx` (JavaScript, not TypeScript), but `typescript` is
  already in `devDependencies`. `typescript-language-server` supports `.js`/
  `.jsx` files with a `jsconfig.json` — same engine, no migration needed.
- Understands React JSX natively via `tsconfig`/`jsconfig` `jsx` setting.
- Go-to-definition follows imports into `node_modules` with type declarations.

**What it resolves in this codebase:**
- `wsClient.send(msg)` → `WSClient.send(msg: object): string`
- `jobQueue.enqueue({...})` → parameter shape validation
- `params.setDenoiseStrength(v)` → which overload is actually called
- `message.params` → inferred shape from usage

**Config file:** `lcm-sr-ui/jsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "checkJs": false,
    "allowJs": true,
    "noEmit": true,
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist"]
}
```
`checkJs: false` initially — enables navigation and go-to-def without
surfacing type errors in untyped JS. Can be enabled file-by-file with
`// @ts-check` when annotations are ready.

**Installation:**
```bash
npm install -g typescript-language-server typescript
```

---

### 1.3 Environment summary

| Component | Runtime | Install | Config |
|---|---|---|---|
| Pyright LSP v1.1.408 | Node 25.x (present) | `npm install -g pyright` | `pyrightconfig.json` |
| typescript-language-server | Node 25.x (present) | `npm install -g typescript-language-server typescript` | `lcm-sr-ui/jsconfig.json` |
| Python runtime | Python 3.12.10 | system/conda (base env) | `--pythonpath "$(which python)"` CLI flag |

Both language servers communicate over **stdio** — they read/write JSON-RPC on
stdin/stdout. The MCP bridge (Phase 3) manages these processes.

---

## Phase 2 — Deployment Planning

### 2.1 Decision: host-local vs Docker

**Recommendation: host-local for now, Docker-ready by design.**

The language servers are development tools that need:
1. Read access to the entire workspace filesystem
2. Read access to the Python virtual environment (`site-packages`)
3. Read access to `node_modules`
4. Persistent index state between sessions (avoid re-indexing every invocation)

**Why not Docker (yet):**
- Volume-mounting `node_modules` and a Python venv into a container adds
  friction with no isolation benefit for a local dev tool.
- Language server indexing state (`~/.cache/pyright`, tsserver's project
  service cache) naturally lives on the host. Container restarts lose it
  unless an additional cache volume is managed.
- The treesitter MCP is already running host-local; consistency matters.

**When to move to Docker:**
- If the language server needs to run on a different machine (remote dev
  environment, shared team tooling server).
- If isolation is needed (e.g. the LSP daemon has a history of consuming
  unbounded memory and needs cgroup limits).

**Docker-ready design:** Structure the MCP bridge (Phase 3) so its workspace
root and LSP binary paths are environment variables. Switching to Docker then
means adding a `docker-compose.yml` with volume mounts and changing those vars,
not rewriting the bridge.

---

### 2.2 Process model

Each language server runs as a **managed child process** owned by the MCP
bridge. The bridge:
- Spawns the language server on first use (lazy start)
- Keeps it alive across tool calls (avoids re-indexing penalty on every call)
- Restarts it if it crashes
- Shuts it down cleanly when the bridge exits

Two separate bridge instances (or one bridge managing two servers):
- `pyright-langserver --stdio` for Python
- `typescript-language-server --stdio` for JS/JSX

**Workspace initialisation sequence:**
1. Bridge spawns language server
2. Sends LSP `initialize` request with workspace root
3. Sends `initialized` notification
4. Language server begins background indexing (async, non-blocking)
5. First tool call may return partial results while indexing completes;
   subsequent calls are fast

Indexing time estimates (cold start, this codebase):
- Pyright on the Python backend: ~2–5 seconds
- typescript-language-server on `lcm-sr-ui/src`: ~3–8 seconds (node_modules
  types are the bulk of it)

---

### 2.3 Persistence of index state

Pyright caches index data in `~/.cache/pyright/` by default. The tsserver
project service caches in the system temp directory. Both survive process
restarts, so the second invocation of the bridge in a session is fast.

No explicit persistence management needed for Phase 1. If the host is ephemeral
(CI, cloud dev), cache directories can be bind-mounted; this is a Phase 2+
concern.

---

### 2.4 Resource profile

| Server | Idle RSS | Active (during index) | Notes |
|---|---|---|---|
| pyright-langserver | ~80–150 MB | ~250 MB | Node process |
| typescript-language-server | ~100–200 MB | ~400 MB | Node + tsserver |

Both are acceptable on a machine with 16+ GB RAM alongside the ML workload.
If VRAM pressure is a concern, language servers can be stopped between
generation sessions — the MCP bridge handles restart transparently.

---

## Phase 3 — MCP Tooling (HTTP Bridge)

### 3.1 Architecture

```
Claude Code
    │  MCP (stdio or HTTP)
    ▼
LSP MCP Bridge  ──── JSON-RPC stdio ────► pyright-langserver
                └─── JSON-RPC stdio ────► typescript-language-server
```

The bridge is the translation layer: it receives MCP tool calls, converts them
to LSP requests, awaits responses, and returns structured results.

**Transport choice: HTTP**

The bridge exposes an HTTP server (e.g. `localhost:7890`). Claude Code connects
via the MCP HTTP transport. Advantages over stdio-as-MCP:
- The bridge (and the two language servers it manages) live as a persistent
  background process, not a subprocess of Claude Code.
- Multiple Claude Code sessions can share the same bridge without re-indexing.
- The bridge is restartable independently of the editor session.
- HTTP is debuggable with `curl` during development.

---

### 3.2 Existing options to evaluate

Before building, evaluate these existing bridges:

| Project | Notes |
|---|---|
| `mcp-language-server` (Kiwi/community) | Wraps arbitrary LSP via stdio; MCP stdio transport; may need HTTP wrapper |
| `lsp-mcp` variants | Various community experiments; quality varies |
| Roll a minimal custom bridge | ~200–400 lines of Python; full control; matches our exact tool surface |

**Recommendation:** Evaluate `mcp-language-server` first. If it exposes the
tool surface we need (see §3.3) and supports HTTP transport or can be wrapped,
use it. If it is too generic or unmaintained, roll a focused custom bridge. The
custom bridge is small enough that it is not a significant investment.

---

### 3.3 Tool surface

Minimum viable tool set (Phase 3a):

| Tool | LSP method | Input | Output |
|---|---|---|---|
| `lsp/hover` | `textDocument/hover` | file path, line, col | type info + doc string |
| `lsp/definition` | `textDocument/definition` | file path, line, col | target file + line |
| `lsp/references` | `textDocument/references` | file path, line, col | list of {file, line, col} |
| `lsp/diagnostics` | `textDocument/publishDiagnostics` | file path (or project) | list of {severity, message, line} |

Extended tool set (Phase 3b, once the bridge is stable):

| Tool | LSP method | Input | Output |
|---|---|---|---|
| `lsp/rename` | `textDocument/rename` | file, line, col, new name | workspace edit (diff per file) |
| `lsp/call_hierarchy_in` | `callHierarchy/incomingCalls` | file, line, col | callers with locations |
| `lsp/call_hierarchy_out` | `callHierarchy/outgoingCalls` | file, line, col | callees with locations |
| `lsp/signature_help` | `textDocument/signatureHelp` | file, line, col | parameter names + types |

`lsp/rename` should return a **dry-run diff** by default, not apply changes
directly. The agent reviews the diff and calls an apply tool separately. This
matches the Plan → Review → Implement cycle this project already follows.

---

### 3.4 Bridge design (custom path)

If building a custom bridge, the minimal structure:

```
lsp-mcp-bridge/
  bridge.py          # FastAPI HTTP server + MCP tool handlers
  lsp_client.py      # JSON-RPC stdio client for a single language server
  manager.py         # Spawns + monitors LSP processes; routes by file extension
  config.py          # Workspace root, binary paths, port from env vars
```

**Key design points:**

- `manager.py` routes by file extension: `.py` → pyright, `.js`/`.jsx` → tsserver.
- Each `LspClient` holds an open subprocess and a `asyncio` reader/writer on
  its stdio. Requests are multiplexed by `id`.
- The bridge sends `textDocument/didOpen` before any per-file request, and
  `textDocument/didClose` after. This gives the language server enough context
  to resolve imports without requiring a full workspace scan first.
- Diagnostics are pull-based from the agent's perspective: the tool opens the
  file, waits for `publishDiagnostics` notification (with a short timeout), and
  returns whatever arrived. This avoids requiring the agent to subscribe to
  push events.

**HTTP endpoint:**
```
POST /mcp
Body: MCP JSON-RPC tool call
Response: MCP tool result
```

Standard MCP HTTP transport. Claude Code connects by adding to its MCP config:
```json
{
  "mcpServers": {
    "lsp": {
      "transport": "http",
      "url": "http://localhost:7890/mcp"
    }
  }
}
```

---

### 3.5 Integration with treesitter workflow

The combined workflow after Phase 3:

```
Task: understand impact of changing WorkerPool.switch_mode signature
  1. treesitter find_usages → all call sites (fast, syntactic)
  2. lsp/references → type-resolved call sites (filters out false positives)
  3. lsp/hover at each site → confirm argument types match new signature
  4. lsp/diagnostics on affected files → verify no type errors after change
  5. treesitter affected_by_diff → structural impact map
```

Neither tool replaces the other. The agent picks the right one based on what
the question actually requires.

---

## Execution Order

```
Phase 1 (config files + type annotation pass) ✅ DONE
  ├── ✅ npm install -g pyright typescript-language-server typescript
  ├── ✅ create pyrightconfig.json
  ├── ✅ create lcm-sr-ui/jsconfig.json
  ├── ✅ type annotation / # type: ignore pass across backends/, server/,
  │       invokers/, persistence/ (2026-03)
  └── ✅ pyright . → 0 errors, 0 warnings

Phase 2 (process model decision, no new services yet)  ⬜ PENDING
  └── document the host-local deployment decision in env.custom or README

Phase 3a (bridge MVP)  ⬜ PENDING
  ├── evaluate existing mcp-language-server
  ├── build or adopt bridge with lsp/hover, lsp/definition, lsp/references, lsp/diagnostics
  ├── wire into Claude Code MCP config
  └── validate: hover on worker_pool.py fut field returns correct type

Phase 3b (extended tools)  ⬜ PENDING
  ├── lsp/rename (dry-run diff)
  ├── lsp/call_hierarchy_in + _out
  └── update CLAUDE.md tool selection table
```

---

## Success Criteria

| Phase | Done when | Status |
|---|---|---|
| 1 | `pyright .` runs clean; `jsconfig.json` accepted by tsserver with no parse errors | ✅ **DONE** (2026-03) |
| 2 | Bridge process model documented; env vars defined; startup/shutdown scripted | ⬜ pending |
| 3a | Agent can ask "what type is `fut` on line X of `worker_pool.py`?" and get a correct answer | ⬜ pending |
| 3b | Agent can rename `_free_worker` and receive a per-file diff it can review before applying | ⬜ pending |
