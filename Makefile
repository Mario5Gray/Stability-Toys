include Makefile.test

.PHONY: install
install: install-st install-controlnet-scripts install-qrng ## Install all production targets

LOCAL_BIN ?= $(HOME)/.local/bin
GIT_SHA ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
export GIT_SHA

.PHONY: install-st
install-st: ## Install the st and stcn CLI binaries to ~/.local/bin
	mkdir -p $(LOCAL_BIN)
	cd cli/go && go build -o $(LOCAL_BIN)/st ./cmd/st
	cd cli/go && go build -o $(LOCAL_BIN)/stcn ./cmd/stcn

.PHONY: install-qrng
install-qrng: ## Install the qrng quantum-seed script to ~/.local/bin (symlink; edits stay live)
	mkdir -p $(LOCAL_BIN)
	chmod +x $(CURDIR)/qrandom
	ln -sf $(CURDIR)/qrandom $(LOCAL_BIN)/qrng

.PHONY: install-qrng-ibm
install-qrng-ibm: install-qrng ## Also install qiskit for qrng --source ibm
	pip install "qiskit>=1.0" qiskit-ibm-runtime

.PHONY: uninstall-qrng
uninstall-qrng: ## Remove the qrng symlink from ~/.local/bin
	rm -f $(LOCAL_BIN)/qrng

.PHONY: install-controlnet-scripts
install-controlnet-scripts: ## Install st-depth-map, st-pose-map, and st-canny-map console scripts (use EXTRAS=[depth|pose|canny|vector|all])
	pip install "./scripts[$(or $(EXTRAS),all)]"

# Server install (STABL-yztmgiec). Spec: docs/superpowers/specs/2026-11-15-server-install-design.md
# Blocks (dirty tracked tree, same version) exit 0. FORCE=1 proceeds past them.
# Refusals (not a git checkout, PREFIX inside the checkout) exit 2.
PREFIX ?= /opt/stability-toys

.PHONY: install-server
install-server: ## Install the Docker server runtime to PREFIX (default /opt/stability-toys; FORCE=1 proceeds past blocks)
	@set -e; \
	top=$$(git rev-parse --show-toplevel 2>/dev/null) || { echo "install-server: not a git checkout" >&2; exit 2; }; \
	top=$$(cd "$$top" && pwd -P); \
	p='$(PREFIX)'; p=$${p%/}; \
	case "$$p" in /*) ;; *) p="$$PWD/$$p" ;; esac; \
	a="$$p"; rest=""; \
	while [ ! -d "$$a" ]; do rest="/$${a##*/}$$rest"; a=$${a%/*}; [ -n "$$a" ] || a=/; done; \
	base=$$(cd "$$a" && pwd -P); [ "$$base" != / ] || base=""; p="$$base$$rest"; \
	case "$$p/" in "$$top/"*) echo "install-server: PREFIX is inside the checkout: $$p" >&2; exit 2 ;; esac; \
	v=$$(git rev-parse --short HEAD); \
	if [ -z "$(FORCE)" ]; then \
		if [ -n "$$(git status --porcelain --untracked-files=no)" ]; then \
			echo "install-server: checkout has tracked changes; use FORCE=1"; exit 0; \
		fi; \
		if [ -f "$$p/VERSION" ] && [ "$$(cat "$$p/VERSION")" = "$$v" ]; then \
			echo "install-server: $$v already installed in $$p; use FORCE=1"; exit 0; \
		fi; \
	fi; \
	mkdir -p "$$p/conf" "$$p/store" "$$p/model" "$$p/workflows"; \
	seed() { if [ -e "$$2" ]; then echo "kept $$2"; else mkdir -p "$${2%/*}"; cp "$$1" "$$2"; echo "copied $$2"; fi; }; \
	seed "$$top/docker-cuda.yml" "$$p/compose.yaml"; \
	git -C "$$top" ls-files conf | while IFS= read -r f; do seed "$$top/$$f" "$$p/$$f"; done; \
	if [ -e "$$p/env.custom" ]; then echo "kept $$p/env.custom"; else : > "$$p/env.custom"; echo "copied $$p/env.custom"; fi; \
	for f in env.prod env.cuda; do cp "$$top/$$f" "$$p/$$f"; echo "copied $$p/$$f"; done; \
	printf '%s\n' "$$v" > "$$p/VERSION"; \
	echo "installed $$v to $$p"; \
	echo "start: cd \"$$p\" && docker compose up -d"

.PHONY: prod-build
prod-build: ## Full prod image build (--no-cache; override: make prod-build BACKEND=cuda PLATFORM=amd64 IMAGE=foo:tag CACHE=1)
	docker build -f Dockerfile \
	  $(if $(CACHE),,--no-cache) \
	  --build-arg BACKEND=$(or $(BACKEND),cuda) \
	  --platform=linux/$(or $(PLATFORM),amd64) \
	  -t $(or $(IMAGE),harbor.lan/stability-toys:latest) \
	  .

.PHONY: quick-build
quick-build: ## Rebuild only Python source into existing image (no deps/UI/CUDA reinstall). Override: make quick-build IMAGE=foo:tag
	docker build -f Dockerfile.quick \
	  --build-arg BASE_IMAGE=$(or $(IMAGE),harbor.lan/stability-toys:latest) \
	  -t $(or $(IMAGE),harbor.lan/stability-toys:latest) \
	  .

.PHONY: dev
dev: ## Start dev container (uvicorn --reload on Python source changes)
	# --force-recreate: compose reuses an existing container when only the
	# env_file changed, so an edit to env.dev appears to have no effect. That
	# is how a TRACING_ENABLED edit reads as "the feature is broken".
	docker compose -f docker-compose.dev.yml up --force-recreate

.PHONY: dev-build
dev-build: ## Rebuild dev image (source-only overlay, seconds)
	docker compose -f docker-compose.dev.yml build

.PHONY: dev-down
dev-down: ## Stop dev container
	docker compose -f docker-compose.dev.yml down

PYTHON ?= python

.PHONY: mps
mps: ## Start the MPS server natively (no container: Docker cannot expose Metal; see docs/MPS_BACKEND.md)
	@case "$(CONDA_PREFIX)" in \
	  */envs/stability-toys) ;; \
	  *) echo "Activate the project env first: conda activate stability-toys"; \
	     echo "The Miniforge base env drifts outside the project pins (STABL-zisphapv)."; \
	     echo "The MPS path runs natively. Containers on macOS have no Metal."; \
	     exit 2;; \
	esac
	set -a; . ./env.mps; set +a; \
	  $(PYTHON) scripts/check_mps_paths.py && \
	  $(PYTHON) -m uvicorn server.lcm_sr_server:app --host 0.0.0.0 --port 4200

define REQUIRE_DRIFT
	@command -v drift >/dev/null 2>&1 || { \
	  echo "drift not found on PATH. It binds markdown docs to code and lints for staleness."; \
	  echo "Without it these targets cannot run; see .claude/skills/drift/SKILL.md."; \
	  exit 127; \
	}
endef

.PHONY: drift
drift: ## Check doc<->code anchors (exit 1 = a bound doc describes code that has since changed)
	$(REQUIRE_DRIFT)
	drift check

.PHONY: drift-refs
drift-refs: ## Show which docs are bound to a file: make drift-refs FILE=backends/cuda_worker.py
	$(REQUIRE_DRIFT)
	@test -n "$(FILE)" || { echo "usage: make drift-refs FILE=<path>"; exit 2; }
	drift refs $(FILE)

.PHONY: drift-changed
drift-changed: ## Scope the check to docs bound to one path: make drift-changed FILE=server/mode_config.py
	$(REQUIRE_DRIFT)
	@test -n "$(FILE)" || { echo "usage: make drift-changed FILE=<path>"; exit 2; }
	drift check --changed $(FILE)
