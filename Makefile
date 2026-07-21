include Makefile.test

.PHONY: install
install: install-st install-controlnet-scripts install-qrng ## Install all production targets

LOCAL_BIN ?= $(HOME)/.local/bin

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
install-controlnet-scripts: ## Install st-depth-map, st-pose-map, and st-canny-map console scripts (use EXTRAS=[depth|pose|canny|all])
	pip install "./scripts[$(or $(EXTRAS),all)]"

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
	docker compose -f docker-compose.dev.yml up

.PHONY: dev-build
dev-build: ## Rebuild dev image (source-only overlay, seconds)
	docker compose -f docker-compose.dev.yml build

.PHONY: dev-down
dev-down: ## Stop dev container
	docker compose -f docker-compose.dev.yml down

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
