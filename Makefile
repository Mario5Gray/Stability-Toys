include Makefile.test

.PHONY: install
install: install-st install-controlnet-scripts ## Install all production targets

.PHONY: install-st
install-st: ## Install the st CLI to ~/.local/bin
	mkdir -p ~/.local/bin
	cd cli/go && go build -o ~/.local/bin/st ./cmd/st

.PHONY: install-controlnet-scripts
install-controlnet-scripts: ## Install st-depth-map, st-pose-map, and st-canny-map console scripts (use EXTRAS=[depth|pose|canny|all])
	pip install "./scripts[$(or $(EXTRAS),all)]"

.PHONY: quick-build
quick-build: ## Rebuild only Python source into existing image (no deps/UI/CUDA reinstall). Override: make quick-build IMAGE=foo:tag
	docker build -f Dockerfile.quick \
	  --build-arg BASE_IMAGE=$(or $(IMAGE),harbor.lan/lcm-sd-ui:latest) \
	  -t $(or $(IMAGE),harbor.lan/lcm-sd-ui:latest) \
	  .
