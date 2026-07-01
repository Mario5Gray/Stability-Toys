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
