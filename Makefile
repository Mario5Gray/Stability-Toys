include Makefile.test

.PHONY: install-st
install-st: ## Install the st CLI to ~/.local/bin
	mkdir -p ~/.local/bin
	cd cli/go && go build -o ~/.local/bin/st ./cmd/st
