SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

.PHONY: help
help:
	@echo "yA0tzy Make targets"
	@echo ""
	@echo "Core:"
	@echo "  make run                      Start the TUI (release; fast default)"
	@echo "  make run-dev                  Start the TUI (dev; faster compile, slower runtime)"
	@echo "  make tui                      Run the Ratatui UI (yz tui)"
	@echo "  make start-run RUN_NAME=<id> CONFIG=<path>   Start a run from a config file (release; prints table)"
	@echo "  make controller RUN=<id>      Run runs/<id> in foreground and print the iteration table"
	@echo "  make cancel RUN=<id>          Request cancel for runs/<id> (writes cancel.request)"
	@echo "  make infer-server             Run Python inference server (with hot-reload)"
	@echo ""
	@echo "Rust:"
	@echo "  make rust-test                 cargo test --workspace"
	@echo "  make rust-fmt                  cargo fmt"
	@echo "  make rust-clippy               cargo clippy --workspace -- -D warnings"
	@echo ""
	@echo "Python:"
	@echo "  make py-sync                   uv sync (install python deps incl. torch)"
	@echo "  make py-test                   pytest"
	@echo "  make py-ruff                   ruff check + format --check"
	@echo "  make py-ruff-fix               ruff check --fix + format"
	@echo ""
	@echo "Pipelines (require infer-server running):"
	@echo "  make selfplay RUN=<id>         yz selfplay --out runs/<id>/ (uses configs/local_cpu.yaml)"
	@echo "  make train RUN=<id>            python -m yatzy_az train (uses runs/<id>/config.yaml)"
	@echo "  make gate RUN=<id>             yz gate --run runs/<id>/ (uses configs/local_cpu.yaml)"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON=<exe>                   Python executable (default: python)"
	@echo "  INFER_BIND=<endpoint>          Default: unix:///tmp/yatzy_infer.sock"
	@echo "  METRICS_BIND=<host:port>       Default: 127.0.0.1:18080"
	@echo "  CONFIG=<path>                  Default: configs/local_cpu.yaml"
	@echo "  RUN=<id>                       Run id under runs/"
	@echo "  RUN_NAME=<id>                  Run name for start-run (directory under runs/)"
	@echo "  DETACH=1                       Detach start-run (controller runs as child process)"

.PHONY: _print_env
_print_env:
	@echo "PYTHON=$(PYTHON)"
	@echo "INFER_BIND=$(INFER_BIND)"
	@echo "METRICS_BIND=$(METRICS_BIND)"
	@echo "CONFIG=$(CONFIG)"
	@echo "RUN=$(RUN)"

PYTHON ?= python
INFER_BIND ?= unix:///tmp/yatzy_infer.sock
METRICS_BIND ?= 127.0.0.1:18080
CONFIG ?= configs/local_cpu.yaml

# Prefer `uv run` if uv exists; otherwise fall back to plain python.
UV := $(shell command -v uv 2>/dev/null || true)
ifeq ($(strip $(UV)),)
PY_RUN := cd python && $(PYTHON)
RUFF_RUN := cd python && $(PYTHON) -m ruff
PYTEST_RUN := cd python && $(PYTHON) -m pytest
else
PY_RUN := cd python && uv run $(PYTHON)
RUFF_RUN := cd python && uv run ruff
PYTEST_RUN := cd python && uv run $(PYTHON) -m pytest
endif

.PHONY: rust-test rust-fmt rust-clippy
rust-test:
	cargo test --workspace

rust-fmt:
	cargo fmt

rust-clippy:
	cargo clippy --workspace -- -D warnings

.PHONY: py-test py-ruff py-ruff-fix
py-sync:
	cd python && uv sync

py-test:
	$(PYTEST_RUN) -q

py-ruff:
	$(RUFF_RUN) check .
	$(RUFF_RUN) format --check .

py-ruff-fix:
	$(RUFF_RUN) check --fix .
	$(RUFF_RUN) format .

.PHONY: tui
tui:
	cargo run -p yz-cli --bin yz -- tui

.PHONY: run
run:
	@echo "Starting TUI (release) ..."; \
	echo "(Press 'g' on Config screen to start an iteration)"; \
	echo "(The controller will start/reuse the inference server using the current config.)"; \
	cargo run --release -p yz-cli --bin yz -- tui

.PHONY: run-dev
run-dev:
	@echo "Starting TUI (dev) ..."; \
	echo "(Press 'g' on Config screen to start an iteration)"; \
	echo "(The controller will start/reuse the inference server using the current config.)"; \
	cargo run -p yz-cli --bin yz -- tui

.PHONY: infer-server
infer-server:
	$(PY_RUN) -m yatzy_az infer-server --best dummy --cand dummy --bind $(INFER_BIND) --metrics-bind $(METRICS_BIND) --print-stats-every-s 0

# Experiment CLI (release by default)
.PHONY: start-run controller cancel
start-run:
	@if [ -z "$(RUN_NAME)" ]; then echo "Missing RUN_NAME=<id> (e.g. make start-run RUN_NAME=exp1 CONFIG=/tmp/cfg.yaml)"; exit 2; fi
	@if [ -z "$(CONFIG)" ]; then echo "Missing CONFIG=<path> (e.g. make start-run RUN_NAME=exp1 CONFIG=/tmp/cfg.yaml)"; exit 2; fi
	cargo run --release -p yz-cli --bin yz -- start-run --run-name "$(RUN_NAME)" --config "$(CONFIG)" --python-exe "$(PYTHON)" $(if $(DETACH),--detach,)

controller:
	@if [ -z "$(RUN)" ]; then echo "Missing RUN=<id> (e.g. make controller RUN=exp1)"; exit 2; fi
	cargo run --release -p yz-cli --bin yz -- controller --run-dir "runs/$(RUN)" --python-exe "$(PYTHON)" --print-iter-table

cancel:
	@if [ -z "$(RUN)" ]; then echo "Missing RUN=<id> (e.g. make cancel RUN=exp1)"; exit 2; fi
	@mkdir -p "runs/$(RUN)" && echo "ts_ms: $$(date +%s000)" > "runs/$(RUN)/cancel.request" && echo "cancel requested: runs/$(RUN)"

# Convenience wrappers. These assume a run id under ./runs/<id>/.
.PHONY: selfplay train gate
selfplay:
	@if [ -z "$(RUN)" ]; then echo "Missing RUN=<id> (e.g. make selfplay RUN=smoke)"; exit 2; fi
	cargo run -p yz-cli --bin yz -- selfplay --config $(CONFIG) --infer $(INFER_BIND) --out runs/$(RUN)/

.PHONY: selfplay-release
selfplay-release:
	@if [ -z "$(RUN)" ]; then echo "Missing RUN=<id> (e.g. make selfplay-release RUN=smoke)"; exit 2; fi
	cargo run --release -p yz-cli --bin yz -- selfplay --config $(CONFIG) --infer $(INFER_BIND) --out runs/$(RUN)/

train:
	@if [ -z "$(RUN)" ]; then echo "Missing RUN=<id> (e.g. make train RUN=smoke)"; exit 2; fi
	$(PY_RUN) -m yatzy_az train --replay runs/$(RUN)/replay --out runs/$(RUN)/models --config runs/$(RUN)/config.yaml

gate:
	@if [ -z "$(RUN)" ]; then echo "Missing RUN=<id> (e.g. make gate RUN=smoke)"; exit 2; fi
	cargo run -p yz-cli --bin yz -- gate --config $(CONFIG) --infer $(INFER_BIND) --run runs/$(RUN)/

.PHONY: gate-release
gate-release:
	@if [ -z "$(RUN)" ]; then echo "Missing RUN=<id> (e.g. make gate-release RUN=smoke)"; exit 2; fi
	cargo run --release -p yz-cli --bin yz -- gate --config $(CONFIG) --infer $(INFER_BIND) --run runs/$(RUN)/


