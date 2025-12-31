SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

.PHONY: help
help:
	@echo "yA0tzy Make targets"
	@echo ""
	@echo "Core:"
	@echo "  make run                      Start inference server + TUI (all-in-one)"
	@echo "  make tui                      Run the Ratatui UI (yz tui)"
	@echo "  make infer-server             Run Python inference server (with hot-reload)"
	@echo ""
	@echo "Rust:"
	@echo "  make rust-test                 cargo test --workspace"
	@echo "  make rust-fmt                  cargo fmt"
	@echo "  make rust-clippy               cargo clippy --workspace -- -D warnings"
	@echo ""
	@echo "Python:"
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
	@set -euo pipefail; \
	bind="$(INFER_BIND)"; \
	metrics="$(METRICS_BIND)"; \
	log_file="/tmp/yatzy_infer_server.log"; \
	metrics_host="$${metrics%:*}"; \
	metrics_port="$${metrics##*:}"; \
	echo "Starting inference server on $$bind (metrics $$metrics) ..."; \
	echo "Server logs: $$log_file"; \
	existing_pid=$$(lsof -ti :$$metrics_port 2>/dev/null || true); \
	if [[ -n "$$existing_pid" ]]; then \
	  echo "Port $$metrics_port already in use (pid $$existing_pid), killing..."; \
	  kill "$$existing_pid" 2>/dev/null || true; \
	  sleep 0.2; \
	fi; \
	if [[ "$$bind" == unix://* ]]; then \
	  sock="$${bind#unix://}"; \
	  rm -f "$$sock"; \
	fi; \
	( $(PY_RUN) -m yatzy_az infer-server --best dummy --cand dummy --bind "$$bind" --metrics-bind "$$metrics" --print-stats-every-s 0 >"$$log_file" 2>&1 ) & \
	infer_pid="$$!"; \
	trap 'echo ""; echo "Stopping inference server (pid $$infer_pid)"; kill "$$infer_pid" 2>/dev/null || true; wait "$$infer_pid" 2>/dev/null || true; if [[ "$$bind" == unix://* ]]; then rm -f "$${bind#unix://}"; fi' EXIT INT TERM; \
	if [[ "$$bind" == unix://* ]]; then \
	  sock="$${bind#unix://}"; \
	  echo "Waiting for socket $$sock ..."; \
	  for _ in $$(seq 1 100); do \
	    if [[ -S "$$sock" || -e "$$sock" ]]; then break; fi; \
	    sleep 0.05; \
	  done; \
	  if [[ ! -e "$$sock" ]]; then \
	    echo "Server failed to start. Check $$log_file for details."; \
	    cat "$$log_file" | tail -20; \
	    exit 1; \
	  fi; \
	else \
	  sleep 0.25; \
	fi; \
	echo "Starting TUI ..."; \
	echo "(Press 'g' on Config screen to start an iteration)"; \
	cargo run -p yz-cli --bin yz -- tui

.PHONY: infer-server
infer-server:
	$(PY_RUN) -m yatzy_az infer-server --best dummy --cand dummy --bind $(INFER_BIND) --metrics-bind $(METRICS_BIND) --print-stats-every-s 0

# Convenience wrappers. These assume a run id under ./runs/<id>/.
.PHONY: selfplay train gate
selfplay:
	@if [ -z "$(RUN)" ]; then echo "Missing RUN=<id> (e.g. make selfplay RUN=smoke)"; exit 2; fi
	cargo run -p yz-cli --bin yz -- selfplay --config $(CONFIG) --infer $(INFER_BIND) --out runs/$(RUN)/

train:
	@if [ -z "$(RUN)" ]; then echo "Missing RUN=<id> (e.g. make train RUN=smoke)"; exit 2; fi
	$(PY_RUN) -m yatzy_az train --replay runs/$(RUN)/replay --out runs/$(RUN)/models --config runs/$(RUN)/config.yaml

gate:
	@if [ -z "$(RUN)" ]; then echo "Missing RUN=<id> (e.g. make gate RUN=smoke)"; exit 2; fi
	cargo run -p yz-cli --bin yz -- gate --config $(CONFIG) --infer $(INFER_BIND) --run runs/$(RUN)/


