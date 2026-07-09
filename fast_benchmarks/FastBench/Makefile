# FastBench developer commands.  Run `make help` for the list.
VENV ?= .venv
PY   := $(VENV)/bin/python

.DEFAULT_GOAL := help
.PHONY: help setup test run report clean distclean

help:                ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:               ## create the venv and install core solvers
	./setup.sh

test:                ## run the test suite
	$(PY) -m pytest tests/ -q

run:                 ## run the full benchmark (all problems x available solvers)
	$(PY) -m fastbench.cli run --episodes 10

report:              ## rebuild the report from results/results.json
	$(PY) -m fastbench.cli report --results results/results.json

clean:               ## remove build/run artifacts (keeps .venv and acados)
	@$(PY) -m fastbench.cli clean 2>/dev/null || $(MAKE) clean-fallback

distclean:           ## full reset to source-only (also removes .venv, acados, examples)
	@$(PY) -m fastbench.cli clean --all 2>/dev/null || $(MAKE) distclean-fallback

# fallbacks if the venv/python is not present (e.g. before setup or after distclean)
.PHONY: clean-fallback distclean-fallback
clean-fallback:
	rm -rf results fastsqp_build build dist .pytest_cache *.egg-info \
	       c_generated_code* acados_ocp_*.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned build/run artifacts"

distclean-fallback: clean-fallback
	rm -rf $(VENV) third_party examples
	@echo "reset to minimal source"
