# Convenience targets. `make` on its own lists them.
PY ?= python

.PHONY: help setup export validate check clean

help:
	@echo "setup     create venv and install requirements.txt"
	@echo "export    regenerate notebooks/consumer_interest_pipeline.py from the notebook"
	@echo "validate  check the notebook is well-formed"
	@echo "check     validate, plus assert nothing excluded has been committed"
	@echo "clean     remove checkpoints and bytecode (keeps outputs/ and cache/)"

setup:
	$(PY) -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	./venv/bin/python -m ipykernel install --user --name venv

export:
	jupytext --to py:percent --output notebooks/consumer_interest_pipeline.py \
	         notebooks/consumer_interest_pipeline.ipynb

validate:
	$(PY) scripts/validate_notebook.py

check: validate
	jupytext --to py:percent --output /tmp/regenerated.py notebooks/consumer_interest_pipeline.ipynb
	$(PY) scripts/check_export.py notebooks/consumer_interest_pipeline.py /tmp/regenerated.py
	@git ls-files --error-unmatch Olympics_Tokyo_tweets.csv >/dev/null 2>&1 \
	  && (echo "ERROR: corpus must not be committed" && exit 1) || true
	@git ls-files --error-unmatch outputs/coding_sheet.csv >/dev/null 2>&1 \
	  && (echo "ERROR: coding sheet must not be committed" && exit 1) || true
	@echo "check ok"

clean:
	rm -rf __pycache__ .ipynb_checkpoints notebooks/.ipynb_checkpoints
