phony: utils wrapper mlflow

env:
	python -m venv .venv && \
	source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r unit-requirements.txt && \
	pip install -e .

utils:
	source .venv/bin/activate && pytest ./tests/unit/test_utils.py

builder:
	source .venv/bin/activate && pytest ./tests/unit/test_builder.py

model:
	make utils && make builder

lint:
	source .venv/bin/activate && black notebooks/model_builder.py && black notebooks/utils.py