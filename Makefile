phony: utils wrapper

env:
	python -m venv .venv && \
	source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r unit-requirements.txt && \
	pip install -e .

utils:
	source .venv/bin/activate && pytest ./tests/unit/test_utils.py

wrapper:
	rm -rf metastore_db && source .venv/bin/activate && pytest ./tests/unit/test_wrapper.py


model:
	make utils && make wrapper