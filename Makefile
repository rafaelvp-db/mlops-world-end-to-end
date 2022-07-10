env:
	python -m venv .venv && \
	source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r unit-requirements.txt && \
	pip install -e .

utils:
	source .venv/bin/activate && pytest ./tests/unit/test_utils.py

wrapper:
	source .venv/bin/activate && pytest ./tests/unit/test_wrapper.py