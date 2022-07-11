phony: utils wrapper mlflow

env:
	python -m venv .venv && \
	source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r unit-requirements.txt && \
	pip install -e .

data:
	wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv -O /tmp/churn.csv

utils:
	source .venv/bin/activate && pytest ./tests/unit/test_utils.py

builder:
	source .venv/bin/activate && pytest ./tests/unit/test_builder.py

endpoint:
	source .venv/bin/activate && pytest ./tests/integration/test_endpoint.py

model:
	make utils && make builder

lint:
	source .venv/bin/activate && black notebooks/model_builder.py && black notebooks/utils.py

flake:
	source .venv/bin/activate && flake8 notebooks/model_builder.py && flake8 notebooks/utils.py

unit:
	export PYSPARK_SUBMIT_ARGS='--packages io.delta:delta-core_2.12:1.2.1 pyspark-shell' && \
	source .venv/bin/activate && pytest tests/unit