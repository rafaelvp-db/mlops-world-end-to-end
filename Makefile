phony: utils wrapper mlflow deploy-data-prep deploy-build-model

env:
	python -m venv .venv && \
	source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r unit-requirements.txt && \
	pip install -e .

data:
	wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv -O /tmp/ibm_telco_churn.csv

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
	source .venv/bin/activate && pip install -e . && pytest tests/unit

deploy-prep:
	source .venv/bin/activate && dbx deploy --deployment-file=conf/data_prep/deployment.json

deploy-builder:
	source .venv/bin/activate && dbx deploy --deployment-file=conf/build_model/deployment.json

launch-builder:
	source .venv/bin/activate && dbx launch --job build_model --trace

deploy:
	make deploy-prep && make deploy-builder