phony: utils wrapper mlflow deploy-data-prep deploy-build-model deploy-db

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
	black telco_churn_mlops/jobs/ && black telco_churn_mlops/pipelines

flake:
	flake8 telco_churn_mlops/jobs/ && flake8 telco_churn_mlops/pipelines/

unit:
	source .venv/bin/activate && pip install -e . && pytest tests/unit

deploy-prep:
	dbx deploy --deployment-file=conf/data_prep/deployment.json

deploy-builder:
	dbx deploy --deployment-file=conf/build_model/deployment.json

deploy-ab:
	dbx deploy --deployment-file=conf/ab_test/deployment.json

launch-data:
	dbx launch --job data_prep --trace

launch-builder:
	dbx launch --job build_model --trace

execute-builder:
	dbx execute --job ab_test --deployment-file=conf/ab_test/deployment.json --cluster-id 1011-090100-bait793

execute-ab:
	dbx execute --job ab_test --deployment-file=conf/ab_test/deployment.json --cluster-id 1011-090100-bait793

deploy:
	make deploy-prep && make deploy-builder && make deploy-ab