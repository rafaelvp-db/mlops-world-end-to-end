.PHONY: utils wrapper mlflow deploy-data-prep deploy-build-model deploy-db data

env:
	export SYSTEM_VERSION_COMPAT=1 && \
	python -m venv .venv && \
	source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r unit-requirements.txt

data:
	wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv -O ./data/ibm_telco_churn.csv

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
	rm -rf spark-warehouse && \
	source .venv/bin/activate && pip install -e . && pytest tests/unit

deploy-prep:
	dbx deploy --deployment-file=conf/data_prep/deployment.json

deploy-builder:
	dbx deploy --deployment-file=conf/build_model/deployment.json

deploy-model:
	dbx deploy --deployment-file=conf/deploy_model/deployment.json

launch-data:
	dbx launch --job data_prep --trace

launch-builder:
	dbx launch --job build_model --trace

execute-data:
	dbx execute --job data_prep --deployment-file=conf/data_prep/deployment.json --cluster-name "Shared Autoscaling EMEA"

execute-builder:
	dbx execute --job build_model --deployment-file=conf/build_model/deployment.json --cluster-name "Shared Autoscaling EMEA"

execute-deploy:
	dbx execute --job deploy_model --deployment-file=conf/deploy_model/deployment.json --cluster-name "Shared Autoscaling EMEA"

deploy:
	make deploy-prep && make deploy-builder && make deploy-model