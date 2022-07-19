.PHONY: utils wrapper mlflow deploy-data-prep deploy-build-model deploy-db data

.SECONDARY: env

env:
	export SYSTEM_VERSION_COMPAT=1 && \
	python -m venv .venv && \
	source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r unit-requirements.txt && \
	pip install -e .

data:
	wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv -O ./data/ibm_telco_churn.csv

builder: env
	pytest ./tests/unit/test_builder.py

endpoint: env
	pytest ./tests/integration/test_endpoint.py

format:
	black telco_churn_mlops/jobs/ && black telco_churn_mlops/pipelines

lint:
	flake8 telco_churn_mlops/jobs/ && flake8 telco_churn_mlops/pipelines/

clean:
	rm -rf spark-warehouse

unit: env clean
	export MLFLOW_TRACKING_URI="sqlite:///mlruns.db" && \
	pytest --cov-report term --cov=telco_churn_mlops tests/unit

deploy-prep:
	dbx deploy --deployment-file=conf/data_prep/deployment.json

deploy-builder:
	dbx deploy --deployment-file=conf/build_model/deployment.json

deploy-model:
	dbx deploy --deployment-file=conf/deploy_model/deployment.json

execute-data:
	dbx execute --job prep_data --deployment-file=conf/data_prep/deployment.json --cluster-name "Shared Autoscaling EMEA"

execute-builder:
	dbx execute --job build_model --deployment-file=conf/build_model/deployment.json --cluster-name "Shared Autoscaling EMEA"

execute-deploy:
	dbx execute --job deploy_model --deployment-file=conf/deploy_model/deployment.json --cluster-name "Shared Autoscaling EMEA"

deploy: deploy-prep deploy-builder deploy-model
