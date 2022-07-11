"""Unit tests for our Utils module"""

import os
import requests

from notebooks import utils
import pandas as pd

VERSION = 18
URL = 'https://e2-demo-field-eng.cloud.databricks.com/model-endpoint/telco_churn_model/{}/invocations'

def score_model(dataset: pd.DataFrame):
  url = URL.format(VERSION)
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  data_json = dataset.to_dict(orient='split')
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()


def test_endpoint():
    """Test prediction endpoint."""
    
    