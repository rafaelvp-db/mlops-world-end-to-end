"""Unit tests for our Utils module"""

import os
import requests

from notebooks import utils
import pandas as pd

VERSION = 18
URL = 'https://e2-demo-field-eng.cloud.databricks.com/model-endpoint/telco_churn_model/{}/invocations'


def test_endpoint():
    """Basic test prediction endpoint."""

    url = URL.format(VERSION)
    response = requests.post(url = url)
    assert response.status_code == 401 #Unauthorized

    
    