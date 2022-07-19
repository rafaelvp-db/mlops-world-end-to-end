"""Unit tests for our Utils module"""

import os
import requests

from notebooks import utils
import pandas as pd

from fixtures import *


def test_endpoint(endpoint_url, version):
    """Basic test prediction endpoint."""

    url = endpoint_url.format(version)
    response = requests.post(url = url)
    assert response.status_code != 200 #Unauthorized, etc

    
    