from setuptools import find_packages, setup
from telco_churn_mlops import __version__

setup(
    name="telco_churn_mlops",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    version=__version__,
    description="",
    author=""
)
