#  End-to-End MLOps demo with MLFlow for TelcoChurn use case 

## Challenges moving ML project into production

Moving ML project from a standalone notebook to a production-grade data pipeline is complex and require multiple competencies. 

Having a model up and running in a notebook isn't enough. We need to cover the end to end ML Project life cycle and solve the following challenges:

* Update data over time (production-grade ingestion pipeline)
* How to save, share and re-use ML features in the organization
* How to ensure a new model version respect quality standard and won't break the pipeline
* Model governance: what is deployed, how is it trained, by who, which data?
* How to monitor and re-train the model...

In addition, these project typically invole multiple teams, creating friction and potential silos

* Data Engineers, in charge of ingesting, preparing and exposing the data
* Data Scientist, expert in data analysis, building ML model
* ML engineers, setuping the ML infrastructure pipelines (similar to devops)

This has a real impact on the business, slowing down projects and preventing them from being deployed in production and bringing ROI.

## What's MLOps ?

MLOps is is a set of standards, tools, processes and methodology that aims to optimize time, efficiency and quality while ensuring governance in ML projects.

MLOps orchestrate a project life-cycle and adds the glue required between the component and teams to smoothly implement such ML pipelines.

Databricks is uniquely positioned to solve this challenge with the Lakehouse pattern. Not only we bring Data Engineers, Data Scientists and ML Engineers together in a unique platform, but we also provide tools to orchestrate ML project and accelerate the go to production.

## MLOps pipeline we'll implement

In this demo, we'll implement a full MLOps pipeline, step by step:

<img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/mlops-end2end-flow-0.png" width="1200">


## Project in Step by Step 

Our work is devided into multiple steps:
- **prepare and explore your dataset** 
  - ingest original data into Delta table
  - split data into train / test datasets and save them into Delta table
  - run feature preprocessing on Spark
  - save prepared tabls into Delta 
  - explore tables with Bamboolib 
  
- **trained initial baseline model** 
  - train a model with an AutoML for a baseline and initial code generation
  - used HyperOpt on Spark for a hyperparameters tunning for an advance model 
    - track each model with the MlFlow tracking 
  - register the best model into MlFlow registry
  - creating a wrapper in case other team members need to access it and run the model apart 
  
- **promoting model into Stage**
  - 
  