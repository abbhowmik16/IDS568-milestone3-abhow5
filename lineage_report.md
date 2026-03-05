# IDS 568 – Milestone 3

### Project Overview

This report documents the **data lineage of the machine learning pipeline**, describing how data flows through the system from raw input to final model artifacts.

Data lineage ensures transparency, traceability, and reproducibility of machine learning workflows.
# Data Pipeline Overview

The pipeline consists of the following stages:

Raw Dataset  
↓  
Data Preprocessing  
↓  
Processed Dataset  
↓  
Model Training  
↓  
Model Evaluation  
↓  
Model Artifact Generation
# Stage 1: Raw Data

Dataset used:

Iris Dataset

The dataset contains measurements of iris flowers used to classify species.

Features:

- sepal length
- sepal width
- petal length
- petal width

Target variable:

- iris species classification
# Stage 2: Data Preprocessing

Script:
preprocess.py
This script performs the following tasks:

- Loads the raw dataset
- Cleans and formats the data
- Prepares the dataset for model training
- Saves the processed dataset

Output:
data/processed/iris_processed.csv
The processed dataset includes the features required for model training and the target label.

# Stage 3: Model Training

Script:
train.py
The training pipeline performs the following steps:
1. Loads the processed dataset
2. Splits the data into training and testing sets
3. Trains a Logistic Regression model
4. Evaluates model performance
5. Saves the trained model

Output artifacts:
models/model.pkl
reports/metrics.json
Metrics recorded:
- Accuracy
- F1 Score (macro)
# Stage 4: Experiment Tracking

Experiment tracking is implemented using **MLflow**.

MLflow records:

Parameters:

- C
- max_iter
- test_size
- random_state

Metrics:

- accuracy
- f1_macro

Artifacts:

- trained model
- metrics.json

Experiment name:
ids568_milestone3
MLflow enables full reproducibility of the training process.
# Stage 5: Model Validation

Script:
model_validation.py
This stage evaluates the trained model against predefined performance thresholds.

Validation rules:

Accuracy ≥ 0.90  
F1 Score ≥ 0.80

If the model does not meet these criteria, the pipeline fails.

This ensures that only high-quality models pass the validation stage.
# Stage 6: CI/CD Automation

The pipeline is automated using **GitHub Actions**.

Workflow file:
.github/workflows/train_and_validate.yml
CI pipeline steps:

1. Install dependencies
2. Run preprocessing
3. Train model
4. Validate model
5. Upload artifacts

This ensures consistent execution of the pipeline whenever new code is pushed to the repository.
# Lineage Summary

The data lineage for the system can be summarized as:

Raw Dataset  
↓  
Preprocessing (`preprocess.py`)  
↓  
Processed Dataset (`iris_processed.csv`)  
↓  
Model Training (`train.py`)  
↓  
Model Artifact (`model.pkl`)  
↓  
Evaluation Metrics (`metrics.json`)  
↓  
Model Validation (`model_validation.py`)
# Conclusion

The pipeline establishes a fully traceable machine learning workflow where:

- every transformation step is documented
- model artifacts are tracked
- experiments are reproducible
- model performance is validated automatically

This lineage framework supports transparency, reliability, and reproducibility for machine learning development and deployment.
