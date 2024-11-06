# Introduction

# Procedure

## Data Preprocessing
* Data Cleansing: Remove duplicates, malfunction values
* Encoding categorical variables
* Normalization

## EDA
* Visualize data distribution (check for imbalances)
* Identify correlations between features

## Model Selection
* Implement multiple models in `src/models`
* hyperparameter tuning (GridSearch for example) - Hydra?
* Regularization & early stopping
* W&B to monitor the change

## Evaluation
* Confusion Matrix
* Recall / F1-score
* Others? For imbalance data, MPC / Focal loss

## Don't forget the Unit Test!!

## System Design
* clean pipeline for data ingestion, preprocessing, model training, and evaluation. 
* Consider using a framework (e.g., scikit-learn Pipelines) or even creating a custom class for the entire pipeline.
* Data & Model Storage: 
  * Versioning models or saving artifacts like preprocessed data or metrics. 
  * You could use simple solutions like joblib for model serialization or even suggest integrating with a database if scaling is in mind.
  
* Logging: Loguru

