# Health & Anomaly Detection Model

This repository contains a machine learning pipeline for detecting anomalies and predicting health scores based on wearable device data. The models use an XGBoost classifier for anomaly detection and an XGBoost regressor for health score prediction. The pipeline includes data preprocessing, model training, evaluation, and SHAP-based explainability.

## Features
- **Data Preprocessing:** Handles missing values, normalizes numerical features, and encodes categorical features.
- **Anomaly Detection:** Uses an XGBoost classifier to predict whether a user's health data is anomalous.
- **Health Score Prediction:** Uses an XGBoost regressor to estimate a user's health score.
- **SHAP Explainability:** Provides insights into feature contributions for predictions.
- **Human-Readable Explanations:** Generates interpretable text-based explanations for model outputs.

## Installation
Ensure you have Python 3.8+ installed. Then install dependencies:
```sh
pip install pandas numpy shap matplotlib scikit-learn xgboost
```

## Usage

### 1. Data Preparation
Ensure you have a dataset named `merged_output.csv` with the required features.

### 2. Run the Model
Execute the script to train models, evaluate them, and generate explanations.
```sh
python script.py
```

### 3. Making Predictions
Modify the `new_entry` DataFrame to input custom user data and predict health scores and anomaly flags.

## Model Evaluation
- **Classification Report** for anomaly detection
- **MSE, RMSE, and R² Score** for health score prediction

## Explainability
The script provides:
- **SHAP Summary Plots** for feature importance visualization
- **Force Plots** to explain individual predictions
- **Human-Readable Explanations** for interpretability

## Example Output
```
Anomaly Flag Prediction: 0
Health Score Prediction: 85.3

Top Anomaly Explanation Features:
[('Heart_Rate', 0.45), ('Steps', -0.32), ('BMI', 0.29), ('Sleep_Efficiency', -0.21), ('Calories_Intake', 0.18)]

Human-Readable Explanation:
The model predicts that the user's condition is normal. Key factors influencing this decision include: a higher Heart_Rate (which raises the anomaly risk), a lower Steps (which helps maintain a normal status). The predicted health score is 85.3. Important factors affecting this score include: a higher Sleep_Efficiency (which improves the score), a lower BMI (which may lower the score).
```

## SHAP Visualizations
To generate SHAP-based explainability plots, ensure `matplotlib` is installed and run:
```sh
python script.py
```

## Contributing
Feel free to fork this repository and submit pull requests with improvements or bug fixes.

## License
MIT License
