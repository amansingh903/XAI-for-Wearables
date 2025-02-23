import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score

# Load the dataset from CSV
df = pd.read_csv("merged_output.csv")

# Feature Engineering: Compute BMI and Sleep Efficiency
df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
df['Sleep_Efficiency'] = df['Deep_Sleep_Duration'] / df['Sleep_Duration']

# Drop columns that are not used as features
df.drop(['User_ID', 'Timestamp'], axis=1, inplace=True)

# Define feature lists based on your merged dataset
numeric_features = [
    'Age', 'Weight', 'Height', 'Sleep_Duration', 'Deep_Sleep_Duration',
    'REM_Sleep_Duration', 'Wakeups', 'Heart_Rate', 'Blood_Oxygen_Level',
    'Steps', 'Calories_Burned', 'Distance_Covered', 'Exercise_Duration',
    'Calories_Intake', 'Water_Intake', 'Skin_Temperature', 'Ambient_Temperature',
    'Battery_Level', 'Body_Fat_Percentage', 'Muscle_Mass', 'Altitude',
    'UV_Exposure', 'Notifications_Received', 'Screen_Time', 'BMI', 'Sleep_Efficiency'
]

categorical_features = [
    'Gender', 'Medical_Conditions', 'Medication', 'Smoker',
    'Alcohol_Consumption', 'Snoring', 'ECG', 'Exercise_Type',
    'Exercise_Intensity', 'Stress_Level', 'Mood', 'Day_of_Week'
]

# Build preprocessing pipelines for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Separate features and target variables
X = df.drop(['Anomaly_Flag', 'Health_Score'], axis=1)
y_anomaly = df['Anomaly_Flag']
y_health = df['Health_Score']

# Split the data into training and test sets
X_train, X_test, y_train_a, y_test_a, y_train_h, y_test_h = train_test_split(
    X, y_anomaly, y_health, test_size=0.2, random_state=42
)

# Define pipelines for the anomaly classification and health score regression models
anomaly_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(scale_pos_weight=10, eval_metric='logloss'))
])

health_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror'))
])

# Train both models
anomaly_model.fit(X_train, y_train_a)
health_model.fit(X_train, y_train_h)

# Evaluation function to report model performance
def evaluate_models():
    # Evaluate anomaly detection
    y_pred_a = anomaly_model.predict(X_test)
    print("Anomaly Detection Performance:")
    print(classification_report(y_test_a, y_pred_a))
    print(f"Accuracy: {accuracy_score(y_test_a, y_pred_a):.2f}")
    
    # Evaluate health score prediction
    y_pred_h = health_model.predict(X_test)
    mse = mean_squared_error(y_test_h, y_pred_h)
    print("\nHealth Score Prediction Performance:")
    print(f"MSE: {mse:.2f}, RMSE: {np.sqrt(mse):.2f}")
    print(f"R² Score: {health_model.score(X_test, y_test_h):.2f}")

evaluate_models()

# Set up SHAP explainers for both models (note: we use the underlying model from the pipeline)
explainer_anomaly = shap.TreeExplainer(anomaly_model.named_steps['classifier'])
explainer_health = shap.TreeExplainer(health_model.named_steps['regressor'])

# Function for making predictions and generating SHAP explanations for a new sample
def predict_health_and_anomaly(new_data):
    # Preprocess new data using the pipeline
    processed_data = preprocessor.transform(new_data)
    
    # Get predictions
    anomaly_pred = anomaly_model.predict(new_data)[0]
    health_pred = health_model.predict(new_data)[0]
    
    # Calculate SHAP values using the processed data
    shap_values_anomaly = explainer_anomaly.shap_values(processed_data)[0]
    shap_values_health = explainer_health.shap_values(processed_data)[0]
    
    # Retrieve feature names after one-hot encoding for categorical variables
    feature_names = numeric_features + list(
        preprocessor.named_transformers_['cat']
        .named_steps['onehot'].get_feature_names_out(categorical_features)
    )
    
    return {
        'anomaly_prediction': anomaly_pred,
        'health_score_prediction': health_pred,
        'anomaly_explanation': dict(zip(feature_names, shap_values_anomaly)),
        'health_score_explanation': dict(zip(feature_names, shap_values_health))
    }

# Example: Predicting on a new data entry
new_entry = pd.DataFrame([{
    'Age': 35,
    'Gender': 'Male',
    'Weight': 75.5,
    'Height': 175.0,
    'Medical_Conditions': 'None',
    'Medication': 'No',
    'Smoker': 'No',
    'Alcohol_Consumption': 'Moderate',
    'Day_of_Week': 'Monday',
    'Sleep_Duration': 7.2,
    'Deep_Sleep_Duration': 2.5,
    'REM_Sleep_Duration': 1.8,
    'Wakeups': 2,
    'Snoring': 'No',
    'Heart_Rate': 72,
    'Blood_Oxygen_Level': 98.5,
    'ECG': 'Normal',
    'Steps': 8500,
    'Calories_Burned': 8500 * 0.05,
    'Distance_Covered': 8500 * 0.0008,
    'Exercise_Type': 'Running',
    'Exercise_Duration': 1.2,
    'Exercise_Intensity': 'Moderate',
    'Calories_Intake': 2500.0,
    'Water_Intake': 2.5,
    'Stress_Level': 'Low',
    'Mood': 'Neutral',
    'Skin_Temperature': 36.2,
    'Ambient_Temperature': 22.5,
    'Battery_Level': 80,
    'Body_Fat_Percentage': 18.5,
    'Muscle_Mass': 65.0,
    'Altitude': 100,
    'UV_Exposure': 1.5,
    'Notifications_Received': 10,
    'Screen_Time': 2.0,
    # Engineered features
    'BMI': 75.5 / ((175.0 / 100) ** 2),
    'Sleep_Efficiency': 2.5 / 7.2
}])
# Ensure the new_entry DataFrame has all required columns in the proper order
required_columns = numeric_features + categorical_features
new_entry = new_entry[required_columns]

result = predict_health_and_anomaly(new_entry)
print("\nPrediction Results:")
print(f"Anomaly Flag Prediction: {result['anomaly_prediction']}")
print(f"Health Score Prediction: {result['health_score_prediction']:.2f}")

print("\nTop Anomaly Explanation Features:")
print(sorted(result['anomaly_explanation'].items(), key=lambda x: abs(x[1]), reverse=True)[:5])

print("\nTop Health Score Explanation Features:")
print(sorted(result['health_score_explanation'].items(), key=lambda x: abs(x[1]), reverse=True)[:5])


# -------------------------------------------------
# Additional: Generate Human-Readable Explanations
# -------------------------------------------------

def generate_human_explanation(result):
    explanation = ""
    # For anomaly detection:
    if result['anomaly_prediction'] == 0:
        anomaly_status = "normal"
    else:
        anomaly_status = "anomalous"
    explanation += f"The model predicts that the user's condition is {anomaly_status}. "
    explanation += "Key factors influencing this decision include: "
    sorted_anomaly = sorted(result['anomaly_explanation'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    for feat, value in sorted_anomaly:
        if value < 0:
            explanation += f"a lower {feat} (which helps maintain a normal status), "
        else:
            explanation += f"a higher {feat} (which raises the anomaly risk), "
    explanation = explanation.strip().rstrip(",") + ".\n"
    
    # For health score prediction:
    explanation += f"The predicted health score is {result['health_score_prediction']:.1f}. "
    explanation += "Important factors affecting this score include: "
    sorted_health = sorted(result['health_score_explanation'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    for feat, value in sorted_health:
        if value > 0:
            explanation += f"a higher {feat} (which improves the score), "
        else:
            explanation += f"a lower {feat} (which may lower the score), "
    explanation = explanation.strip().rstrip(",") + "."
    return explanation

# Generate and print the human-readable explanation
human_explanation = generate_human_explanation(result)
print("\nHuman-Readable Explanation:")
print(human_explanation)


# -------------------------------------------------
# SHAP Visualizations with Processed Data for Consistency
# -------------------------------------------------

def display_shap_explainability():
    # Obtain feature names after preprocessing
    feature_names = numeric_features + list(
        preprocessor.named_transformers_['cat']
        .named_steps['onehot'].get_feature_names_out(categorical_features)
    )
    
    # Transform the test set using the preprocessor
    processed_X_test = preprocessor.transform(X_test)
    
    # ---------------------------
    # Anomaly Detection Model Explainability
    # ---------------------------
    shap_values_anomaly = explainer_anomaly.shap_values(processed_X_test)
    
    print("\nSHAP Summary Plot for Anomaly Detection Model:")
    shap.summary_plot(shap_values_anomaly, processed_X_test, feature_names=feature_names, show=True)
    
    instance_index = 0
    # Create a DataFrame for the processed features for the instance
    instance_features = pd.DataFrame(processed_X_test[instance_index].reshape(1, -1), columns=feature_names)
    
    print("\nForce Plot for Anomaly Detection Model (Instance 0):")
    force_plot_anomaly = shap.force_plot(
        explainer_anomaly.expected_value, 
        shap_values_anomaly[instance_index], 
        features=instance_features,
        feature_names=feature_names,
        matplotlib=True
    )
    plt.show()
    
    # ---------------------------
    # Health Score Prediction Model Explainability
    # ---------------------------
    shap_values_health = explainer_health.shap_values(processed_X_test)
    
    print("\nSHAP Summary Plot for Health Score Prediction Model:")
    shap.summary_plot(shap_values_health, processed_X_test, feature_names=feature_names, show=True)
    
    print("\nForce Plot for Health Score Prediction Model (Instance 0):")
    force_plot_health = shap.force_plot(
        explainer_health.expected_value, 
        shap_values_health[instance_index], 
        features=instance_features,
        feature_names=feature_names,
        matplotlib=True
    )
    plt.show()

# Call the explainability function to generate SHAP plots
display_shap_explainability()
