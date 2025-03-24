import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv(r"/datasets/Solar Power.csv")

# Preparing the data for modeling
X = df.drop(columns=['power-generated'])  # Update target variable name
y = df['power-generated']  # Update target variable name

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)

# Standardizing the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions with Random Forest
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"Mean Absolute Error on Test Set (Random Forest): {mae_rf}")

# Hyperparameter tuning for Random Forest
params_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 5]
}
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), params_rf, cv=5, verbose=0, n_jobs=-1, scoring='neg_mean_squared_error')
grid_rf.fit(X_train, y_train)

# Predictions with the best Random Forest model
y_pred_rf_best = grid_rf.predict(X_test)
mae_rf_best = mean_absolute_error(y_test, y_pred_rf_best)
print(f"Mean Absolute Error on Test Set (Optimized Random Forest): {mae_rf_best}")

# Save the best model
best_model = grid_rf.best_estimator_

def predict_solar_power(input_features):
    # Create DataFrame from input
    input_df = pd.DataFrame([input_features])
    
    # Scale features
    scaled_features = sc.transform(input_df)
    
    # Make prediction
    prediction = best_model.predict(scaled_features)
    
    return f"Predicted Solar Power: {prediction[0]}"

# Example usage - Update input features according to your dataset's columns.
sample_input = {
    'distance-to-solar-noon': 0.07,
    'temperature': 72,
    'wind-direction': 29,
    'wind-speed': 7.5,
    'sky-cover': 0,
    'visibility': 10.0,
    'humidity': 67,
    'average-wind-speed-(period)': 6.0,
    'average-pressure-(period)': 29.85,
}

print("\n" + predict_solar_power(sample_input))

model_filename = '/models/SolarPower_model.pkl'
with open(model_filename, 'wb') as model_pkl:
    pickle.dump({
        'model': best_model,
        'scaler': sc,
        'feature_order': X.columns.tolist()
    }, model_pkl)  # Save the best model along with necessary preprocessing steps

print("Solar power prediction model saved successfully.")
