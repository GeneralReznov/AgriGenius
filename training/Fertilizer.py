import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv(r"/datasets/Fertilizer Prediction.csv")

# Prepare data for modeling
X = df.drop(columns=['Fertilizer Name'])
y = df['Fertilizer Name']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)

# Identify categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Apply Label Encoding to categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# Standardize the numeric features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Hyperparameter tuning using GridSearchCV
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 6, 7],
    'min_samples_split': [2, 5, 8]
}
grid_rand = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=3, verbose=0, n_jobs=-1)
grid_rand.fit(X_train, y_train)

# Predictions with the best model
pred_rand = grid_rand.predict(X_test)

# Classification report
report = classification_report(y_test, pred_rand, output_dict=True)

# Best parameters and score
best_score = grid_rand.best_score_
best_params = grid_rand.best_params_
accuracy_rand = accuracy_score(y_test, pred_rand)

# Print results
#print("Confusion Matrix:\n", cm)
print("Initial Model Accuracy:", accuracy)
print("\nOptimized Model Metrics:")
#print("Classification Report:\n", classification_report(y_test, pred_rand))
print('Best Validation Score:', best_score)
print('Best Parameters:', best_params)
print(f"Optimized Model Accuracy: {accuracy_rand}")

def recommend_fertilizer(input_features):
    
    # Create DataFrame from input
    input_df = pd.DataFrame([input_features])
    
    # Encode categorical features
    for col in categorical_cols:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])
    
    # Scale features
    scaled_features = sc.transform(input_df)
    
    # Make prediction
    prediction = grid_rand.best_estimator_.predict(scaled_features)
    
    return f"Recommended Fertilizer: {prediction[0]}"

# Example usage
sample_input = {
    'Temperature': 29,
    'Humidity': 58,
    'Moisture': 57,
    'Soil Type': 'Black',
    'Crop Type': 'Sugarcane',
    'Nitrogen': 12,
    'Potassium': 0,
    'Phosphorous': 10
}

print("\n" + recommend_fertilizer(sample_input))

# Save the complete model package
fertilizer_model_filename = '/models/fertilizer_model.pkl'
with open(fertilizer_model_filename, 'wb') as f:
    pickle.dump({
        'model': grid_rand.best_estimator_,
        'scaler': sc,
        'label_encoders': label_encoders,
        'categorical_cols': categorical_cols,
        'feature_order': X.columns.tolist()
    }, f)

