import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import pickle

# Load the dataset
crop = pd.read_csv(r"/datasets/crop_recommendation.csv")

# Map crop labels to numbers
crop_dict = { 
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
    'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15,
    'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19,
    'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}
crop['crop_num'] = crop['label'].map(crop_dict)

# Prepare features and target variable
X = crop.drop(['label', 'crop_num'], axis=1)
Y = crop['crop_num']

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_scaled, Y_train)

# Predictions
y_pred = nb_classifier.predict(X_test_scaled)

# Calculate evaluation metrics
precision = precision_score(Y_test, y_pred, average='weighted')
recall = recall_score(Y_test, y_pred, average='weighted')
f1 = f1_score(Y_test, y_pred, average='weighted')
accuracy = accuracy_score(Y_test, y_pred)

# Display evaluation metrics
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print(f'Accuracy: {accuracy}')

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, y_pred)

# Function to recommend a crop based on input features
def crop_recommend(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    transformed_features = scaler.transform(features)  # Use the same scaler
    prediction = nb_classifier.predict(transformed_features).reshape(1, -1)  # Use the same classifier
    
    # Map prediction back to crop name
    crop_name = [key for key, value in crop_dict.items() if value == prediction[0]][0]
    
    return f"{crop_name} is the best crop to be cultivated."

# Example usage of the crop_recommend function
recommended_crop = crop_recommend(90, 45, 45, 20, 80,6.0, 200)
print(recommended_crop)

# Save model, scaler, and crop dictionary
model_filename = '/models/crop_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump({
        'model': nb_classifier,
        'scaler': scaler,
        'crop_dict': crop_dict,
        'feature_names': X.columns.tolist()
    }, f)
