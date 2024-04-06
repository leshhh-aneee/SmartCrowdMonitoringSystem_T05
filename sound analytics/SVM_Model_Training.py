import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the data
data = pd.read_csv('finalVideo_svm_features_final_data_updated.csv')

# Prepare the dataset
X = data.drop(['Time (s)', 'Bus Presence'], axis=1)  # Features
y = data['Bus Presence']  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='auto')
svm_model.fit(X_train_scaled, y_train)

# Predict using the trained model
y_pred = svm_model.predict(X_test_scaled)

# Print out the results
print("Model accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler for later use
joblib.dump(svm_model, 'bus_detection_model.pkl')
joblib.dump(scaler, 'feature_scalar_file.pkl')  # Saving the scaler

# Example of loading and using the saved model and scaler
loaded_model = joblib.load('bus_detection_model.pkl')
loaded_scaler = joblib.load('feature_scalar_file.pkl')

