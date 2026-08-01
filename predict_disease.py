import joblib

# Load model
model = joblib.load("heart_disease_model.pkl")

# Sample patient data
sample = [[52, 1, 1, 134, 201, 0, 1, 158, 0, 0.8, 2, 1, 2]]

# Predict
prediction = model.predict(sample)

if prediction[0] == 1:
    print("Prediction: Heart Disease Detected")
else:
    print("Prediction: No Heart Disease")