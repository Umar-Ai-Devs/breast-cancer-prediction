import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Predictions for accuracy
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("🔬 Breast Cancer Prediction App")
st.write("This app predicts whether a tumor is **Malignant** (dangerous) or **Benign** (not dangerous).")

# Show model accuracy
st.subheader("📊 Model Performance")
st.write(f"**Model Accuracy:** {accuracy:.2f}")

# Show confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=data.target_names, yticklabels=data.target_names, ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# User input
st.subheader("🧑‍⚕️ Enter Patient Details")

mean_radius = st.number_input("Mean Radius", min_value=0.0, max_value=30.0, step=0.1)
mean_texture = st.number_input("Mean Texture", min_value=0.0, max_value=40.0, step=0.1)
mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0, max_value=1.0, step=0.001)

if st.button("Predict Cancer Type"):
    input_data = np.zeros((1, X.shape[1]))
    input_data[0, X.columns.get_loc("mean radius")] = mean_radius
    input_data[0, X.columns.get_loc("mean texture")] = mean_texture
    input_data[0, X.columns.get_loc("mean smoothness")] = mean_smoothness
    
    prediction = model.predict(input_data)[0]
    result = data.target_names[prediction]
    
    st.success(f"Prediction: The tumor is **{result.upper()}**")
