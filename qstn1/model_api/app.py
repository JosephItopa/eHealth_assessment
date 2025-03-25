from sklearn.preprocessing import StandardScaler
import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load(r'qstn1\model_api\random_forest_model.pkl')

# Load or initialize a scaler (assuming the model was trained with standardized inputs)
scaler = StandardScaler()

# Streamlit UI
st.title('Covid Health Condition Prediction')

# Input fields
features = []
labels = [
    'Chest pain', 'Cough', 'Diarrhea', 'Fatigue or general weakness',
    'Fever', 'Headache', 'Thorax (sore throat)', 'Nausea', 'Runny nose',
    'Sore throat or pharyngitis', 'Vomiting', 'Loss of Taste',
    'Loss of Smell'
]

for label in labels:
    value = st.radio(f'{label}:', ['No', 'Yes'])
    features.append(1 if value == 'Yes' else 0)

# Age input
age = st.number_input('Age', min_value=1, step=1)
features.append(age)

# Sex input
sex = st.radio('Sex:', ['Male', 'Female'])
features.append(1 if sex == 'Female' else 0)  # Sex_Female
features.append(1 if sex == 'Male' else 0)  # Sex_Male

# Convert input to numpy array and standardize
if st.button('Predict'):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array) #
    probability = model.predict_proba(features_array)[:, 1]  # Probability of default
    print(probability)
    result = int(round(prediction[0],1))
    print(result)
    if probability >= 0.6:
        st.write('Prediction: You are Covid infected')
    else:
        st.write('Prediction: You are Covid free')