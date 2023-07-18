
import streamlit as st
import pandas as pd

import pickle

df = pd.read_csv(r'C:\Users\HP\Desktop\Project -Tags\Bengaluru_house_prediction\cleaned_dataset.csv')
X=pd.read_csv(r'C:\Users\HP\Downloads\X.csv')

# Load the model from a pickle file
with open(r'Model_new.pkl', 'rb') as file:
    model = pickle.load(file)

import streamlit as st

st.markdown('<h1 style="color: #FF5733;">Bengaluru House Predictor</h1>', unsafe_allow_html=True)


location = st.selectbox('Select Location', sorted(df['location'].unique().tolist()))
bhk = st.number_input("Enter number of BHK", min_value=1, max_value=10, step=1)
bathroom = st.number_input("Enter number of bathroom", min_value=1, max_value=4, step=1)
btn1 = st.button('Predict Price')



#
if btn1:
    # Prepare the input data for prediction
    input_data = pd.DataFrame([[location, bhk, bathroom]], columns=['location', 'bhk', 'bath'])
    input_data_encoded = pd.get_dummies(input_data, columns=['location'], drop_first=True)

    missing_cols = set(X.columns) - set(input_data_encoded.columns)
    for col in missing_cols:
        input_data_encoded[col] = 0
    input_encoded = input_data_encoded[X.columns]

    input_encoded = input_encoded.drop("Unnamed: 0", axis=1)

    # Make the prediction
    prediction = model.predict(input_encoded)[0]


    # Display the prediction
    st.success("The Price of flat is Rs {:.2f}".format(prediction))























