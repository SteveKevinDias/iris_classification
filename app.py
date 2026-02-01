import pickle
import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import joblib
from pymongo.mongo_client import MongoClient


client = MongoClient(st.secrets["MONGO_URI"])

db = client['iris_db_classification']
collection = db['iris_data']

def load_model(model_name):
    model,scaler = joblib.load(model_name + ".pkl")
    return model,scaler

def preprocessing_input_data(input_data, scaler):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    return input_scaled
    
def predict(model, input_scaled):
    prediction = model.predict(input_scaled)
    return prediction

def classify(outcome):
    if outcome == 1:
        return "setosa"
    elif outcome == 2:
        return "versinicolor"
    else:
        return "virginica"


def main():
    st.title("Iris Flower Prediction App")

    st.write("Select a model and input the features to get the prediction.")

    model_name = st.selectbox("Select Model", ["logistic_model_binary", "logistic_model_multinomial", "logistic_model_ovr","svm_model_binary","svm_model_multi"])

    input_data = {}
    input_data['sepal length (cm)'] = st.number_input("Enter the Sepal Length (cm)", value=0.0)
    input_data['sepal width (cm)'] = st.number_input("Enter the Sepal Width (cm)", value=0.0)
    input_data['petal length (cm)'] = st.number_input("Enter the Petal Length (cm)", value=0.0)
    input_data['petal width (cm)'] = st.number_input("Enter the Petal Width (cm)", value=0.0)

    


    if st.button("Predict"):
        model, scaler = load_model(model_name)
        input_scaled = preprocessing_input_data(input_data, scaler)
        prediction = predict(model, input_scaled)
        input_data['predicted_class'] = classify(prediction[0])
        collection.insert_one(input_data)
        st.success(f"The Prediction of the flower type based on the model that was is selected is : {classify(prediction[0]).capitalize()}")

if __name__ == "__main__":
    main()