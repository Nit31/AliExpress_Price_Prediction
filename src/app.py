import streamlit as st
import mlflow
import json
import requests
import pandas as pd
from data import preprocess_data_from_web

PORT_NUMBER = 5001

def predict(title, rating, launchDate, category, sold, discount, shippingCost, storeName, type):
    features = {
        "id": 0,
        "storeId": 0,
        "storeName": storeName,
        "title": title,
        "rating": rating,
        "lunchTime": str(launchDate),
        "category_name": category,
        "postCategory": 0,
        "sold": sold,
        "discount": discount,
        "shippingCost": shippingCost,
        "type": type
    }
    raw_df = pd.DataFrame(features, index=[0])
    X = preprocess_data_from_web(raw_df)
    example = X.iloc[0, :]
    example = json.dumps({"inputs": example.to_dict()})
    
    print(X.iloc[0, :])
    response = requests.post(
        url=f"http://localhost:{PORT_NUMBER}/invocations",
        data=example,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return "Error in prediction"

# Streamlit app setup
st.title("Product Prediction App")

title = st.text_input("Title of the product:")
rating = st.number_input("Rating of the product on other platforms:")
launchDate = st.date_input("The product is on sale since:")
category = st.text_input("Category of product:")
sold = st.number_input("Amount of product sold:")
discount = st.number_input("Discount(%) that you can suggest:")
shippingCost = st.number_input("Shipping cost for this product:")
storeName = st.text_input("Store Name:")
type = st.selectbox("Is the product well advertised?", ["No", "Yes"])

if st.button("Get Prediction"):
    prediction_result = predict(title, rating, launchDate, category, sold, discount, shippingCost, storeName, type)
    st.write("Prediction result:", prediction_result)
