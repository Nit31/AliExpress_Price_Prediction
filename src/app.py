import streamlit as st
import json
import requests
import pandas as pd
import hydra
from hydra.core.global_hydra import GlobalHydra
from data import preprocess_data

PORT_NUMBER = 5001


def predict(title, rating, launchDate, category, sold, discount, shippingCost, type):

    type = "ad" if type == "Yes" else "natural"
    features = {
        "title": title,
        "rating": rating,
        "lunchTime": str(launchDate),
        "category_name": category,
        "sold": sold,
        "discount": discount,
        "shippingCost": shippingCost,
        "type": type,
    }
    raw_df = pd.DataFrame(features, index=[0])
    X = preprocess_data(raw_df, skip_target=True)[0]

    example = X.iloc[0, :]
    example = json.dumps({"inputs": example.to_dict()})

    if GlobalHydra.instance().is_initialized():
        print("Using existing Hydra global instance.")
        cfg = hydra.compose(config_name="main")
    else:
        print("Initializing a new Hydra global instance.")
        hydra.initialize(
            config_path="../configs", job_name="streamlit", version_base=None
        )
        cfg = hydra.compose(config_name="main")
    print(cfg)
    response = requests.post(
        url=f"http://localhost:{cfg.api_port}/predict",
        data=example,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return "Error in prediction"


# Streamlit app setup
st.title("AliExpress Price Prediction App")

title = st.text_input("Title of the product:")
rating = st.number_input("Rating of the product on other platforms:")
launchDate = st.date_input("The product is on sale since:")
category = st.text_input("Category of product:")
sold = st.number_input("Amount of product sold:")
discount = st.number_input("Discount(%) that you can suggest:")
shippingCost = st.number_input("Shipping cost for this product:")
type = st.selectbox("Is the product well advertised?", ["No", "Yes"])

if st.button("Get Prediction"):
    if not title:
        st.error("Title of the product cannot be empty")
    elif not category:
        st.error("Category of product cannot be empty")
    elif rating < 0.0 or rating > 5.0:
        st.error("Rating must be between 0 and 5")
    elif discount < 0.0 or discount > 100.0:
        st.error("Discount must be between 0 and 100")
    elif sold < 0:
        st.error("Amount of product sold must be non-negative")
    elif shippingCost < 0.0:
        st.error("Shipping cost must be non-negative")
    else:
        prediction_result = predict(
            title, rating, launchDate, category, sold, discount, shippingCost, type
        )
        st.write("Predicted product price:", prediction_result, "dirham")
