import streamlit as st
from pathlib import Path
from tensorflow.keras.models import load_model
from keras import backend as K
import torch.nn as nn
import pandas as pd

from helping_functions import (
    get_facebook_url,
    get_facebook_followers,
    get_likes_followers,
    get_wait,
    text_to_num,
)


st.set_page_config(
    page_title="Omdena-Voy-Finance",
    page_icon=":moneybag:",
    initial_sidebar_state="auto",
)
st.header("Omdena: Voy Finance 💰", anchor=False)
st.subheader("Deterministic Digital Score (DDS)", anchor=False)
st.write(
    "The following application will help you assign a Digital score to a company, if you provide the company's name."
)

# AVG = 951938.0811764706
# frequency_ratio = 2.09 / 3.05

name = st.text_input("Enter the company name 👇")
df = pd.read_csv("assets/csv/Digital_score_linkedin.csv")

if name:
    try:
        digital_score = df[df["Company"] == name]["digital_score"].values[0]
        ws_rank = df[df["Company"] == name]["ws_rank"].values[0]
        st.write("Digital Score: ", digital_score)
        st.write(
            "The digital score reflects how large a company's active social media following compared to an average fortune 1000 company. Here ",
            name,
            "has ",
            digital_score,
            " times more active followers that the average fortune 1000 company",
        )
        st.write("WS Rank: ", ws_rank)
        st.write(
            "WS Rank indicates the quality of the company's offical webiste. This includes webiste latency, usability and keywords"
        )

    except Exception as e:
        digital_score = 0
        ws_rank = 0
        st.write("Digital Score: ", digital_score)
        st.write("WS Rank: ", ws_rank)
        st.write("The requested company is not present in our Database.")
        st.write("Please try again.")
