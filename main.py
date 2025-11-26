
import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI 

st.title("AI Retail Insight Agent")

uploaded = st.file_uploader("Upload your retail data", type=["csv", "xlsx"])

if uploaded:
    df = pd.read_csv(uploaded) if uploaded.name.endswith("csv") else pd.read_excel(uploaded)
    st.subheader("Preview")
    st.write(df.head())

    # Basic metrics
    st.subheader("Key Sales Signals")
    st.write("Top Selling Products:", df.groupby("product")["sales"].sum().sort_values(ascending=False).head())
    st.write("Fastest Growing:", df.groupby("product")["sales"].sum().pct_change().nlargest(5))

    # AI Insight Engine
    client = OpenAI()
    prompt = f"""
    You are a Retail AI Analyst.
    Analyse this dataset and tell me:
    - Demand trends
    - Stockout risk
    - Winning attributes
    - Failing attributes
    - Assortment opportunities
    - Pricing signals
    - 3 immediate actions
    Data:
    {df.head(50).to_string()}
    """
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    st.subheader("AI Retail Insights")
    st.write(res.choices[0].message["content"])