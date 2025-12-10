import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
client = OpenAI()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("AI Retail Insight Agent")

uploaded = st.file_uploader("Upload your retail data", type=["csv", "xlsx"])

if uploaded:
    df = pd.read_csv(uploaded) if uploaded.name.endswith("csv") else pd.read_excel(uploaded)

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    st.subheader("Preview")
    st.write(df.head())

    # ---------------------------------------------
    # 1. Sell Through Analysis
    # ---------------------------------------------
    st.subheader("Sell Through Analysis")

    if {"sales_units", "stock", "product"}.issubset(df.columns):
        df["sell_through"] = df["sales_units"] / (df["sales_units"] + df["stock"]).replace(0, np.nan)
        sell_through_product = df.groupby("product")["sell_through"].mean().sort_values(ascending=False).head(10)
        sell_through_category = df.groupby("category")["sell_through"].mean().sort_values(ascending=False)
        sell_through_channel = df.groupby("channel")["sell_through"].mean().sort_values(ascending=False)

        st.write("By Product", sell_through_product)
        st.write("By Category", sell_through_category)
        st.write("By Channel", sell_through_channel)
    else:
        st.warning("Sell through requires sales_units, stock, product.")

    # ---------------------------------------------
    # 2. Stock Cover & Replenishment
    # ---------------------------------------------
    st.subheader("Stock Cover & Replenishment")

    if {"sales_units", "stock", "product"}.issubset(df.columns):
        avg_sales = df.groupby("product")["sales_units"].mean()
        stock = df.groupby("product")["stock"].sum()
        stock_cover = stock / avg_sales.replace(0, np.nan)

        replenishment_qty = (avg_sales * 4) - stock  # target 4 weeks of cover
        replenishment_qty = replenishment_qty[replenishment_qty > 0]

        st.write("Weeks of Cover", stock_cover.sort_values())
        st.write("Recommended Replenishment Quantity", replenishment_qty.sort_values(ascending=False).head(20))
    else:
        st.warning("Stock cover requires sales_units, stock, product.")

    # ---------------------------------------------
    # 3. Discount Dependency
    # ---------------------------------------------
    st.subheader("Discount Dependency")

    if {"discount_percent", "sales_units", "product"}.issubset(df.columns):
        discount_dependency = df.groupby("product")[["discount_percent", "sales_units"]].corr().iloc[0::2, -1]
        discount_dependency.name = "correlation"

        high_dependency = discount_dependency.sort_values(ascending=False).head(10)
        red_flags = df[df["discount_percent"] > 50].groupby("product")["sales_units"].sum().sort_values(ascending=False)

        st.write("Discount Impact on Sales Units (Correlation)", high_dependency)
        st.write("High Discount Red Flags", red_flags.head(10))
    else:
        st.warning("Discount dependency requires discount_percent, sales_units, product.")

    # ---------------------------------------------
    # 4. Profitability Signals
    # ---------------------------------------------
    st.subheader("Profitability Signals")

    if {"mrp", "final_price", "sales_units", "product"}.issubset(df.columns):
        df["margin_per_unit"] = df["mrp"] - df["final_price"]
        df["total_margin"] = df["margin_per_unit"] * df["sales_units"]

        margin_contribution = df.groupby("product")["total_margin"].sum().sort_values(ascending=False).head(15)

        st.write("Margin Contribution by Product", margin_contribution)
    else:
        st.warning("Profitability requires mrp, final_price, sales_units, product.")

    # ---------------------------------------------
    # 5. Stockout Loss Estimation
    # ---------------------------------------------
    st.subheader("Stockout Loss Estimation")

    if {"stockout", "sales_units", "product"}.issubset(df.columns):
        potential_loss = df[df["stockout"] == 1].groupby("product")["sales_units"].sum().sort_values(ascending=False)

        st.write("Estimated Units Lost Due To Stockout", potential_loss.head(15))
    else:
        st.warning("Stockout estimation requires stockout, sales_units, product.")

    # ---------------------------------------------
    # 6. Price Sensitivity (Elasticity Signals)
    # ---------------------------------------------
    st.subheader("Price Sensitivity Analysis")

    if {"final_price", "sales_units", "product"}.issubset(df.columns):
        elasticity = df.groupby("product")[["final_price", "sales_units"]].corr().iloc[0::2, -1]
        elasticity.name = "elasticity"

        st.write("Price Elasticity by Product", elasticity.sort_values().head(15))
    else:
        st.warning("Elasticity requires final_price, sales_units, product.")

    # ---------------------------------------------
    # 7. Category Winners vs Losers
    # ---------------------------------------------
    st.subheader("Category Winners and Losers")

    if {"category", "sales_units", "week", "product"}.issubset(df.columns):
        weekly = df.groupby(["product", "week"])["sales_units"].sum()
        growth = weekly.groupby(level=0).pct_change()

        winners = growth.groupby(level=0).last().nlargest(10)
        losers = growth.groupby(level=0).last().nsmallest(10)
        volume_winners = df.groupby("product")["sales_units"].sum().sort_values(ascending=False).head(10)

        st.write("Top Growth Products", winners)
        st.write("Bottom Growth Products", losers)
        st.write("Top Volume Products", volume_winners)
    else:
        st.warning("Winners and losers requires product, category, sales_units, week.")

    # ---------------------------------------------
    # 8. Cluster Insights
    # ---------------------------------------------
    st.subheader("Cluster Insights")

    if {"state", "city", "locality", "sales_units"}.issubset(df.columns):
        region_sales = df.groupby(["state", "city"])["sales_units"].sum().sort_values(ascending=False).head(20)
        locality_sales = df.groupby("locality")["sales_units"].sum().sort_values(ascending=False).head(20)

        st.write("Top Region Clusters", region_sales)
        st.write("Top Locality Clusters", locality_sales)
    else:
        st.warning("Cluster insights require state, city, locality, sales_units.")

    # ---------------------------------------------
    # AI Retail Analyst
    # ---------------------------------------------

    prompt = f"""
    You are a senior Retail AI Analyst.

    Interpret the full analytics output and generate executive retail insights covering:
    - Demand drivers
    - Stockout risk pattern
    - Discount efficiency
    - Category strategy
    - Price sensitivity pattern
    - Inventory risk levels
    - Top replenishment priorities
    - High potential clusters
    - 5 actions for the category head

    Data:
    {df.head(50).to_string()}
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    st.subheader("AI Retail Insights")
    st.write(res.choices[0].message["content"])






