
import streamlit as st
import pandas as pd
import numpy as np
import os
from openai import OpenAI

st.set_page_config(page_title="AI Retail Demand Insight Engine", layout="wide")

st.title("AI Retail Demand Insight Engine")
st.caption("Demand signals, stockout risk, winning products, weak products and immediate retail actions.")

# -----------------------------
# Synthetic Data Generator
# -----------------------------
@st.cache_data
def generate_synthetic_data(rows=1200):
    np.random.seed(42)

    dates = pd.date_range(end=pd.Timestamp.today(), periods=180)

    regions = ["North", "South", "West", "East"]
    stores = [
        "Gurgaon Store", "Delhi Store", "Mumbai Store", "Pune Store",
        "Bangalore Store", "Chennai Store", "Hyderabad Store", "Kolkata Store"
    ]

    categories = {
        "Fashion": ["Cotton Kurta", "Denim Jacket", "Linen Shirt", "Formal Trouser"],
        "Beauty": ["Lipstick", "Face Serum", "Sunscreen", "Compact Powder"],
        "Home": ["Storage Box", "Cookware Set", "Dinner Plate", "Glass Tumbler"],
        "Grocery": ["Cold Pressed Juice", "Organic Rice", "Breakfast Cereal", "Green Tea"],
        "Footwear": ["Comfort Sandal", "Formal Shoe", "Sneaker", "Block Heel"]
    }

    data = []

    for _ in range(rows):
        category = np.random.choice(list(categories.keys()))
        product = np.random.choice(categories[category])

        price = np.random.randint(299, 4999)
        discount = np.random.choice([0, 5, 10, 15, 20, 30], p=[0.25, 0.20, 0.20, 0.15, 0.12, 0.08])

        base_demand = np.random.randint(5, 80)

        # Product behaviour logic
        if product in ["Cold Pressed Juice", "Sunscreen", "Comfort Sandal", "Storage Box"]:
            demand_multiplier = np.random.uniform(1.3, 2.2)
        elif product in ["Denim Jacket", "Formal Trouser", "Compact Powder"]:
            demand_multiplier = np.random.uniform(0.5, 0.9)
        else:
            demand_multiplier = np.random.uniform(0.8, 1.3)

        units_sold = int(base_demand * demand_multiplier)
        sales = units_sold * price * (1 - discount / 100)

        stock_on_hand = np.random.randint(10, 500)
        margin_pct = np.random.randint(22, 68)

        stock_cover_days = round(stock_on_hand / max(units_sold, 1) * 7, 1)

        if stock_cover_days < 10:
            stock_status = "Stockout Risk"
        elif stock_cover_days > 60:
            stock_status = "Overstock Risk"
        else:
            stock_status = "Healthy"

        data.append({
            "date": np.random.choice(dates),
            "region": np.random.choice(regions),
            "store": np.random.choice(stores),
            "category": category,
            "product": product,
            "sku": f"{category[:3].upper()}-{np.random.randint(1000, 9999)}",
            "price": price,
            "discount_pct": discount,
            "units_sold": units_sold,
            "sales": round(sales, 2),
            "stock_on_hand": stock_on_hand,
            "stock_cover_days": stock_cover_days,
            "margin_pct": margin_pct,
            "stock_status": stock_status,
            "channel": np.random.choice(["Store", "Online", "Marketplace"]),
            "customer_segment": np.random.choice(["Value Buyer", "Premium Buyer", "Repeat Buyer", "New Buyer"])
        })

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


# -----------------------------
# Upload or Default Synthetic Data
# -----------------------------
uploaded = st.file_uploader("Upload your retail data", type=["csv", "xlsx"])

if uploaded:
    df = pd.read_csv(uploaded) if uploaded.name.endswith("csv") else pd.read_excel(uploaded)
    st.success("Your uploaded data is being used.")
else:
    df = generate_synthetic_data()
    st.info("Demo mode active. Synthetic retail data is loaded automatically.")

# -----------------------------
# Data Cleaning
# -----------------------------
df.columns = df.columns.str.lower().str.strip()

required_cols = ["product", "sales"]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

if "region" in df.columns:
    selected_region = st.sidebar.multiselect(
        "Region",
        sorted(df["region"].dropna().unique()),
        default=sorted(df["region"].dropna().unique())
    )
    df = df[df["region"].isin(selected_region)]

if "category" in df.columns:
    selected_category = st.sidebar.multiselect(
        "Category",
        sorted(df["category"].dropna().unique()),
        default=sorted(df["category"].dropna().unique())
    )
    df = df[df["category"].isin(selected_category)]

if "channel" in df.columns:
    selected_channel = st.sidebar.multiselect(
        "Channel",
        sorted(df["channel"].dropna().unique()),
        default=sorted(df["channel"].dropna().unique())
    )
    df = df[df["channel"].isin(selected_channel)]

# -----------------------------
# KPI Cards
# -----------------------------
st.subheader("Business Snapshot")

total_sales = df["sales"].sum()
total_units = df["units_sold"].sum() if "units_sold" in df.columns else 0
avg_margin = df["margin_pct"].mean() if "margin_pct" in df.columns else 0
stockout_count = (df["stock_status"] == "Stockout Risk").sum() if "stock_status" in df.columns else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales", f"₹{total_sales:,.0f}")
col2.metric("Units Sold", f"{total_units:,.0f}")
col3.metric("Average Margin", f"{avg_margin:.1f}%")
col4.metric("Stockout Risk SKUs", f"{stockout_count}")

# -----------------------------
# Preview
# -----------------------------
with st.expander("View Data Preview"):
    st.dataframe(df.head(30), use_container_width=True)

# -----------------------------
# Key Sales Signals
# -----------------------------
st.subheader("Key Sales Signals")

col1, col2 = st.columns(2)

with col1:
    st.write("Top Selling Products")
    top_products = df.groupby("product")["sales"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_products)

with col2:
    if "category" in df.columns:
        st.write("Category Sales")
        category_sales = df.groupby("category")["sales"].sum().sort_values(ascending=False)
        st.bar_chart(category_sales)

# -----------------------------
# Demand Trend
# -----------------------------
if "date" in df.columns:
    st.subheader("Demand Trend")

    daily_sales = df.groupby(df["date"].dt.date)["sales"].sum()
    st.line_chart(daily_sales)

# -----------------------------
# Stock Risk
# -----------------------------
if "stock_status" in df.columns:
    st.subheader("Stock Risk View")

    risk_summary = df["stock_status"].value_counts()
    st.bar_chart(risk_summary)

    stockout_products = df[df["stock_status"] == "Stockout Risk"]

    with st.expander("Products at Stockout Risk"):
        st.dataframe(
            stockout_products[
                ["store", "category", "product", "sku", "stock_on_hand", "units_sold", "stock_cover_days", "sales"]
            ].sort_values("stock_cover_days").head(50),
            use_container_width=True
        )

# -----------------------------
# Winners and Weak Products
# -----------------------------
st.subheader("Winners and Weak Products")

product_summary = df.groupby("product").agg(
    sales=("sales", "sum"),
    units_sold=("units_sold", "sum") if "units_sold" in df.columns else ("sales", "count"),
    avg_margin=("margin_pct", "mean") if "margin_pct" in df.columns else ("sales", "mean"),
    avg_discount=("discount_pct", "mean") if "discount_pct" in df.columns else ("sales", "mean"),
    avg_stock_cover=("stock_cover_days", "mean") if "stock_cover_days" in df.columns else ("sales", "mean")
).reset_index()

winning_products = product_summary.sort_values(["sales", "avg_margin"], ascending=False).head(5)
weak_products = product_summary.sort_values(["sales", "avg_stock_cover"], ascending=[True, False]).head(5)

col1, col2 = st.columns(2)

with col1:
    st.write("Winning Products")
    st.dataframe(winning_products, use_container_width=True)

with col2:
    st.write("Weak Products")
    st.dataframe(weak_products, use_container_width=True)

# -----------------------------
# Rule-Based Retail Insights
# -----------------------------
def generate_rule_based_insights(df):
    top_product = df.groupby("product")["sales"].sum().idxmax()
    top_sales = df.groupby("product")["sales"].sum().max()

    slow_product = df.groupby("product")["sales"].sum().idxmin()

    insights = []

    insights.append(f"Highest demand is coming from {top_product}, contributing ₹{top_sales:,.0f} in sales.")
    insights.append(f"{slow_product} is the weakest demand product and should be reviewed for markdown, repositioning or reduced buying.")

    if "stock_status" in df.columns:
        stockout_count = (df["stock_status"] == "Stockout Risk").sum()
        overstock_count = (df["stock_status"] == "Overstock Risk").sum()

        insights.append(f"{stockout_count} SKUs are showing stockout risk. These need urgent replenishment.")
        insights.append(f"{overstock_count} SKUs are showing overstock risk. These need markdown or redistribution action.")

    if "discount_pct" in df.columns:
        high_discount_sales = df[df["discount_pct"] >= 20]["sales"].sum()
        insights.append(f"High-discount products are generating ₹{high_discount_sales:,.0f}. Check whether discounting is driving real demand or only margin leakage.")

    insights.append("Immediate actions:")
    insights.append("1. Replenish high-selling products with less than 10 days of stock cover.")
    insights.append("2. Mark down products with high stock cover and low sales.")
    insights.append("3. Shift stock from low-demand stores to high-demand stores before fresh buying.")

    return "\n\n".join(insights)

st.subheader("Retail Demand Insights")
st.write(generate_rule_based_insights(df))

# -----------------------------
# AI Insight Engine
# -----------------------------
st.subheader("AI Retail Analyst")

use_ai = st.toggle("Generate OpenAI-powered insights", value=False)

if use_ai:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.warning("OpenAI API key not found. Add your key in environment variables or Streamlit secrets.")
    else:
        client = OpenAI(api_key=api_key)

        prompt = f"""
        You are a Retail AI Analyst.

        Analyse this retail dataset and give clear business insights under these headings:

        1. Demand trends
        2. Stockout risk
        3. Winning products
        4. Weak products
        5. Assortment opportunities
        6. Pricing signals
        7. Three immediate actions

        Keep the answer practical for a retail business owner.

        Data sample:
        {df.head(80).to_string()}
        """

        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            st.write(res.choices[0].message.content)

        except Exception as e:
            st.error(f"AI insight generation failed: {e}")
