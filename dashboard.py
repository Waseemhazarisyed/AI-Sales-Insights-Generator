
import os
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI-Powered Sales Insights",
    layout="wide"
)

DATA_PATH = "data/online_sales.csv"


# --------- DATA LOADING & CLEANING ---------
@st.cache_data
def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardize columns
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Convert date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        st.error("No 'date' column found in the dataset.")
        st.stop()

    # Numeric conversions
    for col in ["total_items", "total_cost"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create quantity & revenue
    if "total_items" in df.columns:
        df["quantity"] = df["total_items"]
    else:
        st.error("No 'total_items' column found to create quantity.")
        st.stop()

    if "total_cost" in df.columns:
        df["revenue"] = df["total_cost"]
    else:
        st.error("No 'total_cost' column found to create revenue.")
        st.stop()

    # Drop invalid rows
    df = df.dropna(subset=["date", "quantity", "revenue"])

    # Year-month for grouping
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    return df


df_clean = load_and_clean_data(DATA_PATH)

# --------- KPI CALCULATION ---------
total_revenue = df_clean["revenue"].sum()
total_transactions = df_clean.shape[0]
avg_order_value = df_clean["revenue"].mean()
total_items_sold = df_clean["quantity"].sum()

revenue_by_product = (
    df_clean.groupby("product")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

revenue_by_city = None
if "city" in df_clean.columns:
    revenue_by_city = (
        df_clean.groupby("city")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

revenue_by_month = (
    df_clean.groupby("year_month")["revenue"]
    .sum()
    .sort_index()
)


# --------- KPI SUMMARY TEXT FOR AI ---------
def build_kpi_summary_text():
    lines = []
    lines.append("=== High-Level Sales Summary ===")
    lines.append(f"Total revenue: ${total_revenue:,.2f}")
    lines.append(f"Total transactions: {total_transactions}")
    lines.append(f"Average order value: ${avg_order_value:,.2f}")
    lines.append(f"Total items sold: {total_items_sold:,.0f}")

    lines.append("\n=== Top 5 Products by Revenue ===")
    for product, value in revenue_by_product.items():
        lines.append(f"- {product}: ${value:,.2f}")

    if revenue_by_city is not None:
        lines.append("\n=== Top 5 Cities by Revenue ===")
        for city, value in revenue_by_city.items():
            lines.append(f"- {city}: ${value:,.2f}")

    lines.append("\n=== Monthly Revenue (chronological) ===")
    for ym, value in revenue_by_month.items():
        lines.append(f"- {ym}: ${value:,.2f}")

    return "\n".join(lines)


kpi_summary_text = build_kpi_summary_text()


# --------- LAYOUT: TITLE ---------
st.title("🧠 AI-Powered Sales Insights Dashboard")
st.caption("Interactive sales analytics with AI-generated business insights")


# --------- LAYOUT: FILTERS (SIDEBAR) ---------
with st.sidebar:
    st.header("Filters")
    if "city" in df_clean.columns:
        cities = ["All"] + sorted(df_clean["city"].unique().tolist())
        selected_city = st.selectbox("City", cities)
    else:
        selected_city = "All"

if selected_city != "All" and "city" in df_clean.columns:
    df_filtered = df_clean[df_clean["city"] == selected_city]
else:
    df_filtered = df_clean.copy()

# Recompute metrics based on filter
filtered_revenue_by_month = (
    df_filtered.groupby("year_month")["revenue"]
    .sum()
    .sort_index()
)

filtered_revenue_by_product = (
    df_filtered.groupby("product")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)


# --------- LAYOUT: KPIs ROW ---------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

kpi_col1.metric("Total Revenue", f"${total_revenue:,.0f}")
kpi_col2.metric("Total Transactions", f"{total_transactions:,}")
kpi_col3.metric("Avg Order Value", f"${avg_order_value:,.2f}")
kpi_col4.metric("Total Items Sold", f"{total_items_sold:,.0f}")

st.markdown("---")


# --------- CHARTS ROW ---------
# Monthly Revenue Trend (filtered)
fig1 = px.line(
    filtered_revenue_by_month.reset_index(),
    x="year_month",
    y="revenue",
    markers=True,
    title=f"📈 Monthly Revenue Trend ({selected_city})",
    labels={"year_month": "Month", "revenue": "Revenue ($)"}
)
fig1.update_layout(template="plotly_white")

# Top 5 Products (filtered)
fig2 = px.bar(
    filtered_revenue_by_product.reset_index(),
    x="product",
    y="revenue",
    title=f"🏆 Top 5 Products by Revenue ({selected_city})",
    labels={"product": "Product", "revenue": "Revenue ($)"},
    text="revenue"
)
fig2.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig2.update_layout(template="plotly_white")


chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.plotly_chart(fig1, use_container_width=True)
with chart_col2:
    st.plotly_chart(fig2, use_container_width=True)

if revenue_by_city is not None:
    fig3 = px.bar(
        revenue_by_city.reset_index(),
        x="city",
        y="revenue",
        title="🌍 Top 5 Cities by Revenue",
        labels={"city": "City", "revenue": "Revenue ($)"},
        text="revenue"
    )
    fig3.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig3.update_layout(template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")


# --------- AI INSIGHTS SECTION ---------
st.subheader("🤖 AI-Generated Sales Insights")

st.write(
    "Click the button below to generate a natural language analysis of the sales performance, "
    "including key insights, risks, opportunities, and recommendations."
)

if st.button("Generate AI Insights"):
    try:
        client = OpenAI()  # uses OPENAI_API_KEY from environment

        prompt = f"""
        You are a senior sales analyst.

        Based on the following sales KPI summary, provide:
        1. 5 Key Insights about sales performance
        2. 3 Potential Risks or concerns
        3. 3 Opportunities for growth
        4. 5 Actionable Recommendations for the business

        Be concise, use bullet points, and keep the language clear for non-technical stakeholders.

        Sales KPI Summary:
        {kpi_summary_text}
        """

        with st.spinner("Generating AI insights..."):
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=prompt,
            )
        ai_text = response.output[0].content[0].text
        st.markdown(ai_text)
    except Exception as e:
        st.error(f"Error generating AI insights: {e}")
        st.info("Make sure your OPENAI_API_KEY is set in your environment.")
