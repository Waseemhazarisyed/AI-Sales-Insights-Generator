📊 AI-Powered Sales Insights Generator

An end-to-end data analytics project that analyzes retail sales transactions, builds interactive dashboards, and uses AI to generate business insights.

This project is designed for non-technical stakeholders, making sales data easy to understand through automated KPIs, visualizations, and AI-driven explanations.

📂 Dataset Information

The original dataset is larger than 25MB, so it cannot be uploaded directly to GitHub.

You can download the dataset here:

👉 Dataset Download Link:
https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset


After downloading, place the file inside:

data/online_sales.csv


A smaller sample dataset can also be added if needed for quick testing.

🚀 Project Overview

This project demonstrates a full analytics workflow:

✔️ Data Loading

Reads and imports raw sales data.

✔️ Data Cleaning

Handles:

Missing values

Inconsistent column names

Incorrect data types

Date parsing

✔️ Feature Engineering

Creates:

quantity

revenue

year_month

✔️ Aggregation & KPI Extraction

Calculates:

Total Revenue

Total Transactions

Average Order Value

Total Items Sold

Top Products

Top Cities

Monthly Revenue Trend

✔️ Visualizations

Interactive charts using Plotly:

Monthly Revenue Trend

Top Products

Top Cities

✔️ AI Insights

Uses OpenAI to automatically generate:

Key insights

Risks

Opportunities

Actionable recommendations

✔️ Interactive Dashboard (Streamlit)

Users can:

Filter by city

View updated KPIs

Interact with charts

Generate AI insights

📁 Project Structure
AI_insights_sales_generator/
│
├── data/
│   └── online_sales.csv          # Download and place here
│
├── dashboard.py                  # Streamlit App
├── AI_Sales_Insights.ipynb       # Notebook analysis
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation

🛠️ Technologies Used

Python

Pandas

Plotly

Streamlit

OpenAI API

Virtual Environment (.venv)

🧠 AI-Generated Insights (Example)

The dashboard uses the OpenAI API to summarize:

Key sales trends

Risks & anomalies

Opportunities for growth

Business recommendations

This helps non-technical teams understand data instantly.

▶️ How to Run the Project
1. Clone or download the project
2. Create & activate a virtual environment:
python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies:
pip install -r requirements.txt

4. Add your OpenAI API key:
export OPENAI_API_KEY="your_key_here"

5. Run the dashboard:
python -m streamlit run dashboard.py

📈 Results

This project enables business teams to:

Understand sales performance clearly

Identify product and regional trends

Spot anomalies or risks

Make data-driven decisions with AI-powered insights

👤 Author

Waseem Hazari Syed
Entry-Level Data Analyst
Passionate about analytics, visualization, and building AI-powered tools.