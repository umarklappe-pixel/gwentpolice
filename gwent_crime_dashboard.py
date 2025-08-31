import io
import os
import typing as t
from datetime import datetime

import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Page config
# -------------------------

st.set_page_config(
    page_title="Gwent Police — Analytics Dashboard",
    page_icon="",
    layout="wide"
)

st.title("EDA — Predictive Analytics Dashboard")
st.caption("EDA • Predictive Modeling • AI and ML")

# -------------------------
# Helper functions
# -------------------------
@st.cache_data(show_spinner=False)
def load_csv(file: t.Union[str, io.BytesIO]) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    if "month" in df.columns:
        try:
            df["month"] = pd.to_datetime(df["month"], errors="coerce")
        except Exception:
            pass
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "crime_type" in df.columns:
        df = df[~df["crime_type"].isna()]
    if "month" in df.columns:
        df["year_month"] = df["month"].dt.to_period("M").astype(str)
    for col in ["lsoa_name", "lsoa_code", "location", "reported_by", "falls_within", "last_outcome_category"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def pick_top_categories(series: pd.Series, top_n: int = 30) -> pd.Series:
    counts = series.value_counts(dropna=False)
    top = counts.head(top_n).index
    return series.where(series.isin(top), other="Other")


def make_confusion_df(y_true, y_pred, labels) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=pd.Index(labels, name="True"), columns=pd.Index(labels, name="Pred"))



# -------------------------
# Sidebar — Data input
# -------------------------

st.sidebar.header("Gwent police Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Police.UK street-level CSV")

# Optional sample path
default_path = "example.csv"
use_sample = st.sidebar.toggle("Use example file name in current folder", value=False)

df = None
if uploaded_file is not None:
    df = load_csv(uploaded_file)
elif use_sample and os.path.exists(default_path):
    df = load_csv(default_path)
else:
    st.info("Upload a CSV in the sidebar to begin.")
    st.stop()

st.success(f"Loaded {len(df):,} rows • {df.shape[1]} columns")

# -------------------------
# Dataset Stats
# -------------------------

# Total crimes = total rows
total_crimes = len(df)

# Total months (if available)
total_months = df["year_month"].nunique() if "year_month" in df.columns else "N/A"

# Drop 'context' column from missing check if it exists
df_check = df.drop(columns=["context"], errors="ignore")

# Find rows with NaN or empty values (excluding 'context')
incomplete_rows = df_check[
    df_check.isna().any(axis=1) | (df_check.astype(str).apply(lambda x: x.str.strip() == "").any(axis=1))
]

incomplete_count = len(incomplete_rows)

# Show summary stats
col1, col2, col3 = st.columns(3)
col1.metric("Total Crimes", f"{total_crimes:,}")
col2.metric("Total Months", f"{total_months}")
col3.metric("Incomplete Rows", f"{incomplete_count:,}")


# -------------------------
# EDA (always full data)
# -------------------------
st.header("Exploratory Data Analysis (EDA)")

colA, colB = st.columns(2)
with colA:
    if "crime_type" in df.columns:
        st.subheader("Crimes by Type")
        s = df["crime_type"].value_counts().reset_index()
        s.columns = ["crime_type", "count"]
        chart = alt.Chart(s).mark_bar().encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("crime_type:N", sort="-x", title="Crime Type"),
            tooltip=["crime_type", "count"]
        )
        st.altair_chart(chart, use_container_width=True)

with colB:
    if "year_month" in df.columns:
        st.subheader("Trend by Month")
        ts = df.groupby("year_month").size().reset_index(name="count")
        ts["year_month"] = pd.to_datetime(ts["year_month"])
        line = alt.Chart(ts).mark_line(point=True).encode(
            x=alt.X("year_month:T", title="Month"),
            y=alt.Y("count:Q", title="Crimes"),
            tooltip=["year_month:T", "count:Q"]
        )
        st.altair_chart(line, use_container_width=True)

colC, colD = st.columns(2)
with colC:
    if "lsoa_name" in df.columns:
        st.subheader("Top Crime Locations")
        top_lsoa = df["lsoa_name"].value_counts().head(20).reset_index()
        top_lsoa.columns = ["lsoa_name", "count"]
        bar = alt.Chart(top_lsoa).mark_bar().encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("lsoa_name:N", sort="-x", title="LSOA"),
            tooltip=["lsoa_name", "count"]
        )
        st.altair_chart(bar, use_container_width=True)

with colD:
    if {"latitude", "longitude"}.issubset(df.columns):
        st.subheader("Crime Map (sample upto 5,000 pts)")
        map_df = df[["latitude", "longitude"]].dropna().sample(min(5000, len(df)), random_state=42)
        st.map(map_df.rename(columns={"latitude":"lat", "longitude":"lon"}))


st.subheader("Heatmap — Top 10 Crime Types by Month")

if {"month", "crime_type"}.issubset(df.columns):
    # Ensure datetime and extract year-month properly
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["year_month"] = df["month"].dt.to_period("M").astype(str)

    # Aggregate counts
    crime_month = (
        df.groupby(["crime_type", "year_month"])
          .size()
          .reset_index(name="count")
    )

    # Keep only top 10 crime types overall
    top10_types = df["crime_type"].value_counts().head(10).index
    crime_month = crime_month[crime_month["crime_type"].isin(top10_types)]
    
    # Heatmap
    heatmap = alt.Chart(crime_month).mark_rect().encode(
        x=alt.X("year_month:N", title="Month",
                sort=sorted(crime_month["year_month"].unique())),
        y=alt.Y("crime_type:N", title="Crime Type"),
        color=alt.Color("count:Q", title="Crimes", scale=alt.Scale(scheme="reds")),
        tooltip=["crime_type", "year_month", "count:Q"]
    )

    st.altair_chart(heatmap, use_container_width=True)
else:
    st.info("Columns 'month' and 'crime_type' are required for this chart.")

# -------------------------
# Predictive Modeling
# -------------------------

st.header("Predictive Model")

if "year_month" in df.columns and "crime_type" in df.columns:
    st.subheader("History Data (Trends — Top 6 Crime Types)")

    # Aggregate monthly counts by crime type
    ts = (
        df.groupby(["year_month", "crime_type"])
          .size()
          .reset_index(name="count")
    )
    ts["year_month"] = pd.to_datetime(ts["year_month"], errors="coerce")

    # Pick top 6 crime types overall
    top6_types = df["crime_type"].value_counts().head(6).index
    ts_top6 = ts[ts["crime_type"].isin(top6_types)]

    # Multi-line chart
    line = alt.Chart(ts_top6).mark_line(point=True).encode(
        x=alt.X("year_month:T", title="Month"),
        y=alt.Y("count:Q", title="Crimes"),
        color=alt.Color("crime_type:N", title="Crime Type"),
        tooltip=["year_month:T", "crime_type", "count:Q"]
    ).properties(height=400)

    st.altair_chart(line, use_container_width=True)
else:
    st.info("Columns 'year_month' and 'crime_type' are required for this chart.")


# Month selection for training
if "year_month" not in df.columns:
    st.warning("No 'month' column available for training restriction.")
    st.stop()

all_months = sorted(df["year_month"].unique())
default_months = sorted(all_months)[-6:]

selected_months = st.multiselect("Select up to 6 months for training", options=all_months, default=default_months)
if len(selected_months) > 6:
    st.error("Please select a maximum of 6 months.")
    st.stop()

# Target selection
possible_targets = [c for c in ["crime_type", "last_outcome_category"] if c in df.columns]
if not possible_targets:
    st.warning("No suitable target column found.")
    st.stop()

target_col = st.selectbox("Choose target to predict", options=possible_targets, index=0)

candidate_features = [c for c in ["lsoa_name", "location", "reported_by", "falls_within", "year_month"] if c in df.columns and c != target_col]
if {"latitude", "longitude"}.issubset(df.columns):
    candidate_features += ["latitude", "longitude"]

selected_features = st.multiselect("Select features", options=candidate_features, default=candidate_features[:5])
if not selected_features:
    st.warning("Select at least one feature to train the model.")
    st.stop()

model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"], index=1)

# Button to start training
if st.button("Start Training"):
    model_df = df[df["year_month"].isin(selected_months)].dropna(subset=[target_col]).copy()

    # Handle missing values in features
    for col in selected_features:
        if model_df[col].dtype == "object":
            model_df[col] = model_df[col].fillna("Unknown")
            model_df[col] = pick_top_categories(model_df[col], top_n=40)
        else:
            model_df[col] = model_df[col].fillna(model_df[col].median())

    X = model_df[selected_features].copy()
    y = model_df[target_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    cat_cols = [c for c in selected_features if X[c].dtype == "object"]
    num_cols = [c for c in selected_features if c not in cat_cols]

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(with_mean=False), num_cols)
    ])

    if model_choice == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000)
    else:
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

    pipe = Pipeline(steps=[("prep", preprocess), ("model", clf)])

    with st.spinner("Training model..."):
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    labels = sorted(y.unique().tolist())
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Accuracy", f"{acc:.3f}")
    mcol2.metric("Macro F1", f"{f1m:.3f}")
    mcol3.metric("Classes", f"{len(labels)}")

    st.subheader("Confusion Matrix")
    cm_df = make_confusion_df(y_test, y_pred, labels)
    cm_chart = px.imshow(cm_df.values, x=labels, y=labels, labels=dict(x="Predicted", y="True", color="Count"))
    st.plotly_chart(cm_chart, use_container_width=True)

    with st.expander("Classification Report", expanded=False):
        st.text(classification_report(y_test, y_pred, zero_division=0))
