import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide"
)

# -------------------------------------------------
# GLOBAL DARK + BLUE UI STYLES
# -------------------------------------------------
st.markdown("""
<style>

/* Force blue Analyze button */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #2563eb, #1e40af) !important;
    color: #ffffff !important;
    height: 2.5rem !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}

/* Remove default focus outline */
div.stButton > button:first-child:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.6) !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODELS & FEATURE SCHEMA
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    clf = joblib.load("models/classification_model.pkl")
    reg = joblib.load("models/regression_model.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")
    return clf, reg, feature_cols

clf_model, reg_model, feature_columns = load_artifacts()

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/india_housing_prices.csv")

df = load_data()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    """
    <h1 style="margin-bottom:0;">🏡 Real Estate Investment Advisor</h1>
    <p style="color:gray;margin-top:4px;">
        This tool helps you check property investment quality and future price estimation.
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# TABS (LIKE REFERENCE APP)
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["⚡ Quick Predictor", "🔎 Property Search", "📊 Market Insights", "ℹ️ About & Skills"]
)

# =================================================
# TAB 1 — QUICK INVESTMENT PREDICTOR
# =================================================
with tab1:

    st.subheader("⚡ Quick Investment Predictor")

    # -------- ROW 1 --------
    c1, c2, c3 = st.columns(3)

    with c1:
        state = st.selectbox("📍 State", sorted(df["State"].unique()))
        city = st.selectbox(
            "🏙️ City",
            sorted(df[df["State"] == state]["City"].unique())
        )

    with c2:
        property_type = st.selectbox(
            "🏠 Property Type",
            sorted(df[
                (df["State"] == state) &
                (df["City"] == city)
            ]["Property_Type"].unique())
        )
        bhk = st.number_input("🛏️ BHK", 1, 10, 2)

    with c3:
        price = st.number_input("💰 Current Price (₹ Lakhs)", 10.0, 10000.0, 150.0)
        size = st.number_input("📐 Size (Sq Ft)", 300, 20000, 1200)

    # -------- ROW 2 --------
    c4, c5, c6 = st.columns(3)

    with c4:
        age = 5
        st.markdown("📅 **Age of Property (Years): 5**")

    with c5:
        parking = st.selectbox("🚗 Parking Space", ["Yes", "No"])
        security = st.selectbox("🛡️ Security", ["Yes", "No"])

    with c6:
        facing = st.selectbox("🧭 Facing", df["Facing"].unique())
        availability = st.selectbox("📦 Availability", df["Availability_Status"].unique())

    # -------- ANALYZE BUTTON --------

    predict_btn = st.button("⚡ Analyze Investment", use_container_width=True)

    # -------- PREDICTION --------
    if predict_btn:

        # Create full feature row
        input_df = pd.DataFrame(columns=feature_columns)
        input_df.loc[0] = 0

        # Categorical
        input_df.at[0, "State"] = state
        input_df.at[0, "City"] = city
        input_df.at[0, "Locality"] = "Unknown"
        input_df.at[0, "Property_Type"] = property_type
        input_df.at[0, "Facing"] = facing
        input_df.at[0, "Availability_Status"] = availability
        input_df.at[0, "Parking_Space"] = parking
        input_df.at[0, "Security"] = security
        input_df.at[0, "Furnished_Status"] = "Unfurnished"
        input_df.at[0, "Owner_Type"] = "Owner"
        input_df.at[0, "Amenities"] = "None"

        # Numeric
        input_df.at[0, "BHK"] = bhk
        input_df.at[0, "Size_in_SqFt"] = size
        input_df.at[0, "Price_in_Lakhs"] = price
        input_df.at[0, "Age_of_Property"] = age

        # Engineered
        input_df.at[0, "Price_per_SqFt"] = (price * 100000) / size
        input_df.at[0, "Nearby_Schools"] = 0
        input_df.at[0, "Nearby_Hospitals"] = 0
        input_df.at[0, "School_Density_Score"] = 0
        input_df.at[0, "Hospital_Density_Score"] = 0
        input_df.at[0, "Public_Transport_Accessibility"] = 1
        input_df.at[0, "Floor_No"] = 1
        input_df.at[0, "Total_Floors"] = 1
        input_df.at[0, "Year_Built"] = 2020
        input_df.at[0, "ID"] = 0

        # Predict
        invest = clf_model.predict(input_df)[0]
        confidence = clf_model.predict_proba(input_df)[0][1]
        future_price = reg_model.predict(input_df)[0]

        st.markdown("---")
        st.subheader("📊 Investment Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            if invest == 1:
                st.success("✅ Good Investment")
            else:
                st.error("❌ Not a Good Investment")

        with r2:
            st.metric(" Model Confidence", f"{confidence*100:.2f}%")

        with r3:
            st.metric("Estimated Price (5Y)", f"₹ {future_price:.2f} Lakhs")

# =================================================
# TAB 2 — PROPERTY SEARCH (SIMPLE)
# =================================================
with tab2:
    st.subheader("🔎 Property Search")
    st.dataframe(df.head(50), use_container_width=True)

# =================================================
# TAB 3 — MARKET INSIGHTS
# =================================================
with tab3:
    st.subheader("📊 Market Insights")

    # ----------------------------------
    # 1. City-wise Average Price (Bar)
    # ----------------------------------
    st.markdown("### 🏙️ Average Property Price by City")

    city_price = (
        df.groupby("City", as_index=False)["Price_in_Lakhs"]
        .mean()
        .sort_values("Price_in_Lakhs", ascending=False)
        .head(10)
    )

    fig1 = px.bar(
        city_price,
        x="Price_in_Lakhs",
        y="City",
        orientation="h",
        color="Price_in_Lakhs",
        color_continuous_scale="Blues",
        labels={"Price_in_Lakhs": "Avg Price (Lakhs)"},
        title="Top 10 Cities by Average Property Price"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ----------------------------------
    # 2. BHK vs Price (Box Plot)
    # ----------------------------------
    st.markdown("### 🛏️ Price Distribution by BHK")

    fig2 = px.box(
        df,
        x="BHK",
        y="Price_in_Lakhs",
        color="BHK",
        labels={"Price_in_Lakhs": "Price (Lakhs)"},
        title="Price Spread Across BHK Types"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------------
    # 3. Property Type vs Price (Donut)
    # ----------------------------------
    st.markdown("### 🏠 Average Price by Property Type")

    prop_price = (
        df.groupby("Property_Type", as_index=False)["Price_in_Lakhs"]
        .mean()
    )

    fig3 = px.pie(
        prop_price,
        values="Price_in_Lakhs",
        names="Property_Type",
        hole=0.4,
        title="Property Type Price Contribution"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ----------------------------------
    # 4. Size vs Price (Scatter)
    # ----------------------------------
    st.markdown("### 📐 Property Size vs Price")

    fig4 = px.scatter(
        df,
        x="Size_in_SqFt",
        y="Price_in_Lakhs",
        color="BHK",
        size="BHK",
        opacity=0.6,
        labels={
            "Size_in_SqFt": "Size (Sq Ft)",
            "Price_in_Lakhs": "Price (Lakhs)"
        },
        title="Correlation Between Size and Price"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ----------------------------------
    # 5. Age vs Price Trend (Line)
    # ----------------------------------
    st.markdown("### 📅 Property Age vs Price Trend")

    age_price = (
        df.groupby("Age_of_Property", as_index=False)["Price_in_Lakhs"]
        .mean()
        .sort_values("Age_of_Property")
    )

    fig5 = px.line(
        age_price,
        x="Age_of_Property",
        y="Price_in_Lakhs",
        markers=True,
        labels={
            "Age_of_Property": "Age (Years)",
            "Price_in_Lakhs": "Avg Price (Lakhs)"
        },
        title="Price Trend Based on Property Age"
    )
    st.plotly_chart(fig5, use_container_width=True)

    # ----------------------------------
    # 6. City-wise Price Distribution (Violin)
    # ----------------------------------
    st.markdown("### 🎻 Price Distribution Across Major Cities")

    top_cities = df["City"].value_counts().head(5).index
    filtered_df = df[df["City"].isin(top_cities)]

    fig6 = px.violin(
        filtered_df,
        x="City",
        y="Price_in_Lakhs",
        box=True,
        points="all",
        color="City",
        title="Price Distribution in Major Cities"
    )
    st.plotly_chart(fig6, use_container_width=True)

# =================================================
# TAB 4 — ABOUT
# =================================================
with tab4:
    st.subheader("ℹ️ About & Skills")
    st.write("""
    **Real Estate Investment Advisor**

    - Machine Learning: Logistic Regression, XGBoost
    - Data Processing & Feature Engineering
    - ML Pipelines & Model Evaluation
    - Streamlit App Development
    - End-to-End Deployment Ready
    """)
# ---------------------------------------
# FOOTER
# ---------------------------------------
st.markdown("-------")
st.caption("© 2025 • Built with Streamlit & Machine Learning")

st.caption("Developed by Vadla Shiva Kumar")
