import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..'
        )
    )
)

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import io

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

from database.database import (
    insert_prediction,
    fetch_history
)

from src.map_visualization import (
    create_delivery_map,
    display_map
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="Last Mile Delivery Prediction",

    page_icon="🚚",

    layout="wide"

)

# ============================================================
# CREATE REPORTS FOLDER
# ============================================================

os.makedirs("../reports", exist_ok=True)

# ============================================================
# LIGHT THEME CSS
# ============================================================

st.markdown("""
<style>

html, body, .stApp {

    background: linear-gradient(
        135deg,
        #f8fbff,
        #dbeafe,
        #eff6ff,
        #ffffff
    ) !important;

    background-attachment: fixed;
}

.main-title {

    font-size: 55px;

    font-weight: bold;

    text-align: center;

    color: #0f172a;

    margin-top: 10px;
}

.sub-title {

    text-align: center;

    font-size: 22px;

    color: #334155;

    margin-bottom: 25px;
}

section[data-testid="stSidebar"] {

    background: white;

    border-right: 2px solid #93c5fd;
}

.stButton > button {

    width: 100%;

    background: linear-gradient(
        90deg,
        #2563eb,
        #38bdf8
    );

    color: white;

    font-size: 20px;

    font-weight: bold;

    border-radius: 12px;

    border: none;

    padding: 14px;
}

div[data-testid="metric-container"] {

    background: white;

    border-radius: 18px;

    padding: 15px;

    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
}

div[data-baseweb="select"] > div {

    background-color: white !important;

    color: black !important;
}

div[data-baseweb="select"] span {

    color: black !important;
}

li[role="option"] {

    color: black !important;

    background-color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

model = pickle.load(
    open("../models/delivery_failure_model.pkl", "rb")
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚚 Last Mile Delivery Failure Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI-Powered Delivery Intelligence Dashboard</div>',
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.title("📋 Delivery Inputs")

store_latitude = st.sidebar.number_input(
    "Store Latitude",
    value=12.9716
)

store_longitude = st.sidebar.number_input(
    "Store Longitude",
    value=77.5946
)

drop_latitude = st.sidebar.number_input(
    "Drop Latitude",
    value=12.9352
)

drop_longitude = st.sidebar.number_input(
    "Drop Longitude",
    value=77.6245
)

weather = st.sidebar.selectbox(
    "Weather",
    ["Sunny", "Cloudy", "Fog", "Rainy", "Stormy"]
)

traffic = st.sidebar.selectbox(
    "Traffic",
    ["Low", "Medium", "High"]
)

vehicle = st.sidebar.selectbox(
    "Vehicle",
    ["Bike", "Scooter", "Car"]
)

area = st.sidebar.selectbox(
    "Area",
    ["Urban", "Semi-Urban", "Metropolitan"]
)

category = st.sidebar.selectbox(
    "Category",
    ["Food", "Electronics", "Clothing", "Grocery"]
)

agent_age = st.sidebar.slider(
    "Agent Age",
    18,
    50,
    25
)

agent_rating = st.sidebar.slider(
    "Agent Rating",
    1.0,
    5.0,
    4.0
)

day_of_week = st.sidebar.slider(
    "Day Of Week",
    0,
    6,
    2
)

month = st.sidebar.slider(
    "Month",
    1,
    12,
    5
)

is_weekend = st.sidebar.selectbox(
    "Weekend",
    [0, 1]
)

distance_km = st.sidebar.slider(
    "Distance (KM)",
    1.0,
    50.0,
    10.0
)

# ============================================================
# ENCODING MAPS
# ============================================================

weather_map = {
    "Sunny": 0,
    "Cloudy": 1,
    "Fog": 2,
    "Rainy": 3,
    "Stormy": 4
}

traffic_map = {
    "Low": 2,
    "Medium": 1,
    "High": 0
}

vehicle_map = {
    "Bike": 0,
    "Scooter": 1,
    "Car": 2
}

area_map = {
    "Urban": 0,
    "Semi-Urban": 1,
    "Metropolitan": 2
}

category_map = {
    "Food": 0,
    "Electronics": 1,
    "Clothing": 2,
    "Grocery": 3
}

# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.button(
    "🚀 Predict Delivery Status"
)

# ============================================================
# PREDICTION SECTION
# ============================================================

if predict_button:

    input_data = pd.DataFrame([[

        store_latitude,
        store_longitude,

        drop_latitude,
        drop_longitude,

        weather_map[weather],
        traffic_map[traffic],
        vehicle_map[vehicle],
        area_map[area],
        category_map[category],

        agent_age,
        agent_rating,

        day_of_week,
        month,
        is_weekend,

        distance_km

    ]], columns=[

        'Store_Latitude',
        'Store_Longitude',

        'Drop_Latitude',
        'Drop_Longitude',

        'Weather',
        'Traffic',
        'Vehicle',
        'Area',
        'Category',

        'Agent_Age',
        'Agent_Rating',

        'day_of_week',
        'month',
        'is_weekend',

        'distance_km'

    ])

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    success_prob = probability[0][0]

    failure_prob = probability[0][1]

    prediction_text = (

        "Failure"

        if prediction[0] == 1

        else "Success"

    )

    # ========================================================
    # SAVE SESSION STATE
    # ========================================================

    st.session_state["prediction_done"] = True

    st.session_state["success_prob"] = success_prob

    st.session_state["failure_prob"] = failure_prob

    st.session_state["prediction_text"] = prediction_text

    st.session_state["input_data"] = input_data

    # ========================================================
    # SAVE TO DATABASE
    # ========================================================

    insert_prediction(

        weather,
        traffic,
        distance_km,
        prediction_text,
        float(failure_prob)

    )

# ============================================================
# SHOW RESULTS PERMANENTLY
# ============================================================

if st.session_state.get("prediction_done"):

    success_prob = st.session_state["success_prob"]

    failure_prob = st.session_state["failure_prob"]

    prediction_text = st.session_state["prediction_text"]

    input_data = st.session_state["input_data"]

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "✅ Success Probability",
            f"{success_prob:.2%}"
        )

    with col2:

        st.metric(
            "❌ Failure Probability",
            f"{failure_prob:.2%}"
        )

    if prediction_text == "Failure":

        st.error(
            "❌ High Risk: Delivery Failure Predicted"
        )

    else:

        st.success(
            "✅ Delivery Success Predicted"
        )

    # ========================================================
    # FAILURE METER
    # ========================================================

    st.subheader("📉 Failure Risk Meter")

    st.progress(float(failure_prob))

    # ========================================================
    # INPUT DATA
    # ========================================================

    st.subheader("📋 Input Data Summary")

    st.dataframe(
        input_data,
        use_container_width=True
    )

    # ========================================================
    # ANALYTICS DASHBOARD
    # ========================================================

    st.subheader("📊 Delivery Analytics Dashboard")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:

        pie_df = pd.DataFrame({

            "Status": ["Success", "Failure"],

            "Probability": [
                success_prob,
                failure_prob
            ]

        })
        fig_pie = px.pie(

            pie_df,

            names="Status",

            values="Probability",

            title="Delivery Prediction Distribution",

            color="Status",

            color_discrete_map={

                "Success": "#2563eb",

                "Failure": "#60a5fa"

            }

        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    with col_chart2:

        factor_df = pd.DataFrame({

            "Factors": [
                "Weather",
                "Traffic",
                "Distance"
            ],

            "Impact": [

                weather_map[weather] + 1,

                traffic_map[traffic] + 1,

                distance_km / 10

            ]

        })

        fig_bar = px.bar(

            factor_df,

            x="Factors",

            y="Impact",

            title="Risk Factor Analysis",

            color="Factors",

            color_discrete_sequence=[

                "#2563eb",

                "#3b82f6",

                "#60a5fa"

            ]

        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    # ========================================================
    # GAUGE CHART
    # ========================================================

    st.subheader("🎯 Delivery Failure Risk Score")

    fig_gauge = go.Figure(go.Indicator(

        mode="gauge+number",

        value=failure_prob * 100,

        title={'text': "Failure Risk %"},

        gauge={

            'axis': {

                'range': [0, 100]
            },

            'bar': {

                'color': "#2563eb"
            },

            'bgcolor': "#dbeafe",

            'borderwidth': 2,

            'bordercolor': "#1d4ed8"

        }

    ))
    st.plotly_chart(
        fig_gauge,
        use_container_width=True
    )

    # ========================================================
    # SAVE CHART IMAGES
    # ========================================================

    fig_pie.write_image(
        "../reports/pie_chart.png"
    )

    fig_bar.write_image(
        "../reports/bar_chart.png"
    )

    fig_gauge.write_image(
        "../reports/gauge_chart.png"
    )

    # ========================================================
    # DELIVERY MAP
    # ========================================================

    st.subheader("🗺️ Delivery Route Visualization")

    delivery_map = create_delivery_map(

        store_latitude,
        store_longitude,

        drop_latitude,
        drop_longitude,

        prediction_text

    )

    display_map(delivery_map)

    # ========================================================
    # PDF REPORT
    # ========================================================

    def generate_pdf_report():

        pdf_path = "../reports/delivery_report.pdf"

        doc = SimpleDocTemplate(

            pdf_path,

            pagesize=letter

        )

        styles = getSampleStyleSheet()

        elements = []

        elements.append(

            Paragraph(

                "Last Mile Delivery Prediction Report",

                styles['Title']

            )

        )

        elements.append(Spacer(1, 20))

        elements.append(

            Paragraph(

                f"Prediction Result: {prediction_text}",

                styles['Heading2']

            )

        )

        elements.append(Spacer(1, 20))

        elements.append(

            Paragraph(

                f"Success Probability: {success_prob:.2%}",

                styles['BodyText']

            )

        )

        elements.append(

            Paragraph(

                f"Failure Probability: {failure_prob:.2%}",

                styles['BodyText']

            )

        )

        elements.append(Spacer(1, 20))

        elements.append(Image(

            "../reports/pie_chart.png",

            width=400,

            height=250

        ))

        elements.append(Spacer(1, 20))

        elements.append(Image(

            "../reports/bar_chart.png",

            width=400,

            height=250

        ))

        elements.append(Spacer(1, 20))

        elements.append(Image(

            "../reports/gauge_chart.png",

            width=400,

            height=250

        ))

        doc.build(elements)

        return pdf_path

    pdf_path = generate_pdf_report()

    with open(pdf_path, "rb") as pdf_file:

        st.download_button(

            label="📄 Download PDF Report",

            data=pdf_file,

            file_name="delivery_report.pdf",

            mime="application/pdf"

        )

# ============================================================
# PREDICTION HISTORY
# ============================================================

st.divider()

st.subheader("📜 Prediction History")

history = fetch_history()

if history:

    history_df = pd.DataFrame(

        history,

        columns=[

            "ID",
            "Weather",
            "Traffic",
            "Distance",
            "Prediction",
            "Probability",
            "Created At"

        ]

    )

    st.dataframe(

        history_df,

        use_container_width=True

    )

else:

    st.info("No prediction history available")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <center>
        <h4>🚚 Last Mile Delivery Intelligence Platform</h4>
        <p>Powered by Machine Learning, MySQL, Plotly & FastAPI</p>
    </center>
    """,
    unsafe_allow_html=True
)