
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426 0%, #0D2038 55%, #102B40 100%);
    color: #F1F5F9;
}
.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
    padding-bottom: 3rem;
}
h1, h2, h3, p, label {
    color: #F1F5F9 !important;
}
.hero {
    background: linear-gradient(120deg, #122A43, #163C50);
    padding: 30px;
    border: 1px solid #285168;
    border-radius: 22px;
    margin-bottom: 25px;
}
.hero-label {
    color: #49DCC2;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 3px;
}
.hero-title {
    font-size: 34px;
    font-weight: 800;
    margin-top: 8px;
}
.hero-subtitle {
    color: #B8C9D9;
    font-size: 16px;
    margin-top: 8px;
}
.panel {
    background: #10243A;
    border: 1px solid #294158;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 18px;
}
.metric-card {
    background: #10243A;
    border: 1px solid #294158;
    border-radius: 16px;
    padding: 20px;
    min-height: 120px;
}
.metric-label {
    color: #A9BDCE;
    font-size: 13px;
}
.metric-value {
    color: #F8FAFC;
    font-size: 27px;
    font-weight: 750;
    margin-top: 9px;
}
div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(90deg, #16BFA6, #34D9C0);
    color: #071426;
    border: none;
    border-radius: 12px;
    min-height: 48px;
    font-weight: 800;
    width: 100%;
}
div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #6BEBD6;
    color: #071426;
    border: none;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #172F47;
    border-color: #34516A;
}
hr {
    border-color: #294158;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-label">CUSTOMER ANALYTICS / AI</div>
    <div class="hero-title">Churn Intelligence</div>
    <div class="hero-subtitle">
        An intelligent dashboard for predicting customer retention
        and identifying potential churn risk.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("### Customer Profile")
st.write("Enter the customer's information, then run the prediction.")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Personal Details")
        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"]
        )
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )
        age = st.slider("Customer Age", 18, 100, 35)

    with col2:
        st.markdown("#### Financial Profile")
        credit_score = st.number_input(
            "Credit Score", 300, 900, 650
        )
        balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )
        salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=60000.0,
            step=1000.0
        )

    with col3:
        st.markdown("#### Banking Activity")
        tenure = st.slider("Tenure (Years)", 0, 10, 5)
        products = st.selectbox(
            "Number of Products", [1, 2, 3, 4]
        )
        has_card = st.selectbox(
            "Has Credit Card",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        active = st.selectbox(
            "Active Member",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    submitted = st.form_submit_button("✦  Predict Customer Churn")

if submitted:
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [products],
        "HasCrCard": [has_card],
        "IsActiveMember": [active],
        "EstimatedSalary": [salary]
    })

    input_data["AgeGroup"] = pd.cut(
        input_data["Age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=False
    )

    input_data["ProductsRisk"] = (
        input_data["NumOfProducts"] >= 3
    ).astype(int)

    input_data["Inactive"] = (
        input_data["IsActiveMember"] == 0
    ).astype(int)

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    churn_probability = float(probabilities[1]) * 100
    stay_probability = float(probabilities[0]) * 100

    st.markdown("---")
    st.markdown("## Prediction Analytics")
    st.write("Results generated by the trained Random Forest model.")

    result_col, chart_col = st.columns([1, 1.25], gap="large")

    with result_col:
        if prediction == 1:
            st.markdown("""
            <div class="panel">
                <div style="color:#FF8C8C;font-size:13px;
                font-weight:700;letter-spacing:2px;">
                    CHURN RISK DETECTED
                </div>
                <h2 style="color:#FF8C8C !important;">
                    Potential Churn
                </h2>
                <p style="color:#B8C9D9 !important;">
                    The model predicts that this customer may leave
                    the bank.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="panel">
                <div style="color:#49DCC2;font-size:13px;
                font-weight:700;letter-spacing:2px;">
                    RETENTION OUTLOOK
                </div>
                <h2 style="color:#49DCC2 !important;">
                    Likely to Stay
                </h2>
                <p style="color:#B8C9D9 !important;">
                    The model predicts that this customer may
                    remain with the bank.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CHURN PROBABILITY</div>
            <div class="metric-value">{churn_probability:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RETENTION PROBABILITY</div>
            <div class="metric-value">{stay_probability:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with chart_col:
        fig = go.Figure(data=[go.Pie(
            labels=["Likely to Stay", "Potential Churn"],
            values=[stay_probability, churn_probability],
            hole=0.72,
            sort=False,
            marker=dict(
                colors=["#28C7AE", "#FF7185"],
                line=dict(color="#10243A", width=5)
            ),
            textinfo="label+percent",
            textfont=dict(color="#F1F5F9", size=13),
            hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>"
        )])

        fig.update_layout(
            title=dict(
                text="Customer Churn Probability",
                font=dict(color="#F1F5F9", size=19),
                x=0.05
            ),
            annotations=[dict(
                text=f"{churn_probability:.1f}%",
                x=0.5,
                y=0.5,
                font=dict(size=29, color="#F1F5F9"),
                showarrow=False
            )],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F1F5F9"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.12,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=70, b=50, l=15, r=15),
            height=390
        )

        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Probabilities reflect the trained model's estimates, "
        "not a guarantee of future customer behavior."
    )

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#91A8BB;'>"
    "CHURN INTELLIGENCE · MACHINE LEARNING PROJECT"
    "</div>",
    unsafe_allow_html=True
)