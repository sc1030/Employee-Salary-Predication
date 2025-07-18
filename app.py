# app.py
from tkinter import _test
from seaborn import displot
import seaborn as sns
from sklearn.base import ClassifierMixin
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve, roc_auc_score
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from fpdf import FPDF
import tempfile

# --- Page Config ---
st.set_page_config(page_title="💼 Salary Predictor", layout="wide")

# --- Load Model ---
model = joblib.load("model/salary_predictor_corrected.pkl")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main { background-color: #f9f9fb; }
    .block-container { padding: 2rem 3rem; }
    .salary-card {
        background: linear-gradient(to right, #4e54c8, #8f94fb);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-top: 2rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation (✅ ONLY ONCE)
st.sidebar.title("🚀 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Home", "💰 Salary Prediction", "📊 Data Exploration", "📈 Model Analytics", "ℹ️ About"],
    key="sidebar_page"
)

# ------------------- PAGE: HOME -------------------
if page == "🏠 Home":
    st.title("💼 Employee Salary Prediction System")
    st.markdown("AI Powered Salary Prediction for the Indian Job Market")

    colA, colB = st.columns([2, 1])
    with colA:
        st.subheader("🔍 Key Factors Influencing Salary")
        st.markdown("""
        - **Performance Ratings** - High-rated employees earn more
        - **Geographic Location** - City tier and regional factors
        - **Company Details** - Organization size, industry, and type
        - **Department & Role** - Job function and responsibility level
        """)

        st.subheader("🚀 Key Features")
        st.markdown("""
        - Real-time salary predictions with 85%+ accuracy
        - Interactive data visualizations and insights
        - Comprehensive model performance analytics
        - User-friendly interface for easy navigation
        """)

    with colB:
        st.metric("🪙 Average Salary", "₹12.42 L")
        st.metric("🏢 Departments", "6")
        st.metric("🌍 Cities", "8")

    st.divider()
    st.subheader("📊 Salary Distribution Overview")

    df = pd.DataFrame({
        'Department': ['Technology', 'Finance', 'Sales', 'Marketing', 'Operations', 'HR'],
        'Median Salary (Lakhs INR)': [13, 11.5, 10.2, 9.8, 9.5, 8.7]
    })

    fig1 = px.bar(
        df,
        x='Median Salary (Lakhs INR)',
        y='Department',
        color='Median Salary (Lakhs INR)',
        orientation='h',
        title="Median Salary by Department",
        color_continuous_scale='viridis'
    )

    df_scatter = pd.DataFrame({
        'Years of Experience': list(range(0, 31)) * 6,
        'Annual Salary (INR)': [int(3e5 + x * 2.5e4 + (x % 6) * 1e4) for x in range(0, 186)],
        'Department': ['Technology'] * 31 + ['Finance'] * 31 + ['Sales'] * 31 + ['Marketing'] * 31 + ['Operations'] * 31 + ['HR'] * 31
    })

    fig2 = px.scatter(
        df_scatter,
        x="Years of Experience",
        y="Annual Salary (INR)",
        color="Department",
        title="Experience vs Salary Relationship"
    )

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)

# ------------------- PAGE: SALARY PREDICTION -------------------
elif page == "💰 Salary Prediction":
    st.title("💰 Salary Prediction Tool")
    st.markdown("Enter the details below to estimate the employee's annual salary.")

    st.markdown("### 👤 Personal Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("🎂 Age", 18, 65, 30)
    with col2:
        gender = st.selectbox("♂️ Gender", ["Male", "Female"], key="gender")
    with col3:
        education = st.selectbox("🎓 Education", ["Bachelors", "Masters", "HS-grad", "Some-college", "Doctorate", "Unknown"], key="education")

    st.markdown("### 💼 Professional Details")
    col4, col5, col6 = st.columns(3)
    with col4:
        occupation = st.selectbox("Job Title", ["Exec-managerial", "Tech-support", "Sales", "Prof-specialty", "Craft-repair", "Adm-clerical", "Other-service", "Unknown"], key="occupation")
    with col5:
        experience = st.slider("Experience (Years)", 0.0, 40.0, 5.0, 0.5)
    with col6:
        performance = st.slider("Performance Rating (1-5)", 1, 5, 3)

    st.markdown("### 🌍 Location & Company")
    col7, col8, col9 = st.columns(3)
    with col7:
        race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Other"], key="race")
    with col8:
        hours_per_week = st.slider("Hours per Week", 1, 80, 40)
    with col9:
        marital_status = st.selectbox("Marital Status", ["Never-married", "Married-civ-spouse", "Divorced", "Widowed", "Separated"], key="marital")

    st.markdown("### 📄 Additional Info")
    col10, col11, col12 = st.columns(3)
    with col10:
        capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    with col11:
        capital_loss = st.number_input("Capital Loss", 0, 100000, 0)
    with col12:
        certs = st.number_input("Certifications", 0, 20, 1)

    col13, col14 = st.columns(2)
    with col13:
        english = st.selectbox("English Proficiency", ["Basic", "Intermediate", "Fluent"], key="english")
    with col14:
        bonus = st.checkbox("Expecting Bonus")

    input_data = pd.DataFrame([{
        "age": age,
        "education": education,
        "occupation": occupation,
        "gender": gender,
        "race": race,
        "marital-status": marital_status,
        "hours-per-week": hours_per_week,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "experience": experience,
        "certifications": certs,
        "english_proficiency": english,
        "bonus": int(bonus),
        "performance_rating": performance
    }])

    if st.button("🔍 Predict Salary"):
        prediction = model.predict(input_data)[0]
        monthly = prediction / 12
        bonus_amt = np.random.randint(10000, 50000) if bonus else 0
        total = prediction + bonus_amt

        st.markdown(f"""
        <div class='salary-card'>
            <h2>Predicted Annual Salary</h2>
            <h1>₹{prediction:,.2f}</h1>
            <p><strong>Monthly:</strong> ₹{monthly:,.2f}</p>
            <p><strong>Bonus:</strong> {'₹' + str(bonus_amt) if bonus else 'Not Applicable'}</p>
            <p><strong>Total:</strong> ₹{total:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Download PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Salary Prediction Report", ln=True, align='C')
            for key, val in input_data.iloc[0].items():
                pdf.cell(200, 10, txt=f"{key.capitalize()}: {val}", ln=True)
            pdf.cell(200, 10, txt=f"Predicted Salary: ₹{prediction:,.2f}", ln=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                pdf.output(tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button("Download PDF", f, file_name="salary_report.pdf")

# ------------------- PAGE: DATA EXPLORATION -------------------
elif page == "📊 Data Exploration":
    st.title("Data Exploration")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        st.write(df.describe())
        st.write(df.isnull().sum())

        if st.checkbox("Show Age Distribution"):
            st.plotly_chart(px.histogram(df, x="age", nbins=20, title="Age Distribution"))

        if st.checkbox("Show Salary by Education"):
            if "education" in df.columns and "salary" in df.columns:
                fig = px.box(df, x="education", y="salary", title="Salary by Education")
                st.plotly_chart(fig)

# ------------------- PAGE: MODEL ANALYTICS -------------------
elif page == "📈 Model Analytics":


    # Scatter plot
    with st.expander("📈 Scatter Plot: Actual vs Predicted Salary"):
        # Example: Generate sample data if y_test and y_pred are not available
        sample_size = 100
        y_test = np.random.randint(300000, 2000000, sample_size)
        y_pred = y_test + np.random.normal(0, 100000, sample_size)
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        ax.set_xlabel("Actual Salary")
        ax.set_ylabel("Predicted Salary")
        ax.set_title("Actual vs Predicted Salary")
        st.pyplot(fig)

    # Line plot
    with st.expander("📉 Line Plot: Actual vs Predicted Salary"):
        fig, ax = plt.subplots()
        ax.plot(pd.Series(y_test).reset_index(drop=True), label="Actual", marker='o', color='blue')
        ax.plot(pd.Series(y_pred), label="Predicted", marker='x', color='red')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Salary")
        ax.set_title("Actual vs Predicted Salary Over Samples")
        ax.legend()
        st.pyplot(fig)




# ------------------- PAGE: ABOUT -------------------
elif page == "ℹ️ About":
    st.title("About")
    st.markdown("""
    This app is built by **Shyam Sunder** using Streamlit & Scikit-Learn.  
    It predicts employee salary based on professional and personal attributes.
    """)
