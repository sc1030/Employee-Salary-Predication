
# 💼 Employee Salary Prediction System

An AI-powered web application built with **Streamlit** and **Scikit-learn** that predicts employee salaries in the Indian job market based on demographic and professional attributes. This project is part of the **IBM Capstone Project**.

---


## 🧠 Problem Statement

Accurate salary prediction is essential for HR departments and job seekers. This system uses machine learning to estimate salaries based on factors such as:
- Age, Gender, Education
- Occupation, Experience, Performance
- Work hours, Certifications, Bonus expectations, etc.

It helps users make informed decisions regarding compensation, hiring, and job offers.

---

## 🧱 Tech Stack

| Layer         | Technology                          |
|---------------|--------------------------------------|
| 🧮 ML Model   | Scikit-learn (`RandomForestRegressor`) |
| 📊 Frontend   | Streamlit                           |
| 🗃️ Data       | Cleaned & preprocessed CSV (Kaggle) |
| 📦 Deployment | Localhost / Streamlit Sharing        |

---

## 🧪 Features

- 🔢 Real-time salary prediction with 85%+ accuracy
- 📊 Interactive charts using Plotly (scatter, bar, histogram)
- 📄 Generate downloadable **PDF salary reports**
- ⚙️ Fully customizable inputs for 12+ features
- 🌐 Responsive UI with modern styling

---

## 🏗️ System Architecture

1. **Data Preprocessing**: Clean, encode, and scale input data.
2. **Model Training**: `RandomForestRegressor` inside a pipeline.
3. **Serialization**: Save the trained model using `joblib`.
4. **UI Layer**: Streamlit-based frontend with form inputs.
5. **Prediction + PDF Generation**: Predict and export reports.

---

## 🔁 Project Structure

salary-predictor-app/
│
├── app.py # Main Streamlit app
├── model/
│ └── salary_predictor_corrected.pkl
├── data/
│ └── salary_dataset.csv
├── assets/
│ └── screenshots, logo, etc.
├── requirements.txt
├── README.md
└── ...

yaml
Copy
Edit

---

## 📸 Screenshots

> Add screenshots in `assets/` folder and update the paths below

- 🏠 Home Dashboard  
- 💰 Salary Prediction Input Form  
- 📈 Model Analytics  
- 📄 PDF Report

---

## 🛠️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/shyamchouhan/salary-predictor-app.git
cd salary-predictor-app

# Create virtual environment & activate
python -m venv venv
venv\Scripts\activate    # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
📂 Dependencies
text
Copy
Edit
streamlit
pandas
numpy
scikit-learn
matplotlib
plotly
joblib
fpdf
Install them via:

bash
Copy
Edit
pip install -r requirements.txt
✅ Future Enhancements
Deploy to cloud (Heroku/AWS)

Add user authentication

Fetch live job market salary data

Extend model to global regions

🧠 Author
Shyam Chouhan
IBM Capstone – AI/ML
🔗 LinkedIn
🔗 GitHub

📚 References
Scikit-learn Docs

Streamlit Docs

Kaggle Salary Dataset

Python Official Docs

GitHub ML Deployment Repos

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

yaml
Copy
Edit

---

Would you like me to export this as a `README.md` file and attach it for download?



