# 🎓 Placement Prediction System

An AI-powered Placement Prediction System built using Machine Learning and Streamlit.

The application analyzes a student's academic performance, skills, internships, projects, certifications, aptitude score, and extracurricular activities to predict placement probability and provide personalized career recommendations.

---

## 🚀 Live Demo

Add your deployed Streamlit/Hugging Face URL here.

Example:

https://your-app-name.streamlit.app

or

https://your-username-placement-prediction.hf.space

---

## 📌 Features

### Student Profile Assessment

* CGPA Evaluation
* Internship Analysis
* Project Portfolio Assessment
* Workshop & Certification Tracking
* Aptitude Score Evaluation
* Soft Skills Assessment
* Extracurricular Activity Analysis
* Placement Training Evaluation
* SSC Marks Analysis
* HSC Marks Analysis

### Machine Learning Prediction

* Logistic Regression Model
* Placement Probability Prediction
* Confidence Assessment
* Employability Score Generation

### Intelligent Insights

* Personalized Career Recommendations
* Placement Readiness Evaluation
* Strength Identification
* Improvement Suggestions

### Modern Dashboard

* Interactive UI
* Real-time Prediction
* Professional Analytics Cards
* Dynamic Probability Visualization
* Responsive Design

---

## 📊 Machine Learning Pipeline

### Data Processing

* Data Cleaning
* Feature Engineering
* Label Encoding
* Feature Scaling
* Train-Test Split

### Models Evaluated

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Tuned Random Forest

### Final Model Selection

Logistic Regression was selected as the final production model because it achieved the best balance between:

* ROC-AUC Score
* Generalization Performance
* Stability
* Interpretability

### Final Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 80.25% |
| Precision | 74.61% |
| Recall    | 80.21% |
| F1 Score  | 77.31% |
| ROC-AUC   | 88.37% |

Optimal Threshold: 0.448

---

## 🛠 Tech Stack

### Machine Learning

* Scikit-Learn
* Logistic Regression

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit
* Hugging Face Spaces

---

## 📂 Project Structure

```text
Placement Prediction/
│
├── app/
│   ├── app.py
│   ├── utils.py
│   ├── style.css
│   └── assets/
│
├── data/
│   ├── Placement_data.xlsx
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── models/
│   ├── final_model.pkl
│   ├── final_scaler.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest_baseline.pkl
│   ├── random_forest_tuned.pkl
│   └── model_config.json
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Baseline_Models.ipynb
│   ├── 04_Feature_Importance.ipynb
│   ├── 05_Random_Forest_Tuning.ipynb
│   ├── 06_Final_Model_Selection.ipynb
│   └── 07_Model_Explainability.ipynb
│
├── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ Run Locally

Clone the repository:

```bash
git clone <repository-url>
```

Navigate to project directory:

```bash
cd Placement-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/app.py
```

---

## 🎯 Future Enhancements

* SHAP Explainability Dashboard
* Student Resume Analyzer
* AI Career Advisor
* Interview Preparation Assistant
* Skill Gap Analysis
* Multi-Model Ensemble Prediction

---

## 👨‍💻 Author

Thiru Kumaran

M.Sc Applied Data Science

Machine Learning & Data Analytics Enthusiast

---

## ⭐ If you found this project useful

Please consider giving the repository a star.
