# 🤖 AutoML Pipeline System (Web-Based)

### 🚀 Automating Machine Learning — From Upload to Prediction

---

## 🌐 Overview

This project is a **web-based AutoML system** that allows users to upload datasets, train machine learning models, and generate predictions — all through an interactive interface.

It eliminates the need for manual coding by automating the **entire ML lifecycle**.

---

## ✨ Features

🔐 User Authentication (Login/Register)
📁 Dataset Upload via Web Interface
⚙️ Data Preprocessing & Cleaning
🔀 Train-Test Split Functionality
🤖 Model Training (Multiple Algorithms)
📊 Model Evaluation & Visualization
🔮 Prediction on New Data

---

## 🧠 How It Works

```text
User → Upload Dataset → Preprocess Data → Split Data → Train Models → Evaluate → Predict
```

---

## 🏗️ Project Architecture

This project follows a **modular and scalable structure**:

```text
automl_pipeline_system/
├── app.py              # Main Flask application
├── database.py         # Database handling (user data)
├── requirements.txt    # Dependencies

├── templates/          # Frontend (HTML pages)
│   ├── login, register, dashboard, upload, train, predict, etc.

├── static/             # CSS & JavaScript
│   ├── css/
│   └── js/

├── utils/              # Core ML logic
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   └── visualization.py

├── models/             # Saved ML models
├── uploads/            # Uploaded datasets
```

---

## 🛠️ Tech Stack

* **Frontend**: HTML, CSS, JavaScript
* **Backend**: Python (Flask)
* **Machine Learning**: Scikit-learn
* **Database**: SQLite

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

---

## 📊 Key Highlights

* End-to-end ML pipeline integration
* Clean separation of frontend, backend, and ML logic
* Real-world application structure
* User-friendly interface

---

## 🎯 Learning Outcomes

* Building full-stack ML applications
* Working with Flask framework
* Designing modular ML pipelines
* Handling user input and data dynamically

---

## 🔮 Future Improvements

* Add more ML models (XGBoost, Deep Learning)
* Deploy on cloud (AWS / Render)
* Add model explainability (SHAP)
* Improve UI/UX

---

## 👩‍💻 Author

**Apurva Kavade**
Aspiring AI Engineer | B.Tech AI & DS

---


