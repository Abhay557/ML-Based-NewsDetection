# 📰 ML-Based-NewsDetection

*Fake News Detection and Classification using NLP & Machine Learning.*  
This project aims to detect and classify **fake news** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  
The trained model analyzes news articles and predicts whether they are **real or fake** based on their content.

🔗 **Live Demo:** [Click Here](https://ml-based-newsdetection.onrender.com)  
🔗 **Dataset:** [Click Here](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

![App Screenshot](/img.png)

---

## ❓ Problem Statement
In today’s digital era, misinformation spreads rapidly and can have serious consequences. Distinguishing between **credible journalism** and **fake news** is increasingly difficult.  

This project provides an accessible **ML-powered tool** to automatically analyze and classify news text, promoting **media literacy** and reducing the spread of false information.

---


## 🚀 Tech Stack & Core Techniques

### **Tools & Technologies**

| Category          | Technologies |
|-------------------|--------------|
| **Backend**       | Python, Flask, Gunicorn |
| **Frontend**      | HTML5, CSS3, Bootstrap 5, JavaScript (ES6+) |
| **ML & Data Science** | Scikit-learn, Pandas, NLTK, Joblib |
| **Deployment & Tools** | Git, GitHub, Render |

---

## 🧠 Key Machine Learning Techniques

### **Text Vectorization: TF-IDF**
- Converts unstructured text into numerical format.
- Weights words higher if they are frequent in one article but rare across others.
- Provides strong predictive features for classification.

### **Classification Model: Passive Aggressive Classifier**
- Efficient **online learning** algorithm for text classification.
- Remains *passive* on correct predictions, *aggressive* on mistakes.
- Fast, scalable, and highly accurate (achieved **99.6% accuracy**).

---

## 🎯 Learned Objectives (Skills Demonstrated)

### **Machine Learning & NLP**
- Data Analysis & Cleaning (Pandas)
- Text Preprocessing (NLTK – lowercasing, stop-word removal, punctuation removal)
- Feature Engineering (TF-IDF vectorization)
- Model Training (PassiveAggressiveClassifier – 99.6% accuracy)
- Model Persistence (Joblib for saving model & vectorizer)

### **Backend Development**
- Built REST API with Flask
- `/predict` endpoint for JSON predictions
- Configured production server with Gunicorn

### **Frontend Development**
- Clean, responsive UI (HTML + Bootstrap 5)
- Async predictions using JavaScript Fetch API (no page reloads)

### **DevOps & Deployment**
- Version Control with Git/GitHub
- Deployed full-stack app to Render
- Managed dependencies with `requirements.txt`

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ML-Based-NewsDetection.git
   cd ML-Based-NewsDetection
   ```

2. **Create virtual environment & install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Flask app**

   ```bash
   python app.py
   ```

4. Open your browser and go to:

   ```
   http://localhost:5000/
   ```

---

## 📜 License

This project is licensed under the **MIT License** – free to use, modify, and distribute.

---
*one of the biggest backfall is that model is trined on small dataset.*
