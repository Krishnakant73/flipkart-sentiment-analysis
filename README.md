🛒 Flipkart Product Review Sentiment Analysis
📌 Project Overview

Customer reviews play a critical role in influencing purchasing decisions on e-commerce platforms.
This project focuses on sentiment analysis of real-time Flipkart product reviews, classifying them as Positive or Negative, and identifying customer pain points from negative reviews.

The project covers the complete end-to-end machine learning lifecycle, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, error analysis, and deployment using Streamlit.

🎯 Objectives

Classify customer reviews into Positive or Negative sentiment

Handle noisy, real-world scraped text data (emojis, encoding issues, symbols, etc.)

Compare multiple ML models using F1-score

Perform error analysis (False Positives & False Negatives)

Deploy a real-time sentiment analysis web application

📂 Dataset Description

The dataset consists of customer reviews scraped from the Flipkart website for three different product categories:

Dataset	Category	Product
reviews_badminton	Sports	YONEX MAVIS 350 Nylon Shuttle
reviews_tawa	Cookware	Master Superior Induction Base Tawa
reviews_tea	Food	Tata Tea Gold

Each dataset contains the following fields:

Reviewer Name

Rating

Review Title

Review Text

Place of Review

Date of Review

Up Votes

Down Votes

⚠️ Note: Data scraping was already performed. No scraping code is included in this project.

🔧 Tech Stack & Tools

Python

Pandas & NumPy – Data processing

NLTK – Text preprocessing (stopwords, lemmatization)

Scikit-learn – TF-IDF, ML models, evaluation

Matplotlib & Seaborn – Visualization

Streamlit – Web application

Pickle – Model serialization

🧠 Project Workflow
Data Loading
     ↓
Data Cleaning & Standardization
     ↓
Exploratory Data Analysis (EDA)
     ↓
Text Preprocessing
     ↓
Feature Engineering (TF-IDF)
     ↓
Model Training (LR, SVM, Naive Bayes)
     ↓
Model Evaluation (F1-score)
     ↓
Error Analysis
     ↓
Model Saving
     ↓
Streamlit Deployment

🧹 Text Preprocessing

The following cleaning steps were applied to handle real-world noisy text:

Handling missing values safely

Fixing encoding issues (e.g. â?¹, â€™)

Removing emojis, URLs, and special characters

Removing artifacts like READ MORE

Lowercasing text

Stopword removal

Lemmatization

Removing very short reviews

This ensures high-quality input for feature extraction and modeling.

🧪 Feature Engineering

TF-IDF Vectorization

Unigrams and bigrams

Maximum features: 5000

Applied only on training data to avoid leakage

🤖 Models Trained

Three machine learning models were trained and evaluated:

Model	F1 Score
Logistic Regression	0.940
SVM (Linear)	0.945
Naive Bayes	0.947 (Selected)

📌 Final Model Chosen: Multinomial Naive Bayes
📌 Evaluation Metric: F1-score (to handle class imbalance)

🔍 Error Analysis

After model evaluation, error analysis was performed to understand misclassifications:

False Positives: Negative reviews predicted as Positive

False Negatives: Positive reviews predicted as Negative

Common Error Patterns:

Mixed sentiment reviews

Sarcasm

Short or ambiguous text

Context-dependent product terminology

This analysis helped identify limitations of traditional ML models.

🌐 Streamlit Web Application

A Streamlit app was built to perform real-time sentiment analysis on user-provided reviews.

Features:

Real-time sentiment prediction

Input validation & error handling

Same preprocessing pipeline as training

Clean and user-friendly UI

Run the App:
streamlit run app.py

📁 Project Structure
project/
│
├── notebook/
│   └── sentiment_analysis.ipynb
│
├── app/
│   ├── app.py
│   ├── tfidf_vectorizer.pkl
│   └── sentiment_nb_model.pkl
│
├── data/
│   └── final_cleaned_reviews.csv
│
├── requirements.txt
└── README.md

⚙️ Installation & Setup
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Download NLTK resources
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

🚀 Deployment

The Streamlit application can be deployed on AWS EC2

The trained model and vectorizer are loaded using serialized .pkl files

Lightweight deployment with no deep learning dependencies

📌 Key Learnings

Handling real-world noisy text is critical for NLP performance

Consistent preprocessing between training and inference is essential

Error analysis provides insights beyond accuracy metrics

Traditional ML models can perform strongly with proper preprocessing

🔮 Future Improvements

Use BERT or Transformer-based models for better context understanding

Add sentiment intensity (Neutral / Very Positive / Very Negative)

Deploy monitoring & logging for production usage

Store predictions in a database

👤 Author

Sentiment Analysis Project – Flipkart Reviews
Built as part of an end-to-end data science & machine learning workflow.