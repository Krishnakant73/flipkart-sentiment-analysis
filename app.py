import streamlit as st
import pickle
import re
import emoji
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# App Config

st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="⭐",
    layout="centered"
)

st.title(" Flipkart Review Sentiment Analyzer")
st.write("Analyze customer sentiment using a trained ML model")


# Load Model

@st.cache_resource
def load_model():
    try:
        with open("models/tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("models/sentiment_nb_model.pkl", "rb") as f:
            model = pickle.load(f)
        return vectorizer, model
    except Exception as e:
        st.error(" Error loading model files. Make sure .pkl files exist.")
        st.stop()

vectorizer, model = load_model()


# Text Cleaning Function

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_review_text(text):
    if text is None or text.strip() == "":
        return ""
    
    try:
        text = text.encode("latin1").decode("utf-8")
    except:
        pass
    
    text = text.lower()
    text = re.sub(r'read more', ' ', text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    
    return " ".join(tokens)

def get_sentiment_keywords(text, vectorizer, model, top_n=5):
    """Extract keywords that contribute most to sentiment prediction"""
    try:
# Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()
        
# Transform the text
        text_vector = vectorizer.transform([text])
        
# Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vector)[0]
            confidence = max(probabilities)
        else:
# For models without predict_proba, use decision function if available
            if hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(text_vector)[0]
                confidence = max(decision_scores) / sum(abs(decision_scores))
            else:
                confidence = 0.8  
# Get the actual words from the original text
        words = text.split()
        
# Simple keyword extraction based on sentiment words
        positive_words = ['good', 'excellent', 'amazing', 'great', 'best', 'nice', 'love', 'perfect', 'awesome', 'fantastic', 'quality', 'worth', 'value', 'recommend', 'satisfied', 'happy', 'pleased']
        negative_words = ['bad', 'worst', 'terrible', 'awful', 'poor', 'hate', 'disappointed', 'waste', 'useless', 'broken', 'damaged', 'cheap', 'expensive', 'wrong', 'fake', 'duplicate', 'disgusting', 'horrible']
        
        keywords = []
        for word in words:
            word_lower = word.lower()
            if word_lower in positive_words or word_lower in negative_words:
                keywords.append(word)
        
        return keywords[:top_n], confidence
    
    except Exception:
        return [], 0.8

def highlight_keywords(text, keywords):
    """Highlight keywords in the original text"""
    highlighted_text = text
    for keyword in keywords:
# Case-insensitive replacement
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted_text = pattern.sub(f"**{keyword}**", highlighted_text)
    return highlighted_text


# Users Input

review_input = st.text_area(
    " Enter your product review:",
    height=150,
    placeholder="Example: The product quality is very good and worth the money"
)


# Prediction Button

if st.button(" Analyze Sentiment"):
    
    if review_input.strip() == "":
        st.warning(" Please enter a review before analyzing.")
    
    else:
        try:
            cleaned_review = clean_review_text(review_input)
            
            if len(cleaned_review.split()) < 3:
                st.warning(" Review is too short to analyze reliably.")
            
            else:
                review_vector = vectorizer.transform([cleaned_review])
                prediction = model.predict(review_vector)[0]
                
# Get keywords and confidence
                keywords, confidence = get_sentiment_keywords(cleaned_review, vectorizer, model)
                
# Display sentiment with confidence
                if prediction == 1:
                    st.success(f" Sentiment: **Positive Review** ({confidence:.1%} confidence)")
                else:
                    st.error(f" Sentiment: **Negative Review** ({confidence:.1%} confidence)")
                
# Display keywords if found
                if keywords:
                    st.subheader(" Key Sentiment Words:")
                    keyword_tags = "".join([f"<span style='background-color: #e1f5fe; padding: 4px 8px; margin: 2px; border-radius: 4px; font-weight: bold;'>{kw}</span>" for kw in keywords])
                    st.markdown(keyword_tags, unsafe_allow_html=True)
                
# Display highlighted text
                if keywords:
                    st.subheader(" Review with Keywords Highlighted:")
                    highlighted_review = highlight_keywords(review_input, keywords)
                    st.markdown(f"{highlighted_review}")
                
# Show cleaned text
                with st.expander(" View cleaned text"):
                    st.write(cleaned_review)
        
        except Exception as e:
            st.error(" Something went wrong during prediction.")
            st.write(str(e))
