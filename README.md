<h1 align="center">🛒 Flipkart Product Review Sentiment Analysis</h1>

<p align="center">
  <b>End-to-End Machine Learning Project | NLP | Streamlit | AWS Deployment</b>
</p>

<hr/>

<h2>📌 Project Overview</h2>

<p>
Customer reviews play a critical role in influencing purchasing decisions on e-commerce platforms.
This project focuses on <b>sentiment analysis of real-time Flipkart product reviews</b>, classifying them as
<b>Positive</b> or <b>Negative</b>, and identifying customer pain points from negative reviews.
</p>

<p>
The project covers the complete <b>end-to-end machine learning lifecycle</b>, including:
</p>

<ul>
  <li>Data preprocessing & cleaning</li>
  <li>Exploratory Data Analysis (EDA)</li>
  <li>Feature engineering</li>
  <li>Model training & evaluation</li>
  <li>Error analysis</li>
  <li>Deployment using Streamlit</li>
</ul>

---

<h2>🎯 Objectives</h2>

<ul>
  <li>Classify customer reviews into <b>Positive</b> or <b>Negative</b> sentiment</li>
  <li>Handle noisy, real-world scraped text data (emojis, encoding issues, symbols)</li>
  <li>Compare multiple ML models using <b>F1-score</b></li>
  <li>Perform detailed <b>Error Analysis</b> (False Positives & False Negatives)</li>
  <li>Deploy a real-time sentiment analysis web application</li>
</ul>

---

<h2>📂 Dataset Description</h2>

<p>
The dataset consists of customer reviews scraped from the Flipkart website for three product categories:
</p>

<table border="1" cellpadding="8">
  <tr>
    <th>Dataset</th>
    <th>Category</th>
    <th>Product</th>
  </tr>
  <tr>
    <td>reviews_badminton</td>
    <td>Sports</td>
    <td>YONEX MAVIS 350 Nylon Shuttle</td>
  </tr>
  <tr>
    <td>reviews_tawa</td>
    <td>Cookware</td>
    <td>Master Superior Induction Base Tawa</td>
  </tr>
  <tr>
    <td>reviews_tea</td>
    <td>Food</td>
    <td>Tata Tea Gold</td>
  </tr>
</table>

<p><b>Each dataset contains:</b></p>

<ul>
  <li>Reviewer Name</li>
  <li>Rating</li>
  <li>Review Title</li>
  <li>Review Text</li>
  <li>Place of Review</li>
  <li>Date of Review</li>
  <li>Up Votes</li>
  <li>Down Votes</li>
</ul>

<p><b>⚠️ Note:</b> Data scraping was already performed. No scraping code is included.</p>

---

<h2>🔧 Tech Stack & Tools</h2>

<ul>
  <li><b>Python</b></li>
  <li>Pandas & NumPy – Data processing</li>
  <li>NLTK – Text preprocessing</li>
  <li>Scikit-learn – TF-IDF, ML models</li>
  <li>Matplotlib & Seaborn – Visualization</li>
  <li>Streamlit – Web application</li>
  <li>Pickle – Model serialization</li>
</ul>

---

<h2>🧠 Project Workflow</h2>

<pre>
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
</pre>

---

<h2>🧹 Text Preprocessing</h2>

<ul>
  <li>Handling missing values safely</li>
  <li>Fixing encoding issues (â?¹, â€™)</li>
  <li>Removing emojis, URLs, special characters</li>
  <li>Removing artifacts like <b>READ MORE</b></li>
  <li>Lowercasing text</li>
  <li>Stopword removal</li>
  <li>Lemmatization</li>
  <li>Removing very short reviews</li>
</ul>

---

<h2>🧪 Feature Engineering</h2>

<ul>
  <li>TF-IDF Vectorization</li>
  <li>Unigrams & Bigrams</li>
  <li>Maximum features: <b>5000</b></li>
  <li>Applied only on training data to avoid leakage</li>
</ul>

---

<h2>🤖 Models Trained</h2>

<table border="1" cellpadding="8">
  <tr>
    <th>Model</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>0.940</td>
  </tr>
  <tr>
    <td>SVM (Linear)</td>
    <td>0.945</td>
  </tr>
  <tr>
    <td><b>Naive Bayes</b></td>
    <td><b>0.947 (Selected)</b></td>
  </tr>
</table>

<p>
<b>Final Model:</b> Multinomial Naive Bayes <br/>
<b>Evaluation Metric:</b> F1-score
</p>

---

<h2>🔍 Error Analysis</h2>

<ul>
  <li><b>False Positives:</b> Negative reviews predicted as Positive</li>
  <li><b>False Negatives:</b> Positive reviews predicted as Negative</li>
</ul>

<p><b>Common Error Patterns:</b></p>

<ul>
  <li>Mixed sentiment reviews</li>
  <li>Sarcasm</li>
  <li>Short or ambiguous text</li>
  <li>Context-dependent terminology</li>
</ul>

---

<h2>🌐 Streamlit Web Application</h2>

<p>
A Streamlit app was built for real-time sentiment prediction.
</p>

<ul>
  <li>Real-time sentiment prediction</li>
  <li>Input validation & error handling</li>
  <li>Same preprocessing pipeline as training</li>
  <li>Clean and user-friendly UI</li>
</ul>

<pre>
streamlit run app.py
</pre>

---

<h2>📁 Project Structure</h2>

<pre>
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
</pre>

---

<h2>⚙️ Installation & Setup</h2>

<pre>
pip install -r requirements.txt
</pre>

<pre>
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
</pre>

---

<h2>🚀 Deployment</h2>

<ul>
  <li>Deployed on AWS EC2 (Amazon Linux)</li>
  <li>Streamlit app managed via systemd</li>
  <li>Auto-start on reboot enabled</li>
  <li>Publicly accessible via EC2 public IP</li>
</ul>

---

<h2>📌 Key Learnings</h2>

<ul>
  <li>Handling noisy real-world text is critical</li>
  <li>Consistent preprocessing is essential</li>
  <li>Error analysis provides deeper insights</li>
  <li>Traditional ML models can outperform expectations</li>
</ul>

---

<h2>🔮 Future Improvements</h2>

<ul>
  <li>BERT / Transformer-based models</li>
  <li>Multi-class sentiment (Very Positive / Neutral)</li>
  <li>Monitoring & logging</li>
  <li>Database integration</li>
</ul>

---
<hr/>

<h2 align="center">🚀 Live Project Demo</h2>

<p align="center">
  🔗 <b>Flipkart Review Sentiment Analyzer</b><br/>
  👉 <a href="http://18.60.105.58:8501/" target="_blank">
    http://18.60.105.58:8501/
  </a>
</p>

<hr/>

<h2 align="center">📌 Project By</h2>

<p align="center">
  <b>👨‍💻 Krishnakant Rajbhar</b><br/>
  Data Science & Generative AI Enthusiast
</p>

<hr/>

<h2>👨‍🎓 About Me — Krishnakant Rajbhar</h2>

<ul>
  <li>🎓 <b>B.Tech (AI & ML)</b> — 2nd Year, GITAM University, Bhubaneswar</li>
  <li>🎯 Passionate about <b>AI, Data Science, Machine Learning, and Generative AI</b></li>
  <li>📚 Completed <b>Data Science with Generative AI Certification</b> from iScale</li>
  <li>💻 Actively building <b>real-world AI/ML projects</b></li>
  <li>📌 GitHub Profile: 
    <a href="https://github.com/Krishnakant73" target="_blank">
      https://github.com/Krishnakant73
    </a>
  </li>
</ul>

<p>
📩 <b>Email:</b> 
<a href="mailto:krishnakant.kkr1@gmail.com">
krishnakant.kkr1@gmail.com
</a>
</p>

<hr/>

<h2>🔗 Useful Links</h2>

<ul>
  <li>🌐 Website: 
    <a href="https://www.innomatics.in/" target="_blank">
      Innomatics
    </a>
  </li>

  <li>🏢 LinkedIn (Innomatics Research Labs): 
    <a href="https://www.linkedin.com/school/innomatics-research-labs/" target="_blank">
      Innomatics Research Labs
    </a>
  </li>

  <li>💬 Discord Community: 
    <a href="https://discord.com/invite/cSXvEw6Hxd" target="_blank">
      Join Discord
    </a>
  </li>

  <li>🐙 GitHub: 
    <a href="https://github.com/Krishnakant73" target="_blank">
      Krishnakant73
    </a>
  </li>

  <li>🔗 LinkedIn (Personal): 
    <a href="https://www.linkedin.com/in/krishnakantrajbhar/" target="_blank">
      Krishnakant Rajbhar
    </a>
  </li>
</ul>

<hr/>

<h2>🙌 Acknowledgement</h2>

<p>
A huge thank you to <b>Innomatics Research Labs</b> for providing such an amazing internship opportunity and
helping students build strong careers in <b>AI, Data Science, and Machine Learning</b>.
</p>

<hr/>

<p align="center">
  ⭐ If you found this project helpful, please consider giving it a star!
</p>
