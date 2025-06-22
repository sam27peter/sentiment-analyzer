import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# 📦 Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 🧹 Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

# 📁 Load the large dataset
df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding='ISO-8859-1',
    names=["label", "id", "date", "query", "user", "text"]
)

# ✅ Relabel: 0 = Negative, 4 = Positive → convert 4 to 1
df['label'] = df['label'].replace(4, 1)

# 🧼 Clean the text
df['cleaned_text'] = df['text'].apply(clean_text)

# 🧠 Add feedback if exists
if os.path.exists('user_feedback.csv'):
    print("🔁 Adding user feedback...")
    fb = pd.read_csv('user_feedback.csv', names=['label', 'text'])
    fb['cleaned_text'] = fb['text'].apply(clean_text)
    df = pd.concat([df, fb], ignore_index=True)

# ✂️ Train-test split
X = df['cleaned_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 🔠 Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🤖 Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 📊 Evaluate
y_pred = model.predict(X_test_vec)
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# 💾 Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved.")

# 🧹 Delete feedback after training
if os.path.exists('user_feedback.csv'):
    os.remove('user_feedback.csv')
    print("🗑 Feedback cleared.")
