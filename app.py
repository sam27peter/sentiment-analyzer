from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, words as nltk_words
import os

# 📦 Download required NLTK data
nltk.download('stopwords')
nltk.download('words')
stop_words = set(stopwords.words('english'))
english_words = set(nltk_words.words())

# 🔁 Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# 🧹 Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

# 🤖 Detect gibberish: too many unknown words
def is_gibberish(cleaned_text):
    if not cleaned_text:
        return True
    tokens = cleaned_text.split()
    if not tokens:
        return True
    known = [w for w in tokens if w in english_words]
    ratio = len(known) / len(tokens)
    return ratio < 0.4  # More than 60% unknown = gibberish

# 🚀 Initialize Flask
app = Flask(__name__)
app.secret_key = "any-secret-key"

# 🏠 Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_text = request.form.get('user_text')
        cleaned = clean_text(user_text)

        print(f"[DEBUG] Raw input: {user_text}")
        print(f"[DEBUG] Cleaned input: {cleaned}")

        if not cleaned or is_gibberish(cleaned):
            sentiment = "🤔 I do not understand"
        else:
            try:
                vector = vectorizer.transform([cleaned])
                prediction = model.predict(vector)[0]
                sentiment = "😊 Positive" if prediction == 1 else "😞 Negative"
                print(f"[DEBUG] Prediction: {prediction}")
            except Exception as e:
                print(f"[ERROR] {str(e)}")
                sentiment = "🤔 I do not understand"

        return render_template('index.html', result=sentiment, text=user_text)

    return render_template('index.html')

# 👍👎 Feedback handler
@app.route('/feedback', methods=['POST'])
def feedback():
    original_text = request.form.get('original_text')
    predicted_label = request.form.get('predicted_label')
    feedback_value = request.form.get('feedback')

    if not original_text or not predicted_label or not feedback_value:
        flash("❗ Missing feedback data.")
        return redirect(url_for('index'))

    predicted_label = int(predicted_label)

    if feedback_value == 'no':
        corrected_label = 0 if predicted_label == 1 else 1
        df = pd.DataFrame([[corrected_label, original_text]])
        df.to_csv('user_feedback.csv', mode='a', header=False, index=False)
        flash("❗ Feedback recorded. Thank you!")
    else:
        flash("✅ Great! Prediction confirmed.")

    return redirect(url_for('index'))

# 🔁 Retrain route
@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        os.system("python sentiment_trainer.py")
        flash("✅ Model retrained with feedback!")
    except Exception as e:
        flash(f"❌ Retraining failed: {str(e)}")
    return redirect(url_for('index'))

# ▶ Run the app
if __name__ == '__main__':
    app.run(debug=True)
