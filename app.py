from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os

# Initialize Flask app
app = Flask(__name__)

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to clean user text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    cleaned = [word for word in words if word not in stop_words]
    return ' '.join(cleaned)

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    print(">> Flask is rendering the index page")
    if request.method == 'POST':
        user_text = request.form['user_text']
        cleaned = clean_text(user_text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        sentiment = "😊 Positive" if prediction == 1 else "😞 Negative"
        return render_template('index.html', result=sentiment, text=user_text)
    return render_template('index.html')

# Feedback route
@app.route('/feedback', methods=['POST'])
def feedback():
    original_text = request.form['original_text']
    predicted_label = int(request.form['predicted_label'])
    feedback = request.form['feedback']

    if feedback == 'no':
        # Assume corrected label is the opposite of what model predicted
        corrected_label = 0 if predicted_label == 1 else 1

        # Save to CSV
        new_entry = pd.DataFrame({
            'label': [corrected_label],
            'text': [original_text]
        })
        if os.path.exists('user_feedback.csv'):
            new_entry.to_csv('user_feedback.csv', mode='a', header=False, index=False)
        else:
            new_entry.to_csv('user_feedback.csv', index=False)

        message = "❗ Feedback recorded: You corrected the prediction."
    else:
        message = "✅ Great! Glad the prediction was right."

    return render_template('index.html', message=message)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
