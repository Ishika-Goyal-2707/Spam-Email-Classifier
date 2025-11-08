from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
MODEL_DIR = 'models'
model_path = os.path.join(MODEL_DIR, 'spam_classifier.pkl')
vectorizer_path = os.path.join(MODEL_DIR, 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = [message]
    vect = vectorizer.transform(data)
    prediction = model.predict(vect)[0]

    result = "🚫 Spam" if prediction == 1 else "✅ Not Spam"
    return render_template('index.html', message=message, result=result)

if __name__ == '__main__':
    app.run(debug=True)
