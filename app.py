from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('models/naive_bayes_model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('message', '')
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
