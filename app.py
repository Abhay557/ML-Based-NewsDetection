import joblib
import string
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, render_template, jsonify

# One-time download of the stopwords list
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Initialize the Flask application
app = Flask(__name__)

# --- PREPROCESSING and MODEL LOADING ---

# Load the set of English stopwords for faster processing
stop_words = set(stopwords.words('english'))

def preprocess_text_final(text):
    # 1. Lowercase and remove punctuation
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # 2. Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    
    return text

# Load the model and vectorizer when the app starts
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')


# --- FLASK ROUTES ---

@app.route('/')
def home():
    # This renders the main HTML page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get the text from the form
        user_input = request.form['message']
        
        # 2. Preprocess, vectorize, and predict
        cleaned_input = preprocess_text_final(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)
        
        # 3. Determine the result text
        result = "Real News" if prediction[0] == 1 else "Fake News"
        
        # 4. Return a clean JSON response
        return jsonify({'prediction_text': f'The article is classified as: {result}'})

    except Exception as e:
        # Handle potential errors during prediction
        print(f"An error occurred: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500


if __name__ == '__main__':
    app.run(debug=True)

