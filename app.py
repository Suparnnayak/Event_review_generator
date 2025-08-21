import pandas as pd
import numpy as np
import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from flask import Flask, request, render_template_string

# --- 1. Define File Paths ---
# This path should point to the directory containing your scripts and data
base_dir = r"D:\MACHINE_LEARNING\College_event_feedback"
word2vec_model_path = os.path.join(base_dir, 'scripts', 'word2vec_model.bin')
classifier_path = os.path.join(base_dir, 'scripts', 'classifier_model.pkl')

# --- 2. Load Models and Preprocessing Tools ---
print("Loading Word2Vec model and Classifier...")
try:
    # Load the Word2Vec model
    w2v_model = Word2Vec.load(word2vec_model_path)
    
    # Load the trained classifier
    with open(classifier_path, 'rb') as f:
        classifier = pickle.load(f)
    print("Models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure both models are in the 'scripts' folder.")
    exit()

# Set up preprocessing functions again
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Tokenizes, lemmatizes, and removes stopwords from a given text."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    return processed_tokens

def get_document_vector(tokens):
    """
    Creates a single vector for a document by averaging word vectors.

    Args:
        tokens (list): A list of preprocessed words.

    Returns:
        np.array: A 100-dimensional vector representing the document.
    """
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    
    if not vectors:
        # Return a special value to indicate an empty vector, not a zero vector
        return None
    else:
        return np.mean(vectors, axis=0)

# --- 3. Flask App Setup ---
app = Flask(__name__)

# HTML template with Tailwind CSS
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Event Feedback Classifier</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen">
    <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-xl">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">Event Feedback Classifier</h1>
        <p class="text-center text-gray-600 mb-8">Enter your feedback to predict the event's outcome (Success, Neutral, or Failure).</p>
        
        <form action="/predict" method="post" class="flex flex-col gap-4">
            <label for="review_text" class="text-gray-700 font-medium">Event Review:</label>
            <textarea id="review_text" name="review_text" rows="6" 
                      class="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-200"
                      placeholder="e.g., The event was amazing! The speaker was very engaging." required></textarea>
            
            <button type="submit" class="w-full py-3 px-4 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 transition duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
                Predict Outcome
            </button>
        </form>

        {% if prediction %}
        <div class="mt-8 p-6 rounded-lg border-2 border-dashed {% if prediction == 'SUCCESS' %}border-green-500{% elif prediction == 'FAILURE' %}border-red-500{% elif prediction == 'AMBIGUOUS' %}border-gray-500{% else %}border-yellow-500{% endif %}">
            <p class="text-center text-gray-700 font-bold text-lg mb-2">Prediction:</p>
            <p class="text-center text-xl font-extrabold {% if prediction == 'SUCCESS' %}text-green-600{% elif prediction == 'FAILURE' %}text-red-600{% elif prediction == 'AMBIGUOUS' %}text-gray-600{% else %}text-yellow-600{% endif %}">
                {{ prediction }}
            </p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    """Renders the main page with the form."""
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    """Takes a review, predicts the outcome, and renders the result."""
    if request.method == 'POST':
        review_text = request.form['review_text']
        
        # Preprocess the input text
        processed_tokens = preprocess_text(review_text)
        
        # Convert tokens to a single document vector
        vector = get_document_vector(processed_tokens)

        # Handle the case where the input contains no words from the model's vocabulary
        if vector is None:
            prediction = "AMBIGUOUS"
        else:
            # Reshape for the classifier (it expects a 2D array)
            vector = vector.reshape(1, -1)
            
            # Make the prediction
            prediction = classifier.predict(vector)[0]
        
        # Render the template with the prediction
        return render_template_string(html_template, prediction=prediction)

# --- 4. Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
