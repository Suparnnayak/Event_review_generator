import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

# It's a good practice to download necessary NLTK packages once
# If you run this script for the first time, uncomment these lines.
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     nltk.download('stopwords')
#
# try:
#     nltk.data.find('corpora/wordnet')
# except nltk.downloader.DownloadError:
#     nltk.download('wordnet')
#
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')

# Define the file path. Using a raw string (r"...") is best for Windows paths.
file_path = r"D:\MACHINE_LEARNING\College_event_feedback\data\raw\NLP-Sheet.csv"

# --- 1. Load the Dataset ---
print("Loading dataset...")
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
    exit()

print("Dataset loaded successfully.")
print("Original data shape:", df.shape)

# --- 2. Define Preprocessing Functions ---

# Initialize the lemmatizer and stop words set
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and tokenizes a single piece of text.

    Args:
        text (str): The input text (e.g., a review).

    Returns:
        list: A list of cleaned and lemmatized words.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers, keeping only letters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize the text into individual words
    tokens = text.split()
    
    # Remove stop words and perform lemmatization
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    
    return processed_tokens

# --- 3. Apply Preprocessing to the 'review_text' column ---
print("\nPreprocessing text data...")
# Apply the function to the 'review_text' column to create a new list of cleaned words
df['processed_text'] = df['review_text'].apply(preprocess_text)
print("Text preprocessing complete.")

# --- 4. Train the Word2Vec Model ---
print("\nTraining Word2Vec model...")
# The Word2Vec model needs a list of lists of words (tokens)
sentences = df['processed_text'].tolist()

# Train the model. 'vector_size' is the dimension of the word vectors.
# 'window' is the maximum distance between the current and predicted word within a sentence.
# 'min_count' ignores all words with frequency lower than this.
# 'workers' is the number of CPU cores to use for training.
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

print("Word2Vec model training complete.")

# --- 5. Save the Word2Vec Model ---
# This saves the model so you don't have to retrain it every time
model_filename = 'word2vec_model.bin'
w2v_model.save(os.path.join(os.path.dirname(file_path), '..', '..', 'scripts', model_filename))
print(f"Word2Vec model saved as {model_filename}")

# Note: The output of this script is a saved model. To use it, you would
# load it in another script. The next step is to use this model to create
# document vectors for each review.
