import pandas as pd
import numpy as np
import os
import pickle
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Define File Paths ---
base_dir = r"D:\MACHINE_LEARNING\College_event_feedback"
data_path = os.path.join(base_dir, 'data', 'raw', 'NLP-Sheet.csv')
word2vec_model_path = os.path.join(base_dir, 'scripts', 'word2vec_model.bin')
classifier_path = os.path.join(base_dir, 'scripts', 'classifier_model.pkl')

# --- 2. Load Data and Word2Vec Model ---
print("Loading dataset...")
try:
    df = pd.read_csv(data_path)
    # The 'processed_text' column from the previous script is not saved in the CSV.
    # We will need to re-run the preprocessing on the fly.
    # You could also save the processed data to a new CSV to save time.
    # For now, let's define the preprocessing functions again.
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        processed_tokens = [
            lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
        ]
        return processed_tokens

    df['processed_text'] = df['review_text'].apply(preprocess_text)
    print("Dataset loaded and preprocessed.")

except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the file paths are correct.")
    exit()

print("Loading Word2Vec model...")
try:
    w2v_model = Word2Vec.load(word2vec_model_path)
    print("Word2Vec model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please run the word2vec.py script first.")
    exit()

# --- 3. Create Document Vectors (Averaging Word Vectors) ---
print("Creating document vectors...")

def get_document_vector(tokens):
    """
    Creates a single vector for a document by averaging word vectors.

    Args:
        tokens (list): A list of preprocessed words from a review.

    Returns:
        np.array: A 100-dimensional vector representing the document.
                  Returns a zero vector if no words are in the vocabulary.
    """
    # Filter out words that are not in the Word2Vec model's vocabulary
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    
    if not vectors:
        # If the list is empty, return a zero vector of the same size
        return np.zeros(w2v_model.vector_size)
    else:
        # Calculate the mean (average) of the vectors
        return np.mean(vectors, axis=0)

# Apply the function to each row to create the feature matrix
X = np.vstack(df['processed_text'].apply(get_document_vector))
y = df['event_outcome']

print("Document vectors created. Shape of feature matrix (X):", X.shape)

# --- 4. Train the Classifier ---
print("\nTraining the Logistic Regression classifier...")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
print("Classifier training complete.")

# --- 5. Evaluate the Model ---
print("\nEvaluating the model...")
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 6. Save the Trained Classifier ---
print("\nSaving the trained classifier...")
with open(classifier_path, 'wb') as f:
    pickle.dump(classifier, f)
print(f"Classifier saved as {os.path.basename(classifier_path)}.")
print("\nScript finished.")
