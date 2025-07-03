# Import necessary libraries
import numpy as np                     # For numerical operations (not directly used here but good to have)
import tensorflow as tf               # TensorFlow framework for deep learning
from tensorflow.keras.datasets import imdb  # IMDB dataset for word-index reference
from tensorflow.keras.preprocessing import sequence  # For padding sequences
from tensorflow.keras.models import load_model       # To load the saved RNN model
import streamlit as st                # Streamlit for building the interactive web app

# Load the IMDB word index dictionary
# This maps words to their corresponding integer encoding
word_index = imdb.get_word_index()

# Create a reverse mapping from index to word (useful for decoding reviews if needed)
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the trained RNN model that was previously saved as an .h5 file
model = load_model('Simple_RNN_iMDB.h5')

# Function to decode an encoded review (not used in UI, but good for debugging or display)
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess raw text input (review)
def preprocess_text(text):
    words = text.lower().split()  # Convert to lowercase and split into words
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # Encode words using word_index; default to 2 for unknown
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)  # Pad to fixed length of 500
    return padded_review

# ---------------- Streamlit Web App Interface ----------------

# Set the app title
st.title('IMDB Movie Review Sentiment Analysis')

# Instruction for the user
st.write('Enter a movie review to classify it as Positive or Negative.')

# Text input area for the user to enter a movie review
user_input = st.text_area('Movie Review')

# Button to trigger classification
if st.button('Classify'):
    # Preprocess the input review
    preprocessed_input = preprocess_text(user_input)
    
    # Predict sentiment using the loaded RNN model
    prediction = model.predict(preprocessed_input)
    
    # Interpret the prediction score
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    # Display results
    st.write(f'Sentiment : {sentiment}')
    st.write(f'Prediction Score : {round(prediction[0][0], 2)}')

# Message shown if button is not yet clicked
else:
    st.write('Please provide a Movie Review.')