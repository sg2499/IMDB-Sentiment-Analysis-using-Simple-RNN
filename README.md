# 🎬 IMDB Sentiment Analysis with Recurrent Neural Networks (RNN)

![GitHub repo size](https://img.shields.io/github/repo-size/sg2499/IMDB-Sentiment-Analysis-using-Simple-RNN)
![GitHub stars](https://img.shields.io/github/stars/sg2499/IMDB-Sentiment-Analysis-using-Simple-RNN?style=social)
![Last Commit](https://img.shields.io/github/last-commit/sg2499/IMDB-Sentiment-Analysis-using-Simple-RNN)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-orange)

This repository provides a complete solution to perform **Sentiment Analysis** on movie reviews from the IMDB dataset using a **Simple Recurrent Neural Network (RNN)**.

The project includes:
- 📦 Model building and training using TensorFlow/Keras
- 🧠 Preprocessing and padding of input text
- 🌐 An interactive **Streamlit web app** for live sentiment prediction
- 📊 Binary classification (Positive / Negative Sentiment)

---

## 📁 Project Folder Structure

```
📦imdb-rnn-sentiment/
├── iMDB Project using RNN.ipynb       # Training notebook
├── iMDB Predictions.ipynb            # Testing + Prediction notebook
├── app.py                            # Streamlit app for live prediction
├── Simple_RNN_iMDB.h5                # Trained RNN model
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
```

---

## 🧠 Model Overview

- **Dataset:** IMDB Movie Reviews (Keras built-in)
- **Architecture:**
  - Embedding Layer (128 dimensions)
  - SimpleRNN Layer (128 units, ReLU)
  - Dense Output Layer (Sigmoid activation)
- **Task:** Binary Sentiment Classification
- **Training Data:** 25,000 samples
- **Test Data:** 25,000 samples

---

## 🌐 Web App – Sentiment Classifier

The Streamlit app allows users to:
- Enter their own movie review
- Instantly classify it as **Positive** or **Negative**
- View the prediction confidence score

### 🖥️ Screenshot
> *(Add your app screenshot here for better visualization)*

---

## 💾 Setup Instructions

### 🔧 Clone the Repository

```bash
git clone https://github.com/sg2499/imdb-rnn-sentiment.git
cd imdb-rnn-sentiment
```

### 🐍 Create a Virtual Environment (Recommended)

```bash
conda create -n imdb_env python=3.10
conda activate imdb_env
```

### 📦 Install All Dependencies

```bash
pip install -r requirements.txt
```

### 🚀 Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📚 Dataset

The IMDB dataset used here is a preprocessed binary sentiment dataset built into `keras.datasets.imdb`.

- Contains 50,000 movie reviews (25k train + 25k test)
- Encoded as sequences of word indices
- Each review is labeled as 0 (negative) or 1 (positive)

---

## ✅ Requirements

Refer to `requirements.txt`. Key dependencies:

- `tensorflow==2.15.0`
- `streamlit==1.34.0`
- `numpy==1.26.4`
- `pandas==2.2.2`

---

## ✨ Key Features

- Simple and interpretable RNN architecture
- Clean and modular implementation
- Interactive UI with real-time predictions
- Easily extendable to LSTM/GRU models

---

## 📬 Contact

For any queries or collaboration requests, feel free to connect:

- 📧 [shaileshgupta841@gmail.com]
- 🧑‍💻 [GitHub Profile](https://github.com/sg2499)

---

> Built with ❤️ using TensorFlow and Streamlit.
