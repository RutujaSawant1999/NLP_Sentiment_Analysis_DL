# Restaurant Review Sentiment Analysis using NLP & Deep Learning


🔹 Project Summary

This project focuses on building a sentiment analysis system that classifies restaurant reviews as positive or negative using Natural Language Processing (NLP) and a Deep Learning model.

The dataset consists of restaurant customer reviews stored in a TSV file. The text data is preprocessed, transformed into numerical features using TF-IDF Vectorization, and then passed to a Neural Network model built with Keras for binary classification.

The goal is to help businesses automatically understand customer feedback and improve service quality based on sentiment trends.


🔹 Key Features

• Text preprocessing (cleaning, tokenization, stopword removal)

• Feature extraction using TF-IDF Vectorizer

• Binary classification: Positive vs Negative sentiment

• Deep Learning model built using Keras Sequential API

• Model performance evaluation using accuracy

• Scalable for real-world feedback analysis


🔹 Dataset

• File: Restaurant_Reviews.tsv

• Columns:

   Review – Customer feedback text

   Liked – Sentiment label (1 = Positive, 0 = Negative)
   

🔹 Tools & Technologies

• Python

• Pandas, NumPy – Data handling

• Scikit-learn – TF-IDF, train-test split

• Keras (TensorFlow backend) – Neural Network

• Matplotlib – Training performance visualization


🔹 Model Architecture

• Input Layer (TF-IDF features)

• Dense Layer (128 units, ReLU)

• Dense Layer (64 units, ReLU)

• Output Layer (1 unit, Sigmoid)

Loss Function: Binary Crossentropy
Optimizer: Adam
Metric: Accuracy


🔹 Result

The model successfully learns patterns in customer reviews and predicts sentiment with high accuracy, demonstrating the effectiveness of combining NLP + Deep Learning for text classification.


🔹 Future Improvements

• Use LSTM / BERT for better context understanding

• Add multi-class sentiment labels

• Deploy as a web app using Flask or Streamlit

• Real-time review sentiment dashboard

