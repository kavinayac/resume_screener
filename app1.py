# -*- coding: utf-8 -*-
"""
Resume Screening - Machine Learning Project with Flask Deployment
"""

import numpy as np
import pandas as pd
import re
import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

# Initialize the Flask app
app = Flask(__name__)

# Load the model, vectorizer, and label encoder
clf = pickle.load(open("model.pkl", "rb"))
word_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# Function to clean the resume text
def clean_resume(resume_text):
    resume_text = re.sub(r'http\S+', ' ', resume_text)  # Remove URLs
    resume_text = re.sub(r'RT|cc', ' ', resume_text)  # Remove RT and cc
    resume_text = re.sub(r'#\S+', '', resume_text)  # Remove hashtags
    resume_text = re.sub(r'@\S+', ' ', resume_text)  # Remove mentions
    resume_text = re.sub(r'[%s]' % re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"), ' ', resume_text)  # Remove punctuations
    resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text)  # Remove non-ASCII characters
    resume_text = re.sub(r'\s+', ' ', resume_text).strip()  # Remove extra whitespaces
    return resume_text

# Serve the index.html page
@app.route('/')
def index():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['resume']
    
    # Clean the resume text
    cleaned_data = clean_resume(data)
    
    # Transform the cleaned resume into feature vector
    vectorized_data = word_vectorizer.transform([cleaned_data])
    
    # Make the prediction using the model
    prediction = clf.predict(vectorized_data)
    
    # Decode the predicted category back to its original label
    category = le.inverse_transform(prediction)[0]
    
    # Return the result as JSON
    return jsonify({'category': category})

if __name__ == '__main__':
    app.run(debug=True)
