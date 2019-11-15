import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.externals import joblib


with open('new_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    """
    For handling predictions
    """
    int_features = [34,	5,	5037,	11,	2015,	1.444444,	5037,	19,	19,	5]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(final_features)
    return render_template('index.html', predict_text=prediction)


