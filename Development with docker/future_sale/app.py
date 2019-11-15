import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.externals import joblib


with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

train_data = pd.read_csv("filtered.csv")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    """
    For handling predictions
    """
    int_features2 = [int(x) for x in request.form.values()]
    k_features = train_data[(train_data['item_id'] == int_features2[1]) & (train_data['shop_id'] == int_features2[0])]
    if len(k_features) == 0:
        return render_template('index.html', predict_text=f"Item id: %s and shop id: %s does not exits in our record" % (int_features2[1], int_features2[0]))

    prediction = model.predict(k_features)
    return render_template('index.html', predict_text=prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
