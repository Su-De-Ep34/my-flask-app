from flask import Flask, jsonify, render_template , request
import json
import pandas as pd
import numpy as np
import csv
import pickle
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
models = {}
encoders = {}
target = {
    'hrt' : 'target',
    'brn' : 'stroke',
    'alz' : 'Diagnosis',
    'ast' : 'Severity_Moderate',
    'dia' : 'diabetes',
    'kdy' : 'Diagnosis',
    'lng' : 'LUNG_CANCER',
    'obs' : 'NObeyesdad',
    'tyd' : 'binaryClass',
    'lvr' : 'Dataset'
}
for id , value in target.items():
    target[id] = value.upper()

names = ['hrt','dia','obs','brn','alz','ast','tyd','kdy','lvr','lng']
for name in names:
    model = None
    encoder = None
    with open(f"encoders/{name}_encoders.pkl" , "rb") as f:
        encoder = pickle.load(f)
    encoders[name] = encoder
    with open(f"models/{name}.pkl" , "rb") as f:
        model = pickle.load(f)
    models[name] = model

with open("datasets/attributes.json", "r") as f:
    attributes = json.load(f)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/input')
def input_page():
    return render_template("input.html")

@app.route('/get_attributes/<disease_id>')
def get_attributes(disease_id):
    if disease_id in attributes:
        return jsonify(attributes[disease_id])
    else:
        return jsonify({"error": "disease not found"}), 404

@app.route('/upload/<name>', methods=['POST'])
def upload(name):
    if 'filename' not in request.files:
        return "No file part", 400

    file = request.files['filename']

    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        df.columns = df.columns.str.upper()
        df.columns = df.columns.str.strip()
        #op = df[target[name]] not needed
        df = df[attributes[name]]
        df = df[sorted(df.columns)]
        encoder = encoders.get(name)
        model = models.get(name)

        if encoder is None or model is None:
            return f"No model or encoder found for disease: {name}", 404

        for col in df.select_dtypes(include=['object']):
            df[col] = encoder[col].transform(df[col])

        if name == "obs":
            predictions = model.predict(df)
            predicted_labels = encoder[target[name]].inverse_transform(predictions)

            unique, counts = np.unique(predicted_labels, return_counts=True)
            freq = dict(zip(unique, counts))
            most_common_label = max(freq, key=freq.get)
            total = len(predicted_labels)
            percentage = 100 * freq[most_common_label] / total

            result = {
                "most_likely_class": most_common_label,
                "percentage": f"{percentage:.2f}%",
                "message": f'"{most_common_label}" is the most predicted class for {percentage:.2f}% of people in this dataset.'
            }
            return f"<h2>{result['most_likely_class']} <br> {result['percentage']}</h2>"

        
        probas = model.predict_proba(df)
        avg_prob = float(np.mean(probas[:, 1]))

        result = {
            "message": f"{avg_prob * 100:.2f}% of people in this dataset are predicted to have this disease.",
            "percentage": f"{avg_prob * 100:.2f}%",
            "raw_value": avg_prob
        }
        return f"<h2>{result['message']} <br> {result['percentage']}</h2>"


if __name__ == "__main__":
    app.run(debug=True)