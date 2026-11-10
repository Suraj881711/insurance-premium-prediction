import os

import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model = joblib.load(os.path.join(BASE_DIR, 'model.lb'))

REGION_MAP = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/health')
def health():
    return {'status': 'ok'}


@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
    except (KeyError, TypeError, ValueError):
        return render_template(
            'index.html',
            error='⚠️ Age, BMI and children must be valid numbers.')

    sex_raw = request.form.get('sex', '').lower()
    smoker_raw = request.form.get('smoker', '').lower()
    region_raw = request.form.get('region', '').lower()

    if not (18 <= age <= 100):
        return render_template('index.html', error='⚠️ Age must be between 18 and 100.')
    if not (10 <= bmi <= 60):
        return render_template('index.html', error='⚠️ BMI must be between 10 and 60.')
    if not (0 <= children <= 10):
        return render_template('index.html', error='⚠️ Children must be between 0 and 10.')
    if sex_raw not in ('male', 'female'):
        return render_template('index.html', error='⚠️ Please select a valid sex.')
    if smoker_raw not in ('yes', 'no'):
        return render_template('index.html', error='⚠️ Please select smoker status.')
    if region_raw not in REGION_MAP:
        return render_template('index.html', error='⚠️ Please select a valid region.')

    sex = 1 if sex_raw == 'male' else 0
    smoker = 1 if smoker_raw == 'yes' else 0
    region_encoded = REGION_MAP[region_raw]

    features = np.array([[age, sex, bmi, children, smoker, region_encoded]])
    prediction = float(model.predict(features)[0])
    prediction = max(0.0, prediction)  # a premium can never be negative

    inputs = {
        'age': age, 'sex': sex_raw, 'bmi': bmi, 'children': children,
        'smoker': smoker_raw, 'region': region_raw,
    }
    return render_template(
        'result.html', prediction=round(prediction, 2), inputs=inputs)


if __name__ == '__main__':
    app.run(debug=True)
