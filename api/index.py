import sys
from pathlib import Path

# Dapatkan root directory proyek secara absolut
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from flask import Flask, jsonify, render_template, request

# Pastikan Flask mengenali lokasi folder templates dari root
app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'))

# Deklarasi lokasi file absolut
MODEL_PATH = BASE_DIR / 'model_lstm_passengers.onnx'
SCALER_PATH = BASE_DIR / 'scaler.pkl'
DATA_PATH = BASE_DIR / 'AirPassengers.csv'

# Load Model & Scaler
session = (
    ort.InferenceSession(str(MODEL_PATH)) if MODEL_PATH.exists() else None
)
scaler = joblib.load(str(SCALER_PATH)) if SCALER_PATH.exists() else None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/data', methods=['GET'])
def get_data():
    if not DATA_PATH.exists():
        return jsonify({'error': 'File AirPassengers.csv tidak ditemukan'}), 404

    df = pd.read_csv(str(DATA_PATH))
    if 'Passengers' in df.columns:
        df.rename(columns={'Passengers': '#Passengers'}, inplace=True)

    data = []
    for idx, row in df.iterrows():
        data.append({
            'date': str(row['Month']),
            'passengers': int(row['#Passengers']),
        })
    return jsonify({'data': data})


@app.route('/api/predict', methods=['POST'])
def predict():
    if session is None or scaler is None:
        return (
            jsonify({
                'error': (
                    f'Model atau Scaler belum dimuat. Model Path Exist:'
                    f' {MODEL_PATH.exists()}, Scaler Path Exist:'
                    f' {SCALER_PATH.exists()}'
                )
            }),
            500,
        )

    payload = request.get_json() or {}
    last_12_months = payload.get('last_12_months', [])

    if len(last_12_months) != 12:
        return (
            jsonify(
                {'error': 'Input harus berupa array berisi 12 data penumpang'}
            ),
            400,
        )

    try:
        input_array = np.array(last_12_months, dtype=np.float32).reshape(-1, 1)
        scaled_input = scaler.transform(input_array)
        X_input = scaled_input.reshape(1, 12, 1).astype(np.float32)

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        pred_scaled = session.run([output_name], {input_name: X_input})[0]
        prediction = scaler.inverse_transform(pred_scaled)[0][0]

        return jsonify({
            'success': True,
            'prediction': int(np.round(prediction)),
            'raw_prediction': float(prediction),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
