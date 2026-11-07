from flask import Flask, jsonify, render_template, request
import pandas as pd
import onnxruntime as ort
import numpy as np
import joblib
import os
import sys

# Tambahkan root directory ke sys.path agar file model, scaler & data bisa dibaca
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Path file
MODEL_PATH = os.path.join(BASE_DIR, 'model_lstm_passengers.onnx')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'AirPassengers.csv')

session = ort.InferenceSession(
    MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/data', methods=['GET'])
def get_data():
    if not os.path.exists(DATA_PATH):
        return jsonify({'error': 'File AirPassengers.csv tidak ditemukan'}), 404

    df = pd.read_csv(DATA_PATH)
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
        return jsonify({'error': 'Model ONNX atau Scaler belum dimuat'}), 500

    payload = request.get_json()
    last_12_months = payload.get('last_12_months', [])

    if len(last_12_months) != 12:
        return (
            jsonify(
                {'error': 'Input harus berupa array berisi 12 data penumpang'}
            ),
            400,
        )

    try:
        # Preprocessing & Scaling
        input_array = np.array(last_12_months, dtype=np.float32).reshape(-1, 1)
        scaled_input = scaler.transform(input_array)

        # Reshape ke 3D (1, 12, 1) float32 untuk ONNX LSTM
        X_input = scaled_input.reshape(1, 12, 1).astype(np.float32)

        # Inferensi ONNX
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
