from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

app = Flask(__name__)

# Load model Keras (.h5) dan Scaler (.pkl)
MODEL_PATH = 'model_lstm_passengers.h5'
SCALER_PATH = 'scaler.pkl'
DATA_PATH = 'AirPassengers.csv'

model = None
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)


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
            'passengers': int(row['#Passengers'])
        })
    return jsonify({'data': data})


@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model atau Scaler belum dimuat'}), 500

    payload = request.get_json()
    last_12_months = payload.get('last_12_months', [])

    if len(last_12_months) != 12:
        return jsonify({'error': 'Input harus berupa array berisi 12 data penumpang'}), 400

    try:
        # Preprocessing & Scaling
        input_array = np.array(last_12_months).reshape(-1, 1)
        scaled_input = scaler.transform(input_array)

        # Reshape ke format 3D (1, 12, 1) untuk LSTM
        X_input = scaled_input.reshape(1, 12, 1)

        # Inperensi Model
        pred_scaled = model.predict(X_input)
        prediction = scaler.inverse_transform(pred_scaled)[0][0]

        return jsonify({
            'success': True,
            'prediction': int(np.round(prediction)),
            'raw_prediction': float(prediction)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

    app = app
