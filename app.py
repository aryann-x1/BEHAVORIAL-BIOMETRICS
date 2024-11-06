from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load the trained LSTM model
model = load_model('auth_lstm_model.h5')  # Ensure this matches your model file name

@app.route('/')
def home():
    return "Welcome to the Continuous Authentication System. Use POST /authenticate to check authentication."

@app.route('/authenticate', methods=['POST'])
def authenticate():
    try:
        data = request.json

        if 'timeDiffs' not in data:
            return jsonify({'status': 'error', 'message': 'Missing time differences field'}), 400

        time_diffs = data['timeDiffs']

        sequence_length = 10  # Ensure this matches the sequence length your LSTM model expects
        if len(time_diffs) < sequence_length:
            return jsonify({'status': 'error', 'message': f'Insufficient time differences. At least {sequence_length} required.'}), 400

        # Prepare the input sequence for the LSTM
        input_sequence = np.array(time_diffs[-sequence_length:]).reshape(1, sequence_length, 1)  # Reshape to (samples, timesteps, features)

        # Predict with the LSTM model
        prediction = model.predict(input_sequence)

        print("Model Prediction Output:", prediction)  # Log the raw output of the model

        # Determine authentication status based on the model output
        if prediction[0] >= 0.98:  # Adjust threshold as needed
            return jsonify({'status': 'Authenticated'})
        else:
            return jsonify({'status': 'Anomaly Detected'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
