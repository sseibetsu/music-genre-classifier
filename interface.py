from flask import Flask, request, jsonify, send_file
import numpy as np
import librosa
import tensorflow.keras as keras
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATHS = {
    "cnn": "model.h5",
    "mlp": "mlp_model.h5",
    "rnn": "rnn_model.h5"
}

def load_model(model_type="cnn"):
    model_path = MODEL_PATHS.get(model_type, "model.h5")
    return keras.models.load_model(model_path)

def extract_mfcc(file_path):
    SAMPLE_RATE = 22050
    DURATION = 30
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
    num_segments = 10
    hop_length = 512
    n_mfcc = 13
    n_fft = 2048

    signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)

    mfccs = []
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    for s in range(num_segments):
        start_sample = num_samples_per_segment * s
        finish_sample = start_sample + num_samples_per_segment

        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                    sr=SAMPLE_RATE,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)
        mfcc = mfcc.T
        mfccs.append(mfcc.tolist())

    return mfccs

def predict_genre(model, mfccs):
    genre_mapping = {
        0: 'blues',
        1: 'classical',
        2: 'country',
        3: 'disco',
        4: 'hiphop',
        5: 'jazz',
        6: 'metal',
        7: 'pop',
        8: 'reggae',
        9: 'rock'
    }

    predictions = np.zeros(10)
    for mfcc in mfccs:
        mfcc = np.array(mfcc)
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        prediction = model.predict(mfcc)
        predictions += prediction[0]

    predictions /= len(mfccs)
    predicted_indices = np.argsort(predictions)[::-1]

    return {genre_mapping[idx]: round(pred * 100, 2) for idx, pred in zip(predicted_indices, predictions[predicted_indices]) if pred > 0}

@app.route('/')
def index():
    return send_file('interface.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'})

    audio_file = request.files['audio']
    audio_file_path = 'temp_audio.wav'
    audio_file.save(audio_file_path)

    model_type = request.form.get("model_type", "cnn")
    model = load_model(model_type)

    mfccs = extract_mfcc(audio_file_path)

    predictions = predict_genre(model, mfccs)

    return jsonify({'prediction': predictions})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
