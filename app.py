from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import random
import tensorflow as tf
import librosa
import sounddevice as sd

app = Flask(__name__)

# Load the models
video_model = tf.keras.models.load_model('LeNet5_model.h5')
audio_model = tf.keras.models.load_model('history3.h5')

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

video_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral', 7: 'Laugh', 8: 'Confused', 9: 'Contempt'}
audio_labels = {0: 'Fear', 1: 'Pleasant Surprise', 2: 'Sad', 3: 'Angry', 4: 'Disgust', 5: 'Happy', 6: 'Neutral', 7: 'Angry', 8: 'Disgust', 9: 'Fear', 10: 'Happy', 11: 'Neutral', 12: 'Pleasant Surprise', 13: 'Sad'}

emotion_list = list(video_labels.values())

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

target_emotion = None

def generate_frames():
    global target_emotion
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (48, 48))
                img = face_img.reshape(1, 48, 48, 1) / 255.0

                pred = video_model.predict(img)
                emotion_index = np.argmax(pred[0])
                emotion_label = video_labels[emotion_index]
                emotion_percentage = round(pred[0][emotion_index] * 100, 2)
                text = f"{emotion_label}: {emotion_percentage}%"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if target_emotion and emotion_label == target_emotion and emotion_percentage >= 85:
                    cv2.putText(frame, "Done", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_random_emotion')
def get_random_emotion():
    global target_emotion
    target_emotion = random.choice(emotion_list)
    return jsonify({'emotion': target_emotion})

@app.route('/predict_audio_emotion')
def predict_audio_emotion():
    duration = 5
    fs = 44100
    audio = record_audio(duration, fs)
    features = extract_features(audio, fs)
    print(f"Extracted Features: {features}")  # Log extracted features

    features = features.reshape(1, -1)
    prediction = audio_model.predict(features)
    predicted_index = np.argmax(prediction)
    print(f"Prediction Array: {prediction}")  # Log prediction array
    print(f"Predicted Index: {predicted_index}")  # Log predicted index

    if predicted_index in audio_labels:
        predicted_emotion = audio_labels[predicted_index]
        predicted_percentage = round(prediction[0][predicted_index] * 100, 2)
    else:
        predicted_emotion = "Unknown"
        predicted_percentage = 0

    response = {
        'emotion': predicted_emotion,
        'percentage': predicted_percentage
    }

    global target_emotion
    if predicted_emotion == target_emotion and predicted_percentage >= 85:
        response['result'] = "Done"
    else:
        response['result'] = "Wrong"

    return jsonify(response)

def record_audio(duration=5, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    print("Recording finished")
    return np.squeeze(recording)

def extract_features(audio, sr=44100):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)

    mfccs = np.mean(mfccs.T, axis=0)
    chroma = np.mean(chroma.T, axis=0)
    mel = np.mean(mel.T, axis=0)

    features = np.hstack((mfccs, chroma, mel))
    features = (features - np.mean(features)) / np.std(features)

    if len(features) < 40:
        features = np.pad(features, (0, 40 - len(features)), 'constant')
    else:
        features = features[:40]

    return features

if __name__ == "__main__":
    app.run(debug=True)
