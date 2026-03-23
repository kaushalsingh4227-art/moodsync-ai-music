from groq import Groq
import os
import json
import cv2
import numpy as np
import tensorflow as tf
import requests
import random

from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin,
    login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash


# ================= API KEYS =================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not set")

if not YOUTUBE_API_KEY:
    raise ValueError("❌ YOUTUBE_API_KEY not set")

groq_client = Groq(api_key=GROQ_API_KEY)


# ================= FLASK =================
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "moodsync_secret_key")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///moodsync.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


# ================= DATABASE =================
db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"


# ================= MODELS =================
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class DetectionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    emotion = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


# 🧠 NEW: USER MEMORY MODEL
class UserMemory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, unique=True)
    mood_history = db.Column(db.Text)  # JSON
    favorite_mood = db.Column(db.String(50))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ================= LOAD MODEL =================
MODEL_PATH = "emotion_cnn_v1.h5"
LABELS_PATH = "class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(LABELS_PATH) as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ================= MEMORY FUNCTIONS =================
def update_user_memory(user_id, emotion):
    memory = UserMemory.query.filter_by(user_id=user_id).first()

    if not memory:
        memory = UserMemory(
            user_id=user_id,
            mood_history=json.dumps([]),
            favorite_mood=emotion
        )
        db.session.add(memory)

    history = json.loads(memory.mood_history)
    history.append(emotion)

    history = history[-20:]  # keep last 20

    fav = max(set(history), key=history.count)

    memory.mood_history = json.dumps(history)
    memory.favorite_mood = fav

    db.session.commit()

    return fav, history


def get_user_memory(user_id):
    memory = UserMemory.query.filter_by(user_id=user_id).first()

    if not memory:
        return "neutral", []

    return memory.favorite_mood, json.loads(memory.mood_history)


# ================= EMOTION DETECTION =================
def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    return face


def detect_emotion(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        if len(faces) == 0:
            return None, 0, None

        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]

        face = gray[y:y+h, x:x+w]
        processed = preprocess_face(face)

        preds = model.predict(processed, verbose=0)[0]

        idx = np.argmax(preds)
        emotion = labels.get(idx, "neutral")
        confidence = float(np.max(preds))

        return emotion, confidence, (int(x), int(y), int(w), int(h))

    except Exception as e:
        print("❌ Emotion detection error:", e)
        return None, 0, None


# ================= MUSIC =================
def get_music_query(emotion):
    mapping = {
        "happy": "happy bollywood songs",
        "sad": "sad hindi songs",
        "angry": "motivational songs",
        "neutral": "lofi music",
        "romantic": "romantic songs",
        "surprised": "party songs"
    }
    return mapping.get(emotion, "music")


cache = {}

def get_youtube_video(query):
    try:
        url = "https://www.googleapis.com/youtube/v3/search"

        params = {
            "part": "snippet",
            "q": query,
            "key": YOUTUBE_API_KEY,
            "maxResults": 5,
            "type": "video"
        }

        res = requests.get(url, params=params)
        data = res.json()

        items = data.get("items", [])
        if not items:
            return None

        video = random.choice(items)
        return video["id"]["videoId"]

    except Exception as e:
        print("❌ YouTube API Error:", e)
        return None


# ================= ROUTES =================
@app.route("/")
def intro():
    return render_template("intro.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("dashboard"))

        flash("Invalid credentials")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":

        username = request.form.get("username")
        password = request.form.get("password")

        if User.query.filter_by(username=username).first():
            flash("User exists")
            return redirect(url_for("login"))

        user = User(
            username=username,
            password=generate_password_hash(password)
        )

        db.session.add(user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=current_user.username)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ================= DETECT =================
@app.route("/detect", methods=["POST"])
@login_required
def detect():

    file = request.files.get("frame")

    if not file:
        return jsonify({"error": "No frame"})

    img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    emotion, confidence, box = detect_emotion(frame)

    if emotion is None:
        return jsonify({"emotion": "-", "confidence": 0})

    session["last_emotion"] = emotion

    # 🧠 MEMORY UPDATE
    favorite_mood, history = update_user_memory(current_user.id, emotion)

    # 🎯 SMART MUSIC
    query = get_music_query(favorite_mood)
    video_id = get_youtube_video(query)

    db.session.add(DetectionHistory(
        user_id=current_user.id,
        emotion=emotion,
        confidence=confidence
    ))
    db.session.commit()

    return jsonify({
        "emotion": emotion,
        "confidence": round(confidence * 100, 2),
        "box": list(box) if box else None,
        "video_url": f"https://www.youtube.com/watch?v={video_id}" if video_id else None
    })


# ================= GET SONG =================
@app.route("/get_song", methods=["POST"])
@login_required
def get_song():

    mood = request.json.get("mood", "neutral")

    favorite_mood, _ = get_user_memory(current_user.id)

    final_mood = favorite_mood if favorite_mood else mood

    query = get_music_query(final_mood)
    video_id = get_youtube_video(query)

    return jsonify({
        "video_url": f"https://www.youtube.com/watch?v={video_id}" if video_id else None
    })


# ================= AI =================
@app.route("/ask_ai", methods=["POST"])
@login_required
def ask_ai():
    try:
        data = request.get_json()
        query = data.get("query")

        if not query:
            return jsonify({"response": "Please ask something."})

        current_mood = session.get("last_emotion", "neutral")
        favorite_mood, history = get_user_memory(current_user.id)

        prompt = f"""
        User current mood: {current_mood}
        Favorite mood: {favorite_mood}
        Recent moods: {history[-5:]}

        User message: {query}

        Respond naturally, emotionally aware, and helpful.
        """

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )

        reply = completion.choices[0].message.content.strip()

        return jsonify({"response": reply})

    except Exception as e:
        print("❌ GROQ ERROR:", e)
        return jsonify({"response": "AI is not working."})


# ================= RUN =================
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)