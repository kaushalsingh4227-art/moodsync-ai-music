import asyncio
import json
import cv2
import numpy as np
import tensorflow as tf
from collections import deque

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole
from av import VideoFrame

from flask import Blueprint, request, jsonify

webrtc_bp = Blueprint("webrtc", __name__)

# =====================================
# LOAD MODEL
# =====================================

model = tf.keras.models.load_model("emotion_cnn_v1.h5")

with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# =====================================
# TEMPORAL SMOOTHING BUFFER
# =====================================

emotion_buffer = deque(maxlen=10)


def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img / 255.0
    face_img = np.reshape(face_img, (1, 48, 48, 1))
    return face_img


def smooth_emotion(new_emotion):
    emotion_buffer.append(new_emotion)

    if len(emotion_buffer) < 5:
        return new_emotion

    return max(set(emotion_buffer), key=emotion_buffer.count)


# =====================================
# VIDEO TRACK PROCESSOR
# =====================================

class EmotionVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()

        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = gray[y:y+h, x:x+w]

            processed = preprocess_face(face)
            preds = model.predict(processed, verbose=0)[0]

            emotion = labels[np.argmax(preds)]
            emotion = smooth_emotion(emotion)

            cv2.putText(
                img,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


pcs = set()


@webrtc_bp.route("/offer", methods=["POST"])
async def offer():
    params = request.json
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(EmotionVideoTrack(track))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return jsonify(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )