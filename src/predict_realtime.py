import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
from voice import speak
import time

model = load_model("models/libras_model.h5")

actions = np.array([
    'oi',
    'sim',
    'nao',
    'e',
    'voce',
    'tudo bem'
])

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

sequence = deque(maxlen=30)

last_prediction = ""
last_added_time = 0
last_spoken_word = ""

PREDICTION_DELAY = 2

recognized_word = "..."
confidence = 0.0


def extract_keypoints(results):
    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

    while len(keypoints) < 126:
        keypoints.append(0)

    return np.array(keypoints[:126], dtype=np.float32)


def draw_panel(frame, recognized_word, confidence):
    height, width, _ = frame.shape

    black = (8, 8, 12)
    dark = (18, 18, 24)
    baby_pink = (203, 182, 255)
    white_soft = (235, 235, 240)

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (15, 15),
        (width - 15, 145),
        dark,
        -1
    )

    cv2.addWeighted(
        overlay,
        0.88,
        frame,
        0.12,
        0,
        frame
    )

    cv2.rectangle(
        frame,
        (15, 15),
        (width - 15, 145),
        baby_pink,
        2
    )

    cv2.putText(
        frame,
        "JARVIS LIBRAS IA",
        (35, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        baby_pink,
        2
    )

    cv2.putText(
        frame,
        f"PALAVRA: {recognized_word.upper()}",
        (35, 105),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        baby_pink,
        3
    )

    cv2.putText(
        frame,
        f"CONFIANCA {confidence:.2f}",
        (width - 290, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        white_soft,
        2
    )

    bar_x = width - 290
    bar_y = 80
    bar_w = 240
    bar_h = 24

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h),
        black,
        -1
    )

    filled = int(bar_w * confidence)

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + filled, bar_y + bar_h),
        baby_pink,
        -1
    )

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h),
        baby_pink,
        2
    )

    cv2.rectangle(
        overlay,
        (15, height - 60),
        (width - 15, height - 15),
        dark,
        -1
    )

    cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
        frame
    )

    cv2.putText(
        frame,
        "ESC sair   |   BACKSPACE resetar",
        (35, height - 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.6,
        baby_pink,
        2
    )

    return frame


while True:
    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    keypoints = extract_keypoints(results)

    sequence.append(keypoints)

    if len(sequence) == 30:
        input_data = np.array(sequence, dtype=np.float32)

        prediction = model.predict(
            np.expand_dims(input_data, axis=0),
            verbose=0
        )[0]

        predicted_index = np.argmax(prediction)
        predicted_action = actions[predicted_index]
        confidence = float(prediction[predicted_index])

        current_time = time.time()

        if confidence > 0.80:
            recognized_word = predicted_action

            if (
                predicted_action != last_prediction
                or current_time - last_added_time > PREDICTION_DELAY
            ):
                if predicted_action != last_spoken_word:
                    speak(predicted_action)
                    last_spoken_word = predicted_action

                last_prediction = predicted_action
                last_added_time = current_time

    frame = draw_panel(
        frame,
        recognized_word,
        confidence
    )

    cv2.imshow(
        "Jarvis Libras IA",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

    elif key == 8:
        last_prediction = ""
        last_spoken_word = ""
        recognized_word = "..."
        confidence = 0.0
        sequence.clear()

camera.release()
cv2.destroyAllWindows()