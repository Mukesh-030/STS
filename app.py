from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pyttsx3
import threading
import time

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 175)  # Speed up voice output

last_spoken_time = 0

def speak_text(text):
    global last_spoken_time
    if time.time() - last_spoken_time >= 2:  # Set interval of 2 seconds
        last_spoken_time = time.time()
        thread = threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()))
        thread.start()

# Video capture function
def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        gesture_text = "Detecting..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract finger tip coordinates
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                ring_tip = hand_landmarks.landmark[16]
                pinky_tip = hand_landmarks.landmark[20]
                
                # Gesture Recognition
                if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y:
                    gesture_text = "Thumbs Up"
                elif index_tip.y < middle_tip.y and ring_tip.y < pinky_tip.y:
                    gesture_text = "Peace Sign"
                elif all(finger_tip.y > hand_landmarks.landmark[0].y for finger_tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]):
                    gesture_text = "Fist"
                elif all(finger_tip.y < hand_landmarks.landmark[0].y for finger_tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]):
                    gesture_text = "Open Palm"
                elif thumb_tip.x < index_tip.x and thumb_tip.y > index_tip.y:
                    gesture_text = "OK Sign"
                
                # Display gesture continuously
                cv2.putText(frame, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                speak_text(gesture_text)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
