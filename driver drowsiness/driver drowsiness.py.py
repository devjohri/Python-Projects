import cv2
import pygame
from pygame import mixer
from threading import Thread

# Initialize Pygame mixer
pygame.mixer.init()

# Load the alarm sound (you can replace this with your own audio file)
mixer.music.load("C:/Users/devjo/Downloads/alarm-car-or-home-62554.mp3")

# Load the Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Define constants for EAR and thresholds
EYE_AR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 20

def eye_aspect_ratio(eye):
    A = abs(eye[1][0] - eye[5][0]) + abs(eye[1][1] - eye[5][1])
    B = abs(eye[2][0] - eye[4][0]) + abs(eye[2][1] - eye[4][1])
    C = abs(eye[0][0] - eye[3][0]) + abs(eye[0][1] - eye[3][1])
    ear = (A + B) / (2.0 * C)
    return ear

def sound_alarm():
    mixer.music.play()

cap = cv2.VideoCapture(0)  # Use the default camera (change the index if needed)
frame_counter = 0
drowsy_frames = 0
alarm_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_region = roi_gray[ey:ey + eh, ex:ex + ew]
            ear = eye_aspect_ratio(eye_region)

            # Draw the eyes and face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(roi_gray, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            if ear < EYE_AR_THRESHOLD:
                frame_counter += 1

                if frame_counter >= EYE_AR_CONSEC_FRAMES:
                    if not alarm_on:
                        alarm_on = True
                        alarm_thread = Thread(target=sound_alarm)
                        alarm_thread.daemon = True
                        alarm_thread.start()

                    cv2.putText(frame, "Drowsiness Detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                frame_counter = 0
                alarm_on = False

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
