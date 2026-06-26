import os
import time
import cv2
import mediapipe as mp


MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]



def draw_hand_landmarks(frame, hand_landmarks):
    height, width = frame.shape[:2]

    for start, end in HAND_CONNECTIONS:
        start_landmark = hand_landmarks[start]
        end_landmark = hand_landmarks[end]

        start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
        end_point = (int(end_landmark.x * width), int(end_landmark.y * height))

        cv2.line(frame, start_point, end_point, (255, 0, 255), 2)

    for landmark in hand_landmarks:
        point = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
    
    tipIDs = [4, 8, 12, 16, 20]
    if hand_landmarks:
        fingers = []

        if hand_landmarks[tipIDs[0]].x < hand_landmarks[tipIDs[0] - 1].x:
                fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if hand_landmarks[tipIDs[id]].y < hand_landmarks[tipIDs[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)
        cv2.putText(frame, f"Fingers: {totalFingers}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

if not os.path.exists(MODEL_PATH):
    print("Missing hand_landmarker.task. Put it in the same folder as main.py.")
    raise SystemExit

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

pTime = 0
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(
            frame, 
            f"FPS: {int(fps)}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                draw_hand_landmarks(frame, hand_landmarks)

        cv2.imshow("MediaPipe Hand Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
