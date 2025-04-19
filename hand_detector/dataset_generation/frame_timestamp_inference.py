import cv2
from pyzbar.pyzbar import decode, ZBarSymbol
from PIL import Image
import numpy as np
from sklearn.linear_model import LinearRegression
import re
import mediapipe as mp
from predict_timestamps import predict_missing_timestamps

def process_video_with_joints(video_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    frame_data = []  # stores (frame_number, timestamp)
    joint_positions = []  # stores (frame_number, [[right_hand_landmarks], [left_hand_landmarks]]) 
    timestamp_pattern = re.compile(r"[a-zA-Z0-9\-]+@(\d+)")


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        decoded_objects = decode(pil_image, symbols=[ZBarSymbol.QRCODE])

        # Extract timestamp from QR code
        timestamp = None
        if decoded_objects:
            match = timestamp_pattern.match(decoded_objects[0].data.decode("utf-8"))
            if match:
                timestamp = int(match.group(1))
                frame_data.append((frame_number, timestamp))
            else:
                frame_data.append((frame_number, None))
        else:
            frame_data.append((frame_number, None))

        # Process the frame for hand landmarks
        results = hands.process(rgb_frame)
        hand_landmarks = None
        if results.multi_hand_landmarks:
            hand_landmarks = []
            for landmarks in results.multi_hand_landmarks:
                hand_landmarks.append(
                    [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
                )
        joint_positions.append((frame_number, hand_landmarks)) 

    cap.release()
    cv2.destroyAllWindows()

    # Fill missing timestamps using linear regression
    filled_data = predict_missing_timestamps(frame_data)
    
    ordered_output = []
    for (f, timestamp) in filled_data:
        _, joint_pos = joint_positions[f - 1]
        ordered_output.append((timestamp, joint_pos))  

    return ordered_output


#### TESTING ####
video_path = "/Users/eli/Downloads/typing_example2.mov"
result = process_video_with_joints(video_path)
for timestamp, joints in result:
    print(type(timestamp))
    print(type(joints))
    print(f"Timestamp: {timestamp}, Joints: {joints}")
    break 
    

