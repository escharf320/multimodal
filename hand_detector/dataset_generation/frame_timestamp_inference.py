import re
import cv2
from pyzbar.pyzbar import decode, ZBarSymbol
from PIL import Image
import mediapipe as mp
from predict_timestamps import predict_missing_timestamps


def process_video_with_joints(video_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    frame_number = -1  # Start at -1 so that the first += 1 frame is 0
    frame_data = []  # stores (frame_number, timestamp)
    joint_positions = (
        []
    )  # stores [[right_hand_landmarks], [left_hand_landmarks]] or None for every frame
    uuid = None

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
            # Decode the QR code format
            decoded_str = decoded_objects[0].data.decode("utf-8")
            assert "@" in decoded_str, "No timestamp found in QR code"
            decoded_uuid, timestamp = decoded_str.split("@")

            # Ensure the UUID doesn't change
            assert uuid is None or uuid == decoded_uuid, "UUID mismatch"
            uuid = decoded_uuid

            # Append to the frame data
            timestamp = int(timestamp)
            frame_data.append((frame_number, timestamp))
        else:
            frame_data.append((frame_number, None))

        # Process the frame for hand landmarks
        results = hands.process(rgb_frame)
        hand_landmarks = None
        if results.multi_hand_landmarks:
            hand_landmarks = []
            for landmarks in results.multi_hand_landmarks:
                hand_landmarks.append([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
        joint_positions.append(hand_landmarks)

    cap.release()

    # Fill missing timestamps using linear regression
    filled_data = predict_missing_timestamps(frame_data)

    ordered_output = []
    for f, timestamp in filled_data:
        joint_pos = joint_positions[f]
        ordered_output.append((timestamp, joint_pos))

    return uuid, ordered_output
