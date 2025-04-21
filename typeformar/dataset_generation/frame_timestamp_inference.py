import time
import cv2
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol
from PIL import Image
import mediapipe as mp
from predict_timestamps import predict_missing_timestamps
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Optional, Union
import threading


def process_single_qr_frame(
    frame_data: Tuple[int, np.ndarray],
) -> Tuple[int, Optional[str], Optional[int]]:
    """
    Process a single frame to extract QR code.

    Args:
        frame_data: Tuple containing (frame_number, frame)

    Returns:
        Tuple containing (frame_number, uuid, timestamp)
    """
    frame_number, frame = frame_data
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Process QR code
    decoded_objects = decode(pil_image, symbols=[ZBarSymbol.QRCODE])
    uuid = None
    timestamp = None
    if decoded_objects:
        decoded_str = decoded_objects[0].data.decode("utf-8")
        if "@" in decoded_str:
            decoded_uuid, timestamp = decoded_str.split("@")
            uuid = decoded_uuid
            timestamp = int(timestamp)

    return frame_number, uuid, timestamp


def process_single_hand_frame(
    frame: np.ndarray, hands: mp.solutions.hands.Hands
) -> Optional[List]:
    """
    Process a single frame to extract hand landmarks.

    IMPORTANT: These frames must be processed in order otherwise MediaPipe fails.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    hand_landmarks = None
    if results.multi_hand_landmarks:
        hand_landmarks = []
        for landmarks in results.multi_hand_landmarks:
            hand_landmarks.append([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])

    return hand_landmarks


def extract_uuid_and_timestamps(
    video_path: str,
) -> Tuple[str, List[Tuple[int, Optional[int]]]]:
    """
    Extracts QR code and timestamps from a video file.

    Args:
        video_path (str): The path to the video file.

    Returns:
        tuple: A tuple containing the UUID and frame data.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_number = -1

    # Read all frames first
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        frames.append((frame_number, frame))

    cap.release()

    # Process QR frames in parallel
    frame_results = []
    uuid = None
    uuid_lock = threading.Lock()

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_single_qr_frame, frame_data)
            for frame_data in frames
        ]
        for future in futures:
            frame_number, frame_uuid, timestamp = future.result()

            # Handle UUID with thread safety
            if frame_uuid is not None:
                with uuid_lock:
                    if uuid is None:
                        uuid = frame_uuid
                    else:
                        assert uuid == frame_uuid, "UUID mismatch"

            frame_results.append((frame_number, timestamp))

    # Sort results by frame number to maintain order
    frame_results.sort(key=lambda x: x[0])
    frame_results = predict_missing_timestamps(frame_results)

    if uuid is None:
        raise ValueError("No UUID found in video frames")

    return uuid, frame_results


def extract_joint_positions(video_path: str):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    frame_number = -1
    frame_hand_landmarks = []
    # Read all frames first
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        hand_landmarks = process_single_hand_frame(frame, hands)
        frame_hand_landmarks.append(hand_landmarks)

    cap.release()

    return frame_hand_landmarks


def process_video_with_joints(video_path):
    # Extract features from the video
    uuid, inferred_timestamp_map = extract_uuid_and_timestamps(video_path)
    joint_positions = extract_joint_positions(video_path)

    # Combine the infered timestamps with the joint positions
    timestamp_joints = []
    for f, timestamp in inferred_timestamp_map:
        joint_pos = joint_positions[f]
        timestamp_joints.append((timestamp, joint_pos))

    return uuid, timestamp_joints
