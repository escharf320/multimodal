import os
import pickle
from frame_timestamp_inference import (
    extract_uuid_and_timestamps,
    extract_joint_positions,
)
from log_parser import process_logger_file_to_list

data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def has_video_been_processed(video_name):
    """Check if the video has been processed by looking for a pickle file"""
    file_path = os.path.join(data_dir, f"{video_name}.pkl")
    return os.path.exists(file_path)


def save_to_file(name, timestamp_joints, parsed_log_dicts):
    """Store the features and hand landmarks to a pickle file"""
    file_path = os.path.join(data_dir, f"{name}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump({"timestamp_joints": timestamp_joints, "logs": parsed_log_dicts}, f)


def read_from_file(name):
    """Read the timestamp_joints and log_dicts from a pickle file"""
    file_path = os.path.join(data_dir, f"{name}.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # 1. Read the video paths from the data directory
    # If the video is already processed, skip it

    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    video_paths = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".mov")
    ]

    print(f"Found {len(video_paths)} videos in the data directory.")
    print("Running QR code inference...")

    videos_to_process = []
    for video_path in video_paths:
        uuid, inferred_timestamp_map = extract_uuid_and_timestamps(video_path)
        if not has_video_been_processed(uuid):
            videos_to_process.append((video_path, uuid, inferred_timestamp_map))

    print(f"Found {len(videos_to_process)} videos to process.")

    # 2. For each unprocessed video, extract the features and the hand landmarks
    for video_path, uuid, inferred_timestamp_map in videos_to_process:
        print(f"Processing video {uuid}...")
        joint_positions = extract_joint_positions(video_path)

        timestamp_joints = []
        for f, timestamp in inferred_timestamp_map:
            joint_pos = joint_positions[f]
            timestamp_joints.append((timestamp, joint_pos))

        logger_path = os.path.join(data_dir, f"{uuid}.log")
        parsed_log_dicts = process_logger_file_to_list(logger_path)
        print(f"Saving video {uuid} to file...")
        save_to_file(uuid, timestamp_joints, parsed_log_dicts)
