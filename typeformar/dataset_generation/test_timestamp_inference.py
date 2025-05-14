import os
from frame_timestamp_inference import process_video_with_joints
from segment_words.find_words import words_from_log
from segment_words.segment_words_by_space import segment_joints_by_space_pressed
from segment_words.log_parser import process_logger_file_to_list
from loss_array.key_pressed import key_pressed_v3


def segment_joints_by_word(timestamp_joints, timestamped_words, buffer=100):
    """
    Segments joints based on words.

    Args:
        timestamp_joints: List of joint data with timestamps
        timestamped_words: List of words with their corresponding timestamps

    Returns:
        A list of tuples containing the joint data and the corresponding word.
    """
    segmented_joints = {}

    for word, (start_time, end_time) in timestamped_words:
        joints_list = []
        for timestamp, jnts in timestamp_joints:
            if start_time - buffer <= timestamp <= end_time + buffer:
                joints_list.append(jnts)

        segmented_joints[(word, start_time)] = joints_list

    return segmented_joints


#### TESTING ####
key_of_interest = 57

video_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "d2t1.mov")
uuid, timestamp_joints = process_video_with_joints(video_path)
print("UUID: ", uuid)
print("Number of frames: ", len(timestamp_joints), "\n\n")

log_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "d2t1.log")
logger_dicts = process_logger_file_to_list(log_path)

# Print the first frame
for timestamp, joints in timestamp_joints:
    print(f"Timestamp: {timestamp}, Joints: {joints}")
    break


timestamped_words = words_from_log(log_path)

segmented_joints = segment_joints_by_word(timestamp_joints, timestamped_words)

print("Number of words: ", len(timestamped_words))
print("Number of segmented joints: ", len(segmented_joints))
for word, joints in segmented_joints.items():
    print(f"Word: {word[0]}, Start Time: {word[1]}, Joints: {len(joints)}")
