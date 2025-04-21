import os
import time
from typeformar.dataset_generation.loss_array.key_pressed import (
    key_pressed_v1,
    key_pressed_v2,
)
from typeformar.dataset_generation.merger import process_logger_file_to_list
from typeformar.dataset_generation.frame_timestamp_inference import (
    process_video_with_joints,
)

##################################
# Read the logger file
##################################

log_file_name = "d1t2"  # input("Enter the video and log file name (e.g. abc): ")
logger_file_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", f"{log_file_name}.log"
)
logger_dicts = process_logger_file_to_list(logger_file_path)

##################################
# Extract timestamp joint data
##################################

video_file_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", f"{log_file_name}.mov"
)
uuid, timestamp_joint_list = process_video_with_joints(video_file_path)

##################################
# Time the key pressed functions
##################################

key_of_interest = 57

# Test out key_pressed_v1 function and record the time it takes
start_time = time.time()
key_pressed_v1_result = key_pressed_v1(
    logger_dicts, timestamp_joint_list, key_of_interest
)
end_time = time.time()
print(f"Time taken for key_pressed_v1: {end_time - start_time} seconds")

# Test out key_pressed_v2 function and record the time it takes
start_time = time.time()
key_pressed_v2_result = key_pressed_v2(
    logger_dicts, timestamp_joint_list, key_of_interest
)
end_time = time.time()
print(f"Time taken for key_pressed_v2: {end_time - start_time} seconds")

####################################################################
# Output the timestamps when the key starts being pressed
####################################################################


def pretty_print_ms(ms):
    """
    Print the time in a pretty format (HH:MM:SS.MS)
    """
    seconds = ms // 1000
    minutes = seconds // 60
    hours = minutes // 60
    print(f"{hours}:{minutes % 60}:{seconds % 60}.{ms % 1000}")


def print_press_start_timestamps(key_pressed_result):
    """
    Output the timestamps when the key starts being pressed.

    Input: key_pressed_result, a list of booleans where each element is a frame in the video

    Print: every timestamp when the key starts being pressed
    """
    first_frame_timestamp = timestamp_joint_list[0][0]
    has_been_logged = False
    for i, is_pressed in enumerate(key_pressed_result):
        if is_pressed and not has_been_logged:
            timestamp, _ = timestamp_joint_list[i]
            pretty_print_ms(timestamp - first_frame_timestamp)
            has_been_logged = True
        elif not is_pressed:
            has_been_logged = False


print_press_start_timestamps(key_pressed_v1_result)
print_press_start_timestamps(key_pressed_v2_result)
