import os
from frame_timestamp_inference import process_video_with_joints
from segment_words.find_words import words_from_log
from segment_words.segment_words_by_space import segment_joints_by_space_pressed
from segment_words.log_parser import process_logger_file_to_list
from loss_array.key_pressed import key_pressed_v3

#### TESTING ####
video_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "d1t2.mov")
uuid, timestamp_joints = process_video_with_joints(video_path)
print("UUID: ", uuid)
print("Number of frames: ", len(timestamp_joints), "\n\n")

log_path= os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "d1t2.log"
)
logger_dicts = process_logger_file_to_list(log_path)

# Print the first frame
for timestamp, joints in timestamp_joints:
    print(f"Timestamp: {timestamp}, Joints: {joints}")
    break




key_of_interest = 57

# Test out key_pressed_v1 function and record the time it takes
space_pressed_truths = key_pressed_v3(
    logger_dicts, timestamp_joints, key_of_interest) 

words = words_from_log(log_path)

joint_words = segment_joints_by_space_pressed(timestamp_joints, space_pressed_truths)

#write space_pressed_truths to a file
with open("space_pressed_truths.txt", "w") as f:
    for item in space_pressed_truths:
        f.write("%s\n" % item)


print(len(words), " words found in log file")
print(len(joint_words)) 
# print(joint_words[:1], " joint words found in video file")
