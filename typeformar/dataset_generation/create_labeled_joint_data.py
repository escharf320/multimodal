import os
import pickle

from frame_timestamp_inference import process_video_with_joints
from segment_words.find_words import words_from_log
from segment_words.segment_words_by_space import segment_joints_by_word
from segment_words.log_parser import process_logger_file_to_list


#### TESTING ####
video_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "d1t3.mov")
uuid, timestamp_joints = process_video_with_joints(video_path)


log_path= os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "d1t3.log"
)
logger_dicts = process_logger_file_to_list(log_path)





timestamped_words = words_from_log(log_path)
segmented_joints = segment_joints_by_word(timestamp_joints, timestamped_words)

# write dict to file
output_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "segmented_joints_d1t3.pkl")
with open(output_path, "wb") as f:
    pickle.dump(segmented_joints, f)


print("Number of words: ", len(timestamped_words))
print("Number of segmented joints: ", len(segmented_joints))
for word, joints in segmented_joints.items():
    print(f"Word: {word[0]}, Start Time: {word[1]}, Joints: {len(joints)}")

