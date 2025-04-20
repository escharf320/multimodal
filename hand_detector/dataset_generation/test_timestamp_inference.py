import os
from frame_timestamp_inference import process_video_with_joints

#### TESTING ####
video_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "typing_example2.mov"
)
uuid, timestamp_joints = process_video_with_joints(video_path)
print("UUID: ", uuid)
print("Number of frames: ", len(timestamp_joints), "\n\n")

# Print the first frame
for timestamp, joints in timestamp_joints:
    print(f"Timestamp: {timestamp}, Joints: {joints}")
    break
