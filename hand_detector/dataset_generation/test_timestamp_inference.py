import os
from frame_timestamp_inference import process_video_with_joints

#### TESTING ####
video_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "typing_example2.mov"
)
uuid, result = process_video_with_joints(video_path)
print("UUID: ", uuid)
print("Number of frames: ", len(result), "\n\n")

# Print the first frame
for timestamp, joints in result:
    print(f"Timestamp: {timestamp}, Joints: {joints}")
    break
