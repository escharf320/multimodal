from frame_timestamp_inference import process_video_with_joints

#### TESTING ####
video_path = "/Users/eli/Downloads/typing_example2.mov"
result = process_video_with_joints(video_path)
for timestamp, joints in result:
    print(type(timestamp))
    print(type(joints))
    print(f"Timestamp: {timestamp}, Joints: {joints}")
    break
