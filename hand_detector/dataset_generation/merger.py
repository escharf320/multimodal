from frame_timestamp_inference import process_video_with_joints
import json
import os

####### TESTING #######
video_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "abc_video.mov"
)
logger_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "abc.log"
)


print(logger_path)
def process_logger_file_to_list(logger_path):
    """
    Process the logger file to create list of dictionaries
    Dictionaries contain the following keys:
    {
    "time":1745176108039,
    "type":5,
    "shiftKey":false,
    "keycode":18
    }
    """
    log_dicts = []
    
    with open(logger_path, 'r') as file:
        for line in file:
            if line.strip():
                try:
                    log_dict = json.loads(line.strip())
                    log_dicts.append(log_dict)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")
    
    return log_dicts
    
