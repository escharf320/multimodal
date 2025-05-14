

def endpoint_indicies(space_pressed_truths:list): 
    """
    Inputs: 
        space_pressed_truth: list of true or false when space was pressed for each frame 
    Output:
        tuple of INCLUSIVE (start, end) indices for each word in the text

    """
    words_indices = []
    boolean_segments = num_consecutive_bools(space_pressed_truths)
    num_diff_segs = len(boolean_segments)


    for i in range(num_diff_segs-2):
        _, boolean, start, _ = boolean_segments[i]
        _, _, end, _ = boolean_segments[i+2]

        if boolean: #skip adding the joint movements for spaces (this adds joint movements for the spaces as if they were words)
            continue 

        words_indices.append((start, end))
    
    _, _, s1, e1 = boolean_segments[-2]
    _, _, s2, e2= boolean_segments[-1]


    #adding the last segment becuase we skip it in the loop. 
    words_indices.append((s1, e2))
 
    return words_indices
        



def num_consecutive_bools(values):
    """
    Returns a list of tuples where each tuple contains the count of consecutive True or False values, 
    the corresponding boolean value, and the start and end indices of that segment.

    Ouput: (count, value, start, end)
    """
    result = []
    bool_counter = 1  
    start_index = 0 
    
    for i in range(1, len(values)):
        if values[i] == values[i - 1]:  # If the current value is the same as the previous one, increment the counter
            bool_counter += 1
        else:
            result.append((bool_counter, values[i - 1], start_index, i - 1)) 
            bool_counter = 1  # Reset counter for the new value
            start_index = i  # Set new start index for the next segment
            
    result.append((bool_counter, values[-1], start_index, len(values) - 1))  # Final segment with its start and end indices
    return result



def segment_joints_by_space_pressed(timestamp_joint_list, space_pressed_truths):
    """
    Segments the joint data based on space pressed events.
    
    Args:
        timestamp_joint_list: List of joint data with timestamps
        space_pressed_truths: List of boolean values indicating space pressed events

    Returns:
        List of tuples with the segmented joint data based on space pressed events
    """
    assert len(timestamp_joint_list) == len(space_pressed_truths), "Length of timestamp_joint_list and space_pressed_truths must be equal"
    
    words_indices = endpoint_indicies(space_pressed_truths)
    segmented_joints = []

    for start, end in words_indices:
        segmented_joints.append(tuple(timestamp_joint_list[start:end + 1]))

    return segmented_joints


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
            if start_time-buffer <= timestamp <= end_time+buffer:
                joints_list.append(jnts)
        
        if joints_list:
            segmented_joints[(word, start_time)] = joints_list
    

    
    return segmented_joints




    
# #### TESTTING ####
# values = [False, True, False, False, True, False, True, True, False, False]
# # [(0,2), (2,5), (5,8), (6, 9)]
# timestamp_joint_list = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0]] 


# segs = endpoint_indicies(values)
# print("Segmented Joints: ", segs)
# print(len(segs), " segments found")


