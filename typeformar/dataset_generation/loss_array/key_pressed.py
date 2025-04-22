def check_alternating_key_event_order(timestamp_keypress: list):
    return all(
        (i % 2 == 0 and d["type"] == "down") or (i % 2 == 1 and d["type"] == "up")
        for i, d in enumerate(timestamp_keypress)
    )


def key_pressed_v1(timestamp_keypress: list, timestamp_joint_list: list, key: int):
    """
    Check if a key was pressed at the given timestamp.
    """
    specific_key_only_list = filter_by_key(timestamp_keypress, key)
    # Ensure that the list alternates between 4 and 5
    assert check_alternating_key_event_order(specific_key_only_list)

    key_pressed_list = [False] * len(timestamp_joint_list)
    timestamp_keypress_idx = 0
    timestamp_joint_idx = 0

    keypress_time = specific_key_only_list[0]["time"]
    joint_time = timestamp_joint_list[0][0]

    key_pressed = False

    while True:
        if joint_time < keypress_time:
            key_pressed_list[timestamp_joint_idx] = key_pressed
            timestamp_joint_idx += 1
            if timestamp_joint_idx >= len(timestamp_joint_list):
                break
            joint_time = timestamp_joint_list[timestamp_joint_idx][0]

        else:
            key_pressed = not key_pressed
            key_pressed_list[timestamp_joint_idx] = key_pressed
            timestamp_keypress_idx += 1

            if timestamp_keypress_idx >= len(specific_key_only_list):
                break
            keypress_time = specific_key_only_list[timestamp_keypress_idx]["time"]
    return key_pressed_list


def key_pressed_v2(timestamp_keypress: list, timestamp_joint_list: list, key: int):
    """
    Check if a key was pressed at the given timestamp.
    """
    # Filter out the keypresses for the specific key
    specific_key_only_list = filter_by_key(timestamp_keypress, key)
    assert check_alternating_key_event_order(specific_key_only_list)
    specific_key_only_list = clip_logger(specific_key_only_list, timestamp_joint_list)

    # Initialize a list to store the key press status for each joint event
    key_pressed_list = [False] * len(timestamp_joint_list)

    # Initialize indices for key press and joint lists
    timestamp_keypress_idx = 0
    timestamp_joint_idx = 0

    for i, (joint_time, _) in enumerate(timestamp_joint_list):

        for j in range(timestamp_keypress_idx, len(specific_key_only_list)):
            keypress_time = specific_key_only_list[j]["time"]
            key_type = specific_key_only_list[j]["type"]

            if joint_time < keypress_time:
                if key_type == "up":
                    key_pressed_list[i] = True
                elif key_type == "down":
                    key_pressed_list[i] = False
                else:
                    raise ValueError(f"Invalid key type: {key_type}")
                break

    return key_pressed_list


def filter_by_key(timestamp_keypress: list, key: int):
    """
    Only keeps dictionaries where the key was pressed.
    """
    return [d for d in timestamp_keypress if d["keycode"] == key]


def key_pressed_v3(timestamp_keypress: list, timestamp_joint_list: list, key: int):
    """
    Check if a key was pressed at the given timestamp.
    Marks the joint with the closest timestamp to each keypress as True.
    """
    # Filter out the keypresses for the specific key
    specific_key_only_list = filter_by_key(timestamp_keypress, key)
    specific_key_only_list = clip_logger(specific_key_only_list, timestamp_joint_list)

    # Initialize a list to store the key press status for each joint event
    key_pressed_list = [False] * len(timestamp_joint_list)

    for space_dict in specific_key_only_list:
        keypress_time = space_dict["time"]
        key_type = space_dict["type"]

        if key_type == "up":
            continue

        # Find the closest joint time
        min_diff = float("inf")
        closest_idx = -1

        for i, (joint_time, _) in enumerate(timestamp_joint_list):
            # Calculate absolute time difference
            time_diff = abs(joint_time - keypress_time)

            # Update if this is the closest so far
            if time_diff < min_diff:
                min_diff = time_diff
                closest_idx = i

        # Mark the closest joint as True
        if closest_idx != -1:
            key_pressed_list[closest_idx] = True

    return key_pressed_list


def clip_logger(timestamp_keypress: list, timestamp_joint_list: list):
    """
    Clips logger so that the first timestamp is the same as the first joint timestamp
    and the last timestamp is the same as the last joint timestamp
    """
    # Get the first and last timestamps from the joint list
    first_joint_time = timestamp_joint_list[0][0]
    last_joint_time = timestamp_joint_list[-1][0]
    # Clip the logger list
    clipped_logger = [
        d
        for d in timestamp_keypress
        if first_joint_time <= d["time"] <= last_joint_time
    ]
    return clipped_logger
