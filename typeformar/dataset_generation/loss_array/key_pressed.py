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
