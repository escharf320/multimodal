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


###### TESTTING #######
timestamp_keypress = [
    {
        "time": 100,
        "type": "down",
        "shiftKey": False,
        "keycode": 18,
    },  # Key '18' pressed at 100ms
    {
        "time": 500,
        "type": "up",
        "shiftKey": False,
        "keycode": 18,
    },  # Key '18' released at 500ms
    {
        "time": 1500,
        "type": "down",
        "shiftKey": False,
        "keycode": 18,
    },  # Key '18' pressed at 1500ms
    {
        "time": 2500,
        "type": "up",
        "shiftKey": False,
        "keycode": 18,
    },  # Key '18' released at 2500ms
    {
        "time": 3000,
        "type": "down",
        "shiftKey": False,
        "keycode": 20,
    },  # Key '20' pressed at 3000ms
    {
        "time": 3500,
        "type": "up",
        "shiftKey": False,
        "keycode": 20,
    },  # Key '20' released at 3500ms
    {
        "time": 4000,
        "type": "down",
        "shiftKey": False,
        "keycode": 18,
    },  # Key '18' pressed at 4000ms
    {
        "time": 4500,
        "type": "up",
        "shiftKey": False,
        "keycode": 18,
    },  # Key '18' released at 4500ms
    {
        "time": 5000,
        "type": "down",
        "shiftKey": False,
        "keycode": 20,
    },  # Key '20' pressed at 5000ms
    {
        "time": 5500,
        "type": "up",
        "shiftKey": False,
        "keycode": 20,
    },  # Key '20' released at 5500ms
    {
        "time": 6000,
        "type": "down",
        "shiftKey": False,
        "keycode": 21,
    },  # Key '21' pressed at 6000ms
    {
        "time": 6500,
        "type": "up",
        "shiftKey": False,
        "keycode": 21,
    },  # Key '21' released at 6500ms
]

timestamp_joint_list = [
    (200, "joint_data_1"),  # Event at 200ms
    (800, "joint_data_2"),  # Event at 800ms
    (1200, "joint_data_3"),  # Event at 1200ms
    (1800, "joint_data_4"),  # Event at 1800ms
    (2500, "joint_data_5"),  # Event at 2500ms
    (3200, "joint_data_6"),  # Event at 3200ms
    (4000, "joint_data_7"),  # Event at 4000ms
    (5000, "joint_data_8"),  # Event at 5000ms
    (6000, "joint_data_9"),  # Event at 6000ms
    (7000, "joint_data_10"),  # Event at 7000ms
]

# Test the function
key_pressed_list = key_pressed_v1(timestamp_keypress, timestamp_joint_list, 18)
key_pressed_list1 = key_pressed_v2(timestamp_keypress, timestamp_joint_list, 18)
print(f"Key Pressed List (Key 18): {key_pressed_list}")
print(f"Key Pressed List (Key 18): {key_pressed_list1}")

key_pressed_list_20 = key_pressed_v1(timestamp_keypress, timestamp_joint_list, 20)
key_pressed_list_20_1 = key_pressed_v2(timestamp_keypress, timestamp_joint_list, 20)
print(f"Key Pressed List (Key 20): {key_pressed_list_20}")
print(f"Key Pressed List (Key 20): {key_pressed_list_20_1}")

key_pressed_list_21 = key_pressed_v1(timestamp_keypress, timestamp_joint_list, 21)
key_pressed_list_21_1 = key_pressed_v2(timestamp_keypress, timestamp_joint_list, 21)
print(f"Key Pressed List (Key 21): {key_pressed_list_21}")
print(f"Key Pressed List (Key 21): {key_pressed_list_21_1}")
