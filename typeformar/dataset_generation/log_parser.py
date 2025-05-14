import json


def preprocess_log_dict(log_dict):
    """
    Preprocess the log dictionary to ensure the "type" key is either "down" or "up"
    """
    if log_dict["type"] == 4:
        log_dict["type"] = "down"
    elif log_dict["type"] == 5:
        log_dict["type"] = "up"
    else:
        raise ValueError(f"Invalid type: {log_dict['type']}")
    return log_dict


def process_logger_file_to_list(logger_path):
    """
    Process the logger file to create list of dictionaries
    Dictionaries contain the following keys:
    {
    "time":1745176108039,
    "type":"down" or "up",
    "shiftKey":false,
    "keycode":18
    }
    """
    log_dicts = []

    with open(logger_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                try:
                    log_dict = json.loads(line.strip())
                    log_dict = preprocess_log_dict(log_dict)
                    log_dicts.append(log_dict)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")

    return log_dicts
