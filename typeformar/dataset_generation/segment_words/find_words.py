import sys
import os

from log_parser import process_logger_file_to_list
from .int_to_key import int_to_letter, backspace 


def words_from_log(logger_path):
    """
    Process the logger file to extract words based on space pressed events.
    
    Args:
        logger_path: Path to the logger file.
    
    Returns:
        List of words extracted from the logger file.
    """
    words = []
    curr_word = ""
    log_dicts = process_logger_file_to_list(logger_path)


    for log_dict in log_dicts:
        if log_dict["keycode"] == 57 and log_dict["type"] == "down":
            words.append(curr_word)
            curr_word = ""
        elif log_dict["keycode"] in int_to_letter and log_dict["type"] == "down":
            curr_word += int_to_letter[log_dict["keycode"]]
        
        elif log_dict["keycode"] in backspace and log_dict["type"] == "down":
            curr_word = curr_word[:-1]
    return words

