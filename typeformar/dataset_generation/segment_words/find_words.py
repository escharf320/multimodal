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
        Tuple of words and their corresponding timestamps (word, (start_time, end_time)).
    """
    timestamped_words = []
    curr_word = ""
    start_time = 0
    end_time = 0
    log_dicts = process_logger_file_to_list(logger_path)


    for log_dict in log_dicts:
        if log_dict["keycode"] == 57 and log_dict["type"] == "down":
            end_time = log_dict["time"]
            timestamped_words.append((curr_word, (start_time, end_time)))
            curr_word = ""
        elif log_dict["keycode"] in int_to_letter and log_dict["type"] == "down":
            if curr_word == "": 
                start_time = log_dict["time"]
            curr_word += int_to_letter[log_dict["keycode"]]
        
        elif log_dict["keycode"] in backspace and log_dict["type"] == "down":
            curr_word = curr_word[:-1]
    return timestamped_words

