from typeformar.dataset_generation.segment_words.find_words import words_from_log
import os

#### TESTING ####
log_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "d2t1.log")
log = words_from_log(log_path)
print(log)
