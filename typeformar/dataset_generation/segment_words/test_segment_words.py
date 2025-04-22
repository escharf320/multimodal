from find_words import words_from_log

#### TESTING ####
log_path= os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "d1t2.log"
)
t = words_from_log(log_path)
print(t)
