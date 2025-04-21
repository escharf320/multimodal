
def distance_from_space(truths:list):
    '''
    Input: list of true/false values. True indicates a keypress and False indicates no keypress.
    Output: list of distances from space for each keypress.
    '''
    distances = []

    distances = [float('inf')] * len(truths)
    memo = {} 

    for i in range(len(truths)):
        if truths[i]:  
            distances[i] = 0  
            memo[i] = 0  

    # Forward pass
    for i in range(1, len(truths)):
        if distances[i] != 0:
            distances[i] = min(distances[i], distances[i - 1] + 1)

    # Backward pass
    for i in range(len(truths) - 2, -1, -1):
        if distances[i] != 0: 
            distances[i] = min(distances[i], distances[i + 1] + 1)

    return distances


def distance_from_non_space(truths:list):
    '''
    Input: list of true/false values. True indicates a keypress and False indicates no keypress.
    Output: list of distances from non-space for each keypress.
    '''
    distances = []

    distances = [float('inf')] * len(truths)
    memo = {} 

    for i in range(len(truths)):
        if not truths[i]:  
            distances[i] = 0  
            memo[i] = 0  

    # Forward pass
    for i in range(1, len(truths)):
        if distances[i] != 0:
            distances[i] = min(distances[i], distances[i - 1] + 1)

    # Backward pass
    for i in range(len(truths) - 2, -1, -1):
        if distances[i] != 0: 
            distances[i] = min(distances[i], distances[i + 1] + 1)

    return distances



#### TESTING ####
# It's pretty easy to combine these two functions into one, but it might be nice to
# keep them separate for naming purposes 

truths = [True, False, False, True, False, False, True, False, False, False]
distances = distance_from_space(truths)
print(distances) 


distances = distance_from_non_space(truths)
print(distances)