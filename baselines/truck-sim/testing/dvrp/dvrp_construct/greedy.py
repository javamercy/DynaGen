import numpy as np

def select_next_node(current, destination, trucks, unvisited):
    if len(unvisited) == 0:
        return None

    res = np.argmin(np.linalg.norm(unvisited - current, axis=1))
    if res != 0:
        print("Error with unvisited list")
    # this will just always be 0
    return res
