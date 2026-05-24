import numpy as np

def select_next_node(current, destination, trucks, unvisited):
    max_score = float('-inf')
    next_destination = None

    for i, coord in enumerate(unvisited):
        euclidean_dist = np.linalg.norm(current - coord)
        other_trucks_dist_sum = sum([np.linalg.norm(truck - coord) for truck in trucks])
        dist_ratio = euclidean_dist / (other_trucks_dist_sum / (len(trucks) + 1))  # New scoring function
        score = -dist_ratio  # Highest score is prioritized

        if score > max_score:
            max_score = score
            next_destination = i

    return next_destination
