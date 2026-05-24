import numpy as np

def select_next_node(cur_truck_pos, depot_pos, all_truck_pos, unvisited_customers, current_time, truck_speed, deadlines):
    if len(unvisited_customers) == 0:
        return None

    unvisited_indices = np.array([i for i in range(len(unvisited_customers))])
    unvisited_customers = np.array(unvisited_customers)

    distances = np.linalg.norm(unvisited_customers - cur_truck_pos, axis=1)

    available_time = np.array([deadlines[i][0] for i in range(len(unvisited_customers))])
    mask = (current_time + distances) >= available_time  # filter to nodes that can be reached after they are available

    if np.all(np.invert(mask)):
        return None
    unvisited_indices = unvisited_indices[mask]
    unvisited_customers = unvisited_customers[unvisited_indices]

    res = np.argmin(np.linalg.norm(unvisited_customers - cur_truck_pos, axis=1))
    if res != 0:
        print("Error with unvisited list")

    # this will just always be 0
    return unvisited_indices[res]
