import numpy as np


def select_next_node(cur_truck_pos, depot_pos, all_truck_pos, unvisited_customers, current_time, truck_speed,
                     time_windows):
    best_node = None
    min_weighted_score = float('inf')

    for idx, customer_pos in enumerate(unvisited_customers):
        travel_time = np.linalg.norm(customer_pos - cur_truck_pos) / truck_speed
        wait_time = max(0, time_windows[idx][0] - (current_time + travel_time))
        penalty = max(0, current_time + travel_time - time_windows[idx][1])

        weighted_score = 0.2 * penalty + 0.3 * wait_time + 0.5 * travel_time

        for truck_pos in all_truck_pos:
            if np.array_equal(truck_pos, customer_pos):
                weighted_score += 0.2

            other_travel_time = np.linalg.norm(customer_pos - truck_pos) / truck_speed
            max_possible_delay = max(0, (current_time + travel_time) - (current_time + other_travel_time))
            weighted_score += 0.4 * max_possible_delay

        if weighted_score < min_weighted_score:
            best_node = idx
            min_weighted_score = weighted_score

    return best_node