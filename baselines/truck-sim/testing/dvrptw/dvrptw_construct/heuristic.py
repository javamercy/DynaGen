import numpy as np


# Common backbone idea: Multi-factor priority system combining distance, urgency, and strategic decision elements.

def select_next_node(cur_truck_pos, depot_pos, all_truck_pos, unvisited_customers, current_time, truck_speed,
                     deadlines):
    """{This algorithm employs a score-based system focusing on the integration of adaptive waiting time, distance optimization, and customer clustering to dynamically direct trucks while incorporating strategic non-movement decisions.}"""

    def calculate_distance(node_pos):
        return np.linalg.norm(cur_truck_pos - node_pos)

    def calculate_urgency_score(deadline, travel_time):
        urgency = max(0, 1 - ((current_time + travel_time) / deadline))
        return urgency

    best_score = -float('inf')
    best_node = None

    clustering_factor = 1.2
    distance_weight = 0.7
    urgency_weight = 2.0

    distances_to_customers = [calculate_distance(node_pos) for node_pos in unvisited_customers]
    time_to_customers = [distance / truck_speed for distance in distances_to_customers]

    for idx, (node_pos, deadline, time_to_customer) in enumerate(
            zip(unvisited_customers, deadlines, time_to_customers)):
        distance_score = distances_to_customers[idx]
        urgency_score = calculate_urgency_score(deadline, time_to_customer)

        # Cluster adjustment: weigh by closeness to other trucks
        cluster_adjustment = np.mean([
            np.linalg.norm(other_truck - node_pos) - distance_score
            for other_truck in all_truck_pos
        ]) * clustering_factor

        total_score = (distance_weight / (distance_score + 1e-5) +
                       urgency_weight * urgency_score +
                       cluster_adjustment)

        if total_score > best_score:
            best_score = total_score
            best_node = idx

    return best_node
