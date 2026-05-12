import numpy as np


def select_next_node(cur_truck_pos, depot_pos, all_truck_pos, unvisited_customers):
    scores = []

    for customer_pos in unvisited_customers:
        distance = np.linalg.norm(cur_truck_pos - customer_pos)

        vector_to_customer = customer_pos - cur_truck_pos
        vector_to_depot = depot_pos - cur_truck_pos

        dot_product = np.dot(vector_to_customer, vector_to_depot)

        score = distance + 2 * dot_product  # Assigning a score based on distance and dot product with additional weight on the distance
        scores.append(score)

    best_node = np.argmin(scores)

    return best_node
