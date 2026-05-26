import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b, axis=-1)

    current_pos = current_position
    depot = depot_position
    other_trucks = truck_positions
    current_truck_return = dist(current_pos, depot)
    other_returns = dist(other_trucks, depot)
    current_max_return = max(current_truck_return, np.max(other_returns))

    best_idx = None
    best_new_max = float('inf')
    best_new_return = float('inf')
    best_remoteness = -float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = dist(current_pos, customer)
        d_cust_depot = dist(customer, depot)
        new_return = d_truck_cust + d_cust_depot
        new_max = max(new_return, np.max(other_returns))
        # Remoteness: min distance from customer to any truck (including current)
        all_trucks = np.vstack((current_pos.reshape(1,2), other_trucks))
        remoteness = np.min(dist(all_trucks, customer))

        if new_max < best_new_max or (
            new_max == best_new_max and (
                new_return < best_new_return or (
                    new_return == best_new_return and remoteness > best_remoteness
                )
            )
        ):
            best_new_max = new_max
            best_new_return = new_return
            best_remoteness = remoteness
            best_idx = i

    # Wait if serving any customer would increase max return by more than a dynamic threshold
    # Threshold: median distance from current to available customers
    distances_to_avail = dist(current_pos, available_customers)
    threshold = np.median(distances_to_avail) if len(distances_to_avail) > 0 else 0.0
    if best_new_max > current_max_return + threshold:
        return None
    return best_idx