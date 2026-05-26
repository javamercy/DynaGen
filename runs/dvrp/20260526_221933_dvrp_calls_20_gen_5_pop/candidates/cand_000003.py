import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None

    # identify index of deciding truck (closest to current_position)
    diffs = truck_positions - current_position
    dists = np.linalg.norm(diffs, axis=1)
    deciding_idx = np.argmin(dists)

    num_trucks = len(truck_positions)
    depot = depot_position

    def simulate(customer_idx):
        # routes: list of lists of coordinates (start with initial positions)
        routes = [ [truck_positions[i].copy()] for i in range(num_trucks) ]
        # unserved as list of tuples for easy removal
        unserved = [ tuple(available_customers[i]) for i in range(len(available_customers)) ]
        if customer_idx is not None:
            # remove assigned customer from unserved
            assigned_coord = tuple(available_customers[customer_idx])
            unserved.remove(assigned_coord)
            # add to deciding truck's route
            routes[deciding_idx].append(assigned_coord)

        # depth: 2 rounds of assignments
        for _ in range(2):
            if not unserved:
                break
            # iterate over trucks in fixed order (deciding truck already got one)
            for t in range(num_trucks):
                if not unserved:
                    break
                last = routes[t][-1]
                # find nearest unserved customer
                min_dist = float('inf')
                best_cust = None
                best_pos = -1
                for i, cust in enumerate(unserved):
                    d = np.linalg.norm(np.array(last) - np.array(cust))
                    if d < min_dist:
                        min_dist = d
                        best_cust = cust
                        best_pos = i
                # assign
                routes[t].append(best_cust)
                unserved.pop(best_pos)

        # compute estimated TTT
        max_time = 0.0
        for t in range(num_trucks):
            route = routes[t]
            travel = 0.0
            for i in range(len(route)-1):
                travel += np.linalg.norm(np.array(route[i]) - np.array(route[i+1]))
            travel += np.linalg.norm(np.array(route[-1]) - depot)
            finish_time = current_time + travel
            if finish_time > max_time:
                max_time = finish_time
        return max_time

    # evaluate waiting
    wait_ttt = simulate(None)
    best_ttt = wait_ttt
    best_idx = None

    # evaluate each candidate
    for i in range(len(available_customers)):
        cand_ttt = simulate(i)
        if cand_ttt < best_ttt:
            best_ttt = cand_ttt
            best_idx = i
        elif cand_ttt == best_ttt and best_idx is None:
            best_idx = i  # tie but first seen

    return best_idx