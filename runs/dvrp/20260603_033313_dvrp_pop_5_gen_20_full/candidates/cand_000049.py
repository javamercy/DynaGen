import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Identify current truck index
    current_idx = None
    for idx, pos in enumerate(truck_positions):
        if np.linalg.norm(pos - current_position) < 1e-6:
            current_idx = idx
            break
    # Safety: if not found, assume first? but should not happen.
    if current_idx is None:
        current_idx = 0
    # Precompute distances from each truck to depot
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Max return time of other trucks
    max_other_ret = np.max(truck_to_depot[np.arange(len(truck_positions)) != current_idx])
    # For each customer, compute cost for current truck and min alternative
    best_idx = None
    best_advantage = -np.inf
    best_cost_now = np.inf
    # First pass: check if any customer has cost_now <= max_other_ret
    candidates = []
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Compute costs for all trucks
        all_costs = [np.linalg.norm(truck - cust) + np.linalg.norm(cust - depot_position) for truck in truck_positions]
        # Exclude current truck
        other_costs = [all_costs[j] for j in range(len(all_costs)) if j != current_idx]
        min_other = min(other_costs) if len(other_costs) > 0 else np.inf
        advantage = min_other - cost_now
        if cost_now <= max_other_ret + 1e-9:
            candidates.append((i, cost_now, advantage))
    if candidates:
        # Among candidates, pick largest advantage, tie smallest cost_now
        best_idx = max(candidates, key=lambda x: (x[2], -x[1]))[0]
    else:
        # All customers would increase max. If at depot, consider waiting
        if np.linalg.norm(current_position - depot_position) < 1e-6:
            # Check if any customer has advantage >= 0 (i.e., current is best)
            best_adv_overall = -np.inf
            best_idx_temp = None
            for i, cust in enumerate(available_customers):
                cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
                all_costs = [np.linalg.norm(truck - cust) + np.linalg.norm(cust - depot_position) for truck in truck_positions]
                other_costs = [all_costs[j] for j in range(len(all_costs)) if j != current_idx]
                min_other = min(other_costs) if len(other_costs) > 0 else np.inf
                advantage = min_other - cost_now
                if advantage > best_adv_overall or (advantage == best_adv_overall and cost_now < best_cost_now):
                    best_adv_overall = advantage
                    best_idx_temp = i
                    best_cost_now = cost_now
            if best_adv_overall < 0:
                # Not best for any customer; wait
                return None
            else:
                best_idx = best_idx_temp
        else:
            # Not at depot; must serve the customer with smallest cost_now to minimize damage
            best_idx = min(range(len(available_customers)), key=lambda i: np.linalg.norm(current_position - available_customers[i]) + np.linalg.norm(available_customers[i] - depot_position))
    return best_idx