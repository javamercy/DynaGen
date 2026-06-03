def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    n_trucks = truck_positions.shape[0]
    # distances from active truck to each customer + customer to depot
    active_totals = np.linalg.norm(current_position - available_customers, axis=1) + np.linalg.norm(available_customers - depot_position, axis=1)
    # Compute best total distance among other trucks for each customer
    # If only one truck, best_other is large
    best_other = np.full(available_customers.shape[0], np.inf)
    if n_trucks > 1:
        # For each other truck, compute total distances and take min
        for i in range(n_trucks):
            if np.all(truck_positions[i] == current_position):
                continue  # skip active truck
            other_totals = np.linalg.norm(truck_positions[i] - available_customers, axis=1) + np.linalg.norm(available_customers - depot_position, axis=1)
            best_other = np.minimum(best_other, other_totals)
    # Compute ratio: active_total / best_other, handle inf by large number
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(np.isfinite(best_other), active_totals / best_other, np.inf)
    # Select customer with smallest ratio (if tie, smallest active_total)
    min_ratio_idx = np.argmin(ratios)
    # In case of ties, prefer smaller active_total
    min_ratio_val = ratios[min_ratio_idx]
    candidates = np.where(ratios == min_ratio_val)[0]
    if len(candidates) > 1:
        best_idx = candidates[np.argmin(active_totals[candidates])]
    else:
        best_idx = min_ratio_idx
    # Convert to int
    return int(best_idx)