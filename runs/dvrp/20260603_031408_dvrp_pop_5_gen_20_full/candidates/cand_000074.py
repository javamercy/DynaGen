def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    best_idx = None
    best_max = float('inf')
    for i, cust in enumerate(available_customers):
        new_rem_this = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        max_other = 0.0
        first = True
        for pos in truck_positions:
            if not np.array_equal(pos, current_position):
                rem = np.linalg.norm(pos - depot_position)
                if first:
                    max_other = rem
                    first = False
                else:
                    max_other = max(max_other, rem)
        if first:
            max_rem = new_rem_this
        else:
            max_rem = max(new_rem_this, max_other)
        if max_rem < best_max:
            best_max = max_rem
            best_idx = i
    return best_idx