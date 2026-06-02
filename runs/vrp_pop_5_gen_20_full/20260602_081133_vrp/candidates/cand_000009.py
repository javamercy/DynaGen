import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(customers)

    # Regret-2 insertion (same as parent)
    while unassigned:
        best_regret = -1.0
        best_cust = None
        best_route_idx = None
        best_pos = None
        best_cost = float('inf')

        for cust in unassigned:
            costs = []
            for r_idx, route in enumerate(routes):
                if len(route) == 2:
                    cost = distance_matrix[0][cust] + distance_matrix[cust][0] - distance_matrix[0][0]
                    costs.append((cost, r_idx, 1))
                else:
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        curr = route[pos]
                        cost = distance_matrix[prev][cust] + distance_matrix[cust][curr] - distance_matrix[prev][curr]
                        costs.append((cost, r_idx, pos))
            if not costs:
                continue
            costs.sort(key=lambda x: x[0])
            best_cost_cust = costs[0][0]
            second_best_cost = costs[1][0] if len(costs) > 1 else best_cost_cust + 1e9
            regret = second_best_cost - best_cost_cust

            if regret > best_regret or (regret == best_regret and best_cost_cust < best_cost):
                best_regret = regret
                best_cust = cust
                best_cost = best_cost_cust
                best_route_idx = costs[0][1]
                best_pos = costs[0][2]

        if best_cust is None:
            break
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
        report_best_vrp(routes)

    # Adaptive improvement focusing on min-max
    n_cust = len(customers)
    max_phases = min(10, n_cust)
    best_max = float('inf')
    best_routes = [route[:] for route in routes]

    for phase in range(max_phases):
        # Compute route lengths
        lengths = []
        for route in routes:
            l = sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
            lengths.append(l)
        current_max = max(lengths)
        avg_length = sum(lengths) / truck_count
        # Dynamic threshold: start with 1.5 * avg, reduce each phase
        threshold = avg_length * (1.5 - 0.1 * phase)
        if threshold < current_max:
            threshold = current_max  # ensure we don't worsen

        # 2-opt each route (limited)
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            for _ in range(n_cust):  # bounded
                improved = False
                best_gain = 0
                best_pair = None
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        if j == i+1:
                            continue
                        old = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                        new = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                        gain = old - new
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (i, j)
                if best_gain > 0:
                    i, j = best_pair
                    routes[r_idx] = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    improved = True
                if not improved:
                    break

        # Relocate from longest routes to reduce max
        sorted_indices = sorted(range(truck_count), key=lambda i: lengths[i], reverse=True)
        for src_idx in sorted_indices:
            src_route = routes[src_idx]
            src_len = lengths[src_idx]
            if src_len <= threshold:
                continue
            # Try to move customers from this route to others
            for pos in range(1, len(src_route)-1):
                cust = src_route[pos]
                removed_cost = distance_matrix[src_route[pos-1]][cust] + distance_matrix[cust][src_route[pos+1]] - distance_matrix[src_route[pos-1]][src_route[pos+1]]
                for dest_idx in range(truck_count):
                    if dest_idx == src_idx:
                        continue
                    dest_route = routes[dest_idx]
                    dest_len = lengths[dest_idx]
                    best_insert = None
                    best_insert_cost = float('inf')
                    for ins_pos in range(1, len(dest_route)):
                        prev = dest_route[ins_pos-1]
                        curr = dest_route[ins_pos]
                        add_cost = distance_matrix[prev][cust] + distance_matrix[cust][curr] - distance_matrix[prev][curr]
                        if add_cost < best_insert_cost:
                            best_insert_cost = add_cost
                            best_insert = ins_pos
                    if best_insert is not None:
                        new_src_len = src_len - removed_cost
                        new_dest_len = dest_len + best_insert_cost
                        new_max = max(new_src_len, new_dest_len, max(lengths[:src_idx]+lengths[src_idx+1:dest_idx]+lengths[dest_idx+1:]))
                        if new_max < current_max - 1e-6:
                            # apply move
                            routes[src_idx].pop(pos)
                            routes[dest_idx].insert(best_insert, cust)
                            report_best_vrp(routes)
                            current_max = new_max
                            lengths[src_idx] = new_src_len
                            lengths[dest_idx] = new_dest_len
                            # restart scanning from this src route?
                            break
                if current_max < best_max - 1e-6:
                    best_max = current_max
                    best_routes = [route[:] for route in routes]
                if best_max < threshold:
                    break
            if best_max < threshold:
                break

        # Exchange between longest and others
        for i in range(truck_count):
            for j in range(i+1, truck_count):
                route_i = routes[i]
                route_j = routes[j]
                for pos_i in range(1, len(route_i)-1):
                    for pos_j in range(1, len(route_j)-1):
                        cust_i = route_i[pos_i]
                        cust_j = route_j[pos_j]
                        old_i = distance_matrix[route_i[pos_i-1]][cust_i] + distance_matrix[cust_i][route_i[pos_i+1]]
                        old_j = distance_matrix[route_j[pos_j-1]][cust_j] + distance_matrix[cust_j][route_j[pos_j+1]]
                        new_i = distance_matrix[route_i[pos_i-1]][cust_j] + distance_matrix[cust_j][route_i[pos_i+1]]
                        new_j = distance_matrix[route_j[pos_j-1]][cust_i] + distance_matrix[cust_i][route_j[pos_j+1]]
                        gain = (old_i + old_j) - (new_i + new_j)
                        if gain > 1e-6:
                            len_i = lengths[i]
                            len_j = lengths[j]
                            new_len_i = len_i - old_i + new_i
                            new_len_j = len_j - old_j + new_j
                            old_max = max(lengths)
                            temp_lengths = lengths[:]
                            temp_lengths[i] = new_len_i
                            temp_lengths[j] = new_len_j
                            new_max = max(temp_lengths)
                            if new_max < old_max - 1e-6:
                                # swap
                                route_i[pos_i], route_j[pos_j] = route_j[pos_j], route_i[pos_i]
                                report_best_vrp(routes)
                                lengths[i] = new_len_i
                                lengths[j] = new_len_j
                                best_max = new_max
                                best_routes = [route[:] for route in routes]

        # Check if improvement stalled
        new_max = max(lengths)
        if new_max >= current_max - 1e-6:
            # no improvement in this phase, reduce threshold further
            continue
        else:
            current_max = new_max
            best_max = new_max
            best_routes = [route[:] for route in routes]

    return best_routes