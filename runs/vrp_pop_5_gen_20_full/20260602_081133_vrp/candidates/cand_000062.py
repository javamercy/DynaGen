import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0] * truck_count

    # Regret-based insertion
    unassigned = set(customers)
    while unassigned:
        best_cust = None
        best_regret = -1.0
        best_best_cost = float('inf')
        best_insert = None
        for cust in list(unassigned):
            # For each customer, compute best and second best insertion cost (min new max distance)
            best_cost = float('inf')
            second_best = float('inf')
            best_route = None
            best_pos = None
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    old_d = route_distances[r_idx]
                    removed = distance_matrix[route[pos-1]][route[pos]]
                    added = distance_matrix[route[pos-1]][cust] + distance_matrix[cust][route[pos]]
                    new_d = old_d - removed + added
                    other_max = max(route_distances[j] for j in range(truck_count) if j != r_idx) if truck_count > 1 else 0.0
                    new_max = max(new_d, other_max)
                    if new_max < best_cost:
                        second_best = best_cost
                        best_cost = new_max
                        best_route = r_idx
                        best_pos = pos
                    elif new_max < second_best and new_max != best_cost:
                        second_best = new_max
            if second_best == float('inf'):
                second_best = best_cost  # only one option
            regret = second_best - best_cost
            # Tie-breaking
            if regret > best_regret or (regret == best_regret and best_cost < best_best_cost):
                best_regret = regret
                best_best_cost = best_cost
                best_cust = cust
                best_insert = (best_route, best_pos, best_cost)
        # Insert best customer
        r_idx, pos, new_cost = best_insert
        routes[r_idx].insert(pos, best_cust)
        route_distances[r_idx] = new_cost
        unassigned.remove(best_cust)
    report_best_vrp(routes)

    # Local search
    # Intra-route 2-opt for each route
    for r_idx in range(truck_count):
        route = routes[r_idx]
        improved = True
        max_iter = len(route) * len(route)  # bounded
        iter_cnt = 0
        while improved and iter_cnt < max_iter:
            improved = False
            iter_cnt += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # reverse segment i..j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    # compute new distance
                    new_dist = 0.0
                    for k in range(len(new_route)-1):
                        new_dist += distance_matrix[new_route[k]][new_route[k+1]]
                    if new_dist < route_distances[r_idx]:
                        routes[r_idx] = new_route
                        route_distances[r_idx] = new_dist
                        improved = True
                        report_best_vrp(routes)
                        break
                if improved:
                    break

    # Inter-route relocate: focus on route with max distance
    for _ in range(truck_count * n):  # bounded outer loop
        # Find route with max distance
        max_idx = max(range(truck_count), key=lambda i: route_distances[i])
        max_dist = route_distances[max_idx]
        # Try to move a customer from max route to another route
        moved = False
        for cust in routes[max_idx][1:-1]:  # skip depots
            for r_idx in range(truck_count):
                if r_idx == max_idx:
                    continue
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    # Remove cust from its current route
                    old_route_max = routes[max_idx].copy()
                    old_route_max.remove(cust)
                    new_dist_max = 0.0
                    for k in range(len(old_route_max)-1):
                        new_dist_max += distance_matrix[old_route_max[k]][old_route_max[k+1]]
                    # Insert cust into other route
                    new_route_other = route.copy()
                    new_route_other.insert(pos, cust)
                    new_dist_other = 0.0
                    for k in range(len(new_route_other)-1):
                        new_dist_other += distance_matrix[new_route_other[k]][new_route_other[k+1]]
                    # New max distance
                    other_maxes = [route_distances[i] for i in range(truck_count) if i != max_idx and i != r_idx]
                    new_max = max(new_dist_max, new_dist_other, max(other_maxes) if other_maxes else 0.0)
                    if new_max < max_dist:
                        # Accept move
                        routes[max_idx] = old_route_max
                        routes[r_idx] = new_route_other
                        route_distances[max_idx] = new_dist_max
                        route_distances[r_idx] = new_dist_other
                        moved = True
                        report_best_vrp(routes)
                        break
                if moved:
                    break
            if moved:
                break
        if not moved:
            break

    return routes