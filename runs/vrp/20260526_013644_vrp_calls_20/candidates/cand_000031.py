import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0 for _ in range(truck_count)]

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def compute_max_dist(routes):
        return max(route_distance(r) for r in routes)

    def construct_solution():
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(customers)
        while unassigned:
            best_cust = None
            best_regret = -1.0
            best_route_idx = None
            best_pos = None
            best_new_max = None
            for cust in list(unassigned):
                best_cost = float('inf')
                second_best_cost = float('inf')
                best_r = None
                best_p = None
                for r in range(truck_count):
                    route = routes[r]
                    for pos in range(1, len(route)):
                        delta = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        new_dist = route_distances[r] + delta
                        temp_routes = [rr[:] for rr in routes]
                        temp_routes[r].insert(pos, cust)
                        new_max = max(new_dist, compute_max_dist(temp_routes))
                        if new_max < best_cost:
                            second_best_cost = best_cost
                            best_cost = new_max
                            best_r = r
                            best_p = pos
                        elif new_max < second_best_cost:
                            second_best_cost = new_max
                regret = second_best_cost - best_cost
                if regret > best_regret or (regret == best_regret and cust < best_cust):
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = best_r
                    best_pos = best_p
                    best_new_max = best_cost
            routes[best_route_idx].insert(best_pos, best_cust)
            route_distances[best_route_idx] = route_distance(routes[best_route_idx])
            unassigned.remove(best_cust)
        return routes

    # Initial construction
    routes = construct_solution()
    best_routes = [r[:] for r in routes]
    best_max = compute_max_dist(routes)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    current_routes = [r[:] for r in routes]
    current_max = best_max
    deviation = 0.1 * current_max
    max_iter = 200
    no_improve_iter = 0
    stagnation_count = 0

    for it in range(max_iter):
        improved = False
        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            best_i = None
            best_j = None
            best_new_max = current_max + deviation + 1
            for i in range(1, len(route)-2):
                for j in range(i+2, len(route)-1):
                    new_route = route[:i+1] + route[j:i:-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    candidate_max = new_dist
                    for rr in range(truck_count):
                        if rr != r_idx:
                            candidate_max = max(candidate_max, route_distances[rr])
                    if candidate_max < best_new_max:
                        best_new_max = candidate_max
                        best_i = i
                        best_j = j
            if best_i is not None and best_new_max <= current_max + deviation:
                new_route = current_routes[r_idx][:best_i+1] + current_routes[r_idx][best_j:best_i:-1] + current_routes[r_idx][best_j+1:]
                current_routes[r_idx] = new_route
                route_distances[r_idx] = route_distance(new_route)
                current_max = compute_max_dist(current_routes)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in current_routes]
                    try:
                        report_best_vrp(best_routes)
                    except NameError:
                        pass
                    no_improve_iter = 0
                improved = True

        # Inter-route relocate
        best_move = None
        best_new_max = current_max + deviation + 1
        for r_from in range(truck_count):
            if len(current_routes[r_from]) <= 2:
                continue
            for pos_from in range(1, len(current_routes[r_from])-1):
                cust = current_routes[r_from][pos_from]
                prev = current_routes[r_from][pos_from-1]
                next = current_routes[r_from][pos_from+1]
                delta_from = distance_matrix[prev, next] - distance_matrix[prev, cust] - distance_matrix[cust, next]
                new_from_dist = route_distances[r_from] + delta_from
                for r_to in range(truck_count):
                    if r_to == r_from:
                        continue
                    route_to = current_routes[r_to]
                    for pos_to in range(1, len(route_to)):
                        prev_to = route_to[pos_to-1]
                        next_to = route_to[pos_to]
                        delta_to = distance_matrix[prev_to, cust] + distance_matrix[cust, next_to] - distance_matrix[prev_to, next_to]
                        new_to_dist = route_distances[r_to] + delta_to
                        candidate_max = max(new_from_dist, new_to_dist)
                        for rr in range(truck_count):
                            if rr != r_from and rr != r_to:
                                candidate_max = max(candidate_max, route_distances[rr])
                        if candidate_max < best_new_max:
                            best_new_max = candidate_max
                            best_move = (r_from, pos_from, r_to, pos_to, new_from_dist, new_to_dist)
        if best_move is not None and best_new_max <= current_max + deviation:
            r_from, pos_from, r_to, pos_to, new_from_dist, new_to_dist = best_move
            cust = current_routes[r_from].pop(pos_from)
            current_routes[r_to].insert(pos_to, cust)
            route_distances[r_from] = new_from_dist
            route_distances[r_to] = new_to_dist
            current_max = compute_max_dist(current_routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
                no_improve_iter = 0
            improved = True

        # Inter-route swap
        best_move = None
        best_new_max = current_max + deviation + 1
        for r1 in range(truck_count):
            if len(current_routes[r1]) <= 2:
                continue
            for pos1 in range(1, len(current_routes[r1])-1):
                cust1 = current_routes[r1][pos1]
                prev1 = current_routes[r1][pos1-1]
                next1 = current_routes[r1][pos1+1]
                for r2 in range(r1+1, truck_count):
                    if len(current_routes[r2]) <= 2:
                        continue
                    for pos2 in range(1, len(current_routes[r2])-1):
                        cust2 = current_routes[r2][pos2]
                        prev2 = current_routes[r2][pos2-1]
                        next2 = current_routes[r2][pos2+1]
                        delta1 = distance_matrix[prev1, cust2] + distance_matrix[cust2, next1] - distance_matrix[prev1, cust1] - distance_matrix[cust1, next1]
                        new_dist1 = route_distances[r1] + delta1
                        delta2 = distance_matrix[prev2, cust1] + distance_matrix[cust1, next2] - distance_matrix[prev2, cust2] - distance_matrix[cust2, next2]
                        new_dist2 = route_distances[r2] + delta2
                        candidate_max = max(new_dist1, new_dist2)
                        for rr in range(truck_count):
                            if rr != r1 and rr != r2:
                                candidate_max = max(candidate_max, route_distances[rr])
                        if candidate_max < best_new_max:
                            best_new_max = candidate_max
                            best_move = (r1, pos1, r2, pos2, new_dist1, new_dist2)
        if best_move is not None and best_new_max <= current_max + deviation:
            r1, pos1, r2, pos2, new_dist1, new_dist2 = best_move
            cust1 = current_routes[r1][pos1]
            cust2 = current_routes[r2][pos2]
            current_routes[r1][pos1] = cust2
            current_routes[r2][pos2] = cust1
            route_distances[r1] = new_dist1
            route_distances[r2] = new_dist2
            current_max = compute_max_dist(current_routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
                no_improve_iter = 0
            improved = True

        # Diversification: every 50 iterations, if no recent improvement, destroy worst route
        if it % 50 == 0 and it > 0 and no_improve_iter >= 5:
            # Find worst route (max distance)
            worst_idx = max(range(truck_count), key=lambda i: route_distances[i])
            if len(current_routes[worst_idx]) > 2:
                # Remove all customers from worst route
                removed = current_routes[worst_idx][1:-1]
                current_routes[worst_idx] = [0, 0]
                route_distances[worst_idx] = 0.0
                # Reinsert removed customers using regret
                unassigned = list(removed)
                while unassigned:
                    best_cust = None
                    best_regret = -1.0
                    best_route_idx = None
                    best_pos = None
                    best_new_max = None
                    for cust in unassigned:
                        best_cost = float('inf')
                        second_best_cost = float('inf')
                        best_r = None
                        best_p = None
                        for r in range(truck_count):
                            route = current_routes[r]
                            for pos in range(1, len(route)):
                                delta = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                                new_dist = route_distances[r] + delta
                                temp_routes = [rr[:] for rr in current_routes]
                                temp_routes[r].insert(pos, cust)
                                new_max = max(new_dist, compute_max_dist(temp_routes))
                                if new_max < best_cost:
                                    second_best_cost = best_cost
                                    best_cost = new_max
                                    best_r = r
                                    best_p = pos
                                elif new_max < second_best_cost:
                                    second_best_cost = new_max
                        regret = second_best_cost - best_cost
                        if regret > best_regret or (regret == best_regret and cust < best_cust):
                            best_regret = regret
                            best_cust = cust
                            best_route_idx = best_r
                            best_pos = best_p
                            best_new_max = best_cost
                    current_routes[best_route_idx].insert(best_pos, best_cust)
                    route_distances[best_route_idx] = route_distance(current_routes[best_route_idx])
                    unassigned.remove(best_cust)
                current_max = compute_max_dist(current_routes)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in current_routes]
                    try:
                        report_best_vrp(best_routes)
                    except NameError:
                        pass
                    no_improve_iter = 0
                improved = True

        if not improved:
            no_improve_iter += 1
            deviation *= 0.99
        else:
            no_improve_iter = 0

        # Restart if no improvement for 20 consecutive iterations
        if no_improve_iter >= 20:
            current_routes = construct_solution()
            current_max = compute_max_dist(current_routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
            no_improve_iter = 0
            deviation = 0.1 * current_max

    return best_routes