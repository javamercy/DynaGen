import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customer_count = n - 1
    if truck_count >= customer_count:
        routes = [[0, 0] for _ in range(truck_count)]
        for i in range(1, n):
            routes[i-1] = [0, i, 0]
        try:
            report_best_vrp(routes)
        except:
            pass
        return routes

    def compute_distance(route):
        if len(route) <= 2:
            return 0.0
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total

    def try_insert(route, cust, pos):
        return route[:pos] + [cust] + route[pos:]

    # Construction: greedy minimax insertion
    routes = [[0, 0] for _ in range(truck_count)]
    distances = [0.0] * truck_count
    customers = list(range(1, n))
    for cust in customers:
        best_max = float('inf')
        best_truck = -1
        best_pos = -1
        for t in range(truck_count):
            route = routes[t]
            # possible insertion positions: 1 to len(route)-1 (between existing nodes)
            for pos in range(1, len(route)):
                new_route = try_insert(route, cust, pos)
                new_dist = compute_distance(new_route)
                other_max = max([distances[i] for i in range(truck_count) if i != t])
                new_max = max(new_dist, other_max)
                if (new_max < best_max - 1e-9) or (abs(new_max - best_max) < 1e-9 and (best_truck == -1 or t < best_truck or (t == best_truck and pos < best_pos))):
                    best_max = new_max
                    best_truck = t
                    best_pos = pos
        routes[best_truck] = try_insert(routes[best_truck], cust, best_pos)
        distances[best_truck] = compute_distance(routes[best_truck])
    try:
        report_best_vrp(routes)
    except:
        pass

    # Local search: focus on reducing max distance
    max_iter = 50
    for iteration in range(max_iter):
        max_dist = max(distances)
        if max_dist == 0:
            break
        max_idx = distances.index(max_dist)
        route = routes[max_idx]
        # Intra-route 2-opt on longest route
        best_route = route
        best_dist = max_dist
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_dist = compute_distance(new_route)
                if new_dist < best_dist - 1e-9:
                    best_dist = new_dist
                    best_route = new_route
        if best_dist < max_dist - 1e-9:
            routes[max_idx] = best_route
            distances[max_idx] = best_dist
            try:
                report_best_vrp(routes)
            except:
                pass
            continue
        # Inter-route relocate from longest route
        best_move = None
        best_new_max = max_dist
        for idx_cust in range(1, len(route)-1):
            cust = route[idx_cust]
            new_src = route[:idx_cust] + route[idx_cust+1:]
            new_src_dist = compute_distance(new_src)
            for t in range(truck_count):
                if t == max_idx:
                    continue
                dest_route = routes[t]
                for pos in range(1, len(dest_route)):
                    new_dest = try_insert(dest_route, cust, pos)
                    new_dest_dist = compute_distance(new_dest)
                    other_max = max([distances[i] for i in range(truck_count) if i not in (max_idx, t)])
                    candidate_max = max(new_src_dist, new_dest_dist, other_max)
                    if candidate_max < best_new_max - 1e-9:
                        best_new_max = candidate_max
                        best_move = (max_idx, t, idx_cust, pos, new_src, new_dest)
        if best_move is not None:
            src, t, idx_cust, pos, new_src, new_dest = best_move
            routes[src] = new_src
            routes[t] = new_dest
            distances[src] = compute_distance(new_src)
            distances[t] = compute_distance(new_dest)
            try:
                report_best_vrp(routes)
            except:
                pass
            continue
        break  # no improvement
    return routes