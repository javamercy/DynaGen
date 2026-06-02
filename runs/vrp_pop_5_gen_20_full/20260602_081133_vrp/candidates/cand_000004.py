import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    # Initialize routes: each starts at depot
    routes = [[0] for _ in range(truck_count)]
    assigned = [False] * n
    assigned[0] = True
    unassigned = list(range(1, n))

    def compute_distance(route):
        if len(route) <= 1:
            return 0.0
        total = 0.0
        for i in range(len(route) - 1):
            total += distance_matrix[route[i]][route[i+1]]
        total += distance_matrix[route[-1]][0]
        return total

    # Initial assignment
    while unassigned:
        # Find truck with smallest current distance
        best_truck = 0
        min_dist = compute_distance(routes[0])
        for t in range(1, truck_count):
            d = compute_distance(routes[t])
            if d < min_dist - 1e-9:
                min_dist = d
                best_truck = t
        # Find nearest unassigned customer for this truck
        last = routes[best_truck][-1]
        best_cust = None
        best_add = float('inf')
        for cust in unassigned:
            add = distance_matrix[last][cust] + distance_matrix[cust][0] - distance_matrix[last][0]
            if add < best_add - 1e-9:
                best_add = add
                best_cust = cust
        # Assign
        routes[best_truck].append(best_cust)
        unassigned.remove(best_cust)

    # Close all routes with depot
    for t in range(truck_count):
        routes[t].append(0)

    distances = [compute_distance(r) for r in routes]
    max_dist = max(distances)
    # Assuming report_best_vrp is available (imported or built-in)
    try:
        report_best_vrp(routes)
    except:
        pass

    # Improvement
    max_iter = n * truck_count * 10
    for _ in range(max_iter):
        improved = False

        # Intra-route 2-opt
        for t in range(truck_count):
            route = routes[t]
            if len(route) <= 3:
                continue
            best_dist = compute_distance(route)
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = compute_distance(new_route)
                    if new_dist < best_dist - 1e-9:
                        routes[t] = new_route
                        best_dist = new_dist
                        improved = True
        if improved:
            distances = [compute_distance(r) for r in routes]
            try:
                report_best_vrp(routes)
            except:
                pass
            max_dist = max(distances)
            continue

        # Inter-route move from longest route
        max_idx = max(range(truck_count), key=lambda t: distances[t])
        max_dist = distances[max_idx]
        best_move = None
        best_reduction = 0.0
        for cust_idx in range(1, len(routes[max_idx]) - 1):
            cust = routes[max_idx][cust_idx]
            new_route_long = routes[max_idx][:cust_idx] + routes[max_idx][cust_idx+1:]
            for t in range(truck_count):
                if t == max_idx:
                    continue
                other_route = routes[t]
                for pos in range(1, len(other_route)):
                    new_route_other = other_route[:pos] + [cust] + other_route[pos:]
                    new_dist_long = compute_distance(new_route_long)
                    new_dist_other = compute_distance(new_route_other)
                    other_distances = [distances[i] for i in range(truck_count) if i not in (max_idx, t)]
                    new_max = max(new_dist_long, new_dist_other, *other_distances)
                    reduction = max_dist - new_max
                    if reduction > best_reduction + 1e-9:
                        best_reduction = reduction
                        best_move = (max_idx, t, new_route_long, new_route_other)
        if best_reduction > 1e-9:
            max_idx, t, new_route_long, new_route_other = best_move
            routes[max_idx] = new_route_long
            routes[t] = new_route_other
            distances = [compute_distance(r) for r in routes]
            try:
                report_best_vrp(routes)
            except:
                pass
            max_dist = max(distances)
            improved = True
            continue

        break

    return routes