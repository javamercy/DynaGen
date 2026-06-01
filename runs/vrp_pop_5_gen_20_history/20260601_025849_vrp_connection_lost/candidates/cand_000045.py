import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix
    
    # Initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))
    
    def route_distance(route):
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += dist[a][b]
        return d
    
    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)
    
    # Construction: regret-2 with tie-breaking by new max distance, then customer index
    while unassigned:
        candidates = []
        for cust in unassigned:
            best_cost = float('inf')
            second_best_cost = float('inf')
            best_route_idx = -1
            best_pos = -1
            for r_idx, route in enumerate(routes):
                for i in range(len(route) - 1):
                    cost = dist[route[i]][cust] + dist[cust][route[i+1]] - dist[route[i]][route[i+1]]
                    if cost < best_cost:
                        second_best_cost = best_cost
                        best_cost = cost
                        best_route_idx = r_idx
                        best_pos = i + 1
                    elif cost < second_best_cost:
                        second_best_cost = cost
            regret = second_best_cost - best_cost if second_best_cost != float('inf') else float('inf')
            # Compute new max distance if inserting here
            new_route = routes[best_route_idx][:best_pos] + [cust] + routes[best_route_idx][best_pos:]
            new_route_dist = route_distance(new_route)
            current_max = max_route_distance(routes)
            new_max = max(new_route_dist, current_max)
            candidates.append((-regret, new_max, cust, best_route_idx, best_pos))
        # Tie-break: sort by (-regret, new_max, cust) ; cust ensures determinism
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, chosen_customer, chosen_route_idx, chosen_pos = candidates[0]
        routes[chosen_route_idx].insert(chosen_pos, chosen_customer)
        unassigned.remove(chosen_customer)
    
    report_best_vrp(routes)
    best_max = max_route_distance(routes)
    
    # Improvement: bounded local search
    max_iter = n * n
    for _ in range(max_iter):
        improved = False
        current_max = max_route_distance(routes)
        # Identify longest route(s) - pick first one
        longest_route_idx = -1
        for idx, r in enumerate(routes):
            if route_distance(r) == current_max:
                longest_route_idx = idx
                break
        if longest_route_idx == -1:
            break
        route = routes[longest_route_idx]
        # Relocate customers from the longest route to other routes
        for pos in range(1, len(route)-1):
            cust = route[pos]
            for other_idx, other_route in enumerate(routes):
                if other_idx == longest_route_idx:
                    continue
                for other_pos in range(1, len(other_route)):
                    new_route = route[:pos] + route[pos+1:]
                    new_other = other_route[:other_pos] + [cust] + other_route[other_pos:]
                    new_routes = [list(r) for r in routes]
                    new_routes[longest_route_idx] = new_route
                    new_routes[other_idx] = new_other
                    new_max = max_route_distance(new_routes)
                    if new_max < best_max:
                        best_max = new_max
                        routes = new_routes
                        improved = True
                        report_best_vrp(routes)
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 2-opt on each route (only if relocate didn't improve)
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        routes[r_idx] = new_route
                        improved = True
                        current_best = max_route_distance(routes)
                        if current_best < best_max:
                            best_max = current_best
                            report_best_vrp(routes)
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    # Return exactly truck_count routes (ensure depot at start and end)
    final_routes = []
    for route in routes:
        if len(route) == 2:
            final_routes.append([0, 0])
        else:
            final_routes.append([0] + route[1:-1] + [0])
    return final_routes