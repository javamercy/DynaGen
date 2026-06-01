import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def insert_customer(route, pos, cust):
        return route[:pos] + [cust] + route[pos:]
    
    # Greedy insertion
    for cust in customers:
        best_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        current_max = max_route_distance(routes)
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = insert_customer(route, pos, cust)
                new_route_dist = route_distance(new_route)
                other_routes = [routes[i] for i in range(truck_count) if i != r_idx]
                other_max = max(route_distance(r) for r in other_routes) if other_routes else 0.0
                new_max = max(new_route_dist, other_max)
                if new_max < best_max or (new_max == best_max and (r_idx < best_route_idx or (r_idx == best_route_idx and pos < best_pos))):
                    best_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
        routes[best_route_idx] = insert_customer(routes[best_route_idx], best_pos, cust)
    
    best_routes = [list(r) for r in routes]
    best_max = max_route_distance(best_routes)
    
    # Adaptive local search schedule
    max_rounds = n // 2
    round_no = 0
    no_improve_rounds = 0
    patience = max(1, n // 10)
    while round_no < max_rounds and no_improve_rounds < patience:
        round_no += 1
        improved = False
        
        # 2-opt on each route
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        routes[r_idx] = new_route
                        improved = True
                        current_max = max_route_distance(routes)
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                        break
                if improved:
                    break
            if improved:
                break
        
        # Relocate from longest routes
        if not improved:
            max_dist = max_route_distance(routes)
            longest_indices = [i for i, r in enumerate(routes) if route_distance(r) == max_dist]
            for r_idx in longest_indices:
                if len(routes[r_idx]) <= 3:
                    continue
                candidate_positions = list(range(1, len(routes[r_idx])-1))
                max_reloc_candidates = min(3, len(candidate_positions))
                for k in range(max_reloc_candidates):
                    pos = candidate_positions[k]
                    cust = routes[r_idx][pos]
                    for other_idx, other_route in enumerate(routes):
                        if other_idx == r_idx:
                            continue
                        for other_pos in range(1, len(other_route)):
                            new_other = insert_customer(other_route, other_pos, cust)
                            new_self = routes[r_idx][:pos] + routes[r_idx][pos+1:]
                            new_routes = list(routes)
                            new_routes[r_idx] = new_self
                            new_routes[other_idx] = new_other
                            new_max = max_route_distance(new_routes)
                            if new_max < best_max:
                                best_max = new_max
                                best_routes = [list(r) for r in new_routes]
                                routes = new_routes
                                improved = True
                                report_best_vrp(best_routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if improved:
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
    
    return best_routes