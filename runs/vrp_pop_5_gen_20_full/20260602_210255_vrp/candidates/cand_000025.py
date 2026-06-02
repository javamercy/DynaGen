import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    
    def compute_route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d
    
    # Two construction orders: descending distance (ascending index tie), ascending distance (ascending index tie)
    orders = [
        lambda c: (-distance_matrix[0][c], c),
        lambda c: (distance_matrix[0][c], c)
    ]
    
    best_routes = None
    best_max = float('inf')
    
    for order in orders:
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        unassigned.sort(key=order)
        route_dists = [compute_route_dist(r) for r in routes]
        
        # Greedy min-max insertion
        for cust in unassigned:
            best_new_max = float('inf')
            best_route_idx = -1
            best_pos = -1
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    succ = route[pos]
                    increase = distance_matrix[prev][cust] + distance_matrix[cust][succ] - distance_matrix[prev][succ]
                    new_route_dist = route_dists[r_idx] + increase
                    new_max = new_route_dist
                    for other_idx, d in enumerate(route_dists):
                        if other_idx != r_idx and d > new_max:
                            new_max = d
                    if new_max < best_new_max - 1e-12 or (abs(new_max - best_new_max) < 1e-12 and r_idx < best_route_idx):
                        best_new_max = new_max
                        best_route_idx = r_idx
                        best_pos = pos
            route = routes[best_route_idx]
            route.insert(best_pos, cust)
            route_dists[best_route_idx] = compute_route_dist(route)
        
        # Initial best for this start
        current_max = max(route_dists)
        current_routes = [list(r) for r in routes]
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = current_routes
        report_best_vrp(current_routes)
        
        # Improvement: relocate largest contribution from longest route
        max_iter = n * truck_count
        for _ in range(max_iter):
            current_max = max(route_dists)
            max_route_indices = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
            improved = False
            for r_idx in max_route_indices:
                route = routes[r_idx]
                if len(route) <= 2:
                    continue
                # Find customer with largest removal contribution
                best_contrib = -1.0
                best_pos = -1
                best_cust = -1
                for pos in range(1, len(route)-1):
                    cust = route[pos]
                    prev = route[pos-1]
                    succ = route[pos+1]
                    contrib = distance_matrix[prev][cust] + distance_matrix[cust][succ] - distance_matrix[prev][succ]
                    if contrib > best_contrib + 1e-12:
                        best_contrib = contrib
                        best_pos = pos
                        best_cust = cust
                if best_cust == -1:
                    continue
                # Try inserting into other routes
                for other_idx in range(truck_count):
                    if other_idx == r_idx:
                        continue
                    other_route = routes[other_idx]
                    for insert_pos in range(1, len(other_route)):
                        prev2 = other_route[insert_pos-1]
                        succ2 = other_route[insert_pos]
                        increase = distance_matrix[prev2][best_cust] + distance_matrix[best_cust][succ2] - distance_matrix[prev2][succ2]
                        new_dist_r = route_dists[r_idx] - best_contrib
                        new_dist_other = route_dists[other_idx] + increase
                        new_max = max(new_dist_r, new_dist_other)
                        for idx, d in enumerate(route_dists):
                            if idx != r_idx and idx != other_idx and d > new_max:
                                new_max = d
                        if new_max < current_max - 1e-12:
                            # Perform move
                            routes[r_idx].pop(best_pos)
                            route_dists[r_idx] = new_dist_r
                            routes[other_idx].insert(insert_pos, best_cust)
                            route_dists[other_idx] = new_dist_other
                            report_best_vrp(routes)
                            current_max = new_max
                            # Update best if applicable
                            if new_max < best_max - 1e-12:
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    return best_routes