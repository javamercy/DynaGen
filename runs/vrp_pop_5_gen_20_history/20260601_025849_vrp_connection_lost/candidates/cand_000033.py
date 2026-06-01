import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    # Build giant tour by sorting customers by distance to depot descending
    # Compute distances from depot
    depot_dists = distance_matrix[0, 1:]
    # Create list of (distance, customer) pairs, sort descending by distance, then ascending by customer index
    sorted_cust = sorted(customers, key=lambda c: (-depot_dists[c-1], c))
    tour = sorted_cust[:]
    
    # Precompute distances for segments of the tour
    def route_dist(cust_list):
        if not cust_list:
            return 0.0
        d = distance_matrix[0, cust_list[0]]
        for i in range(len(cust_list)-1):
            d += distance_matrix[cust_list[i], cust_list[i+1]]
        d += distance_matrix[cust_list[-1], 0]
        return d
    
    full_dist = route_dist(tour)
    low = 0.0
    high = full_dist
    eps = 1e-9
    
    def feasible(L):
        seg_count = 0
        i = 0
        while i < len(tour):
            seg_count += 1
            if seg_count > truck_count:
                return False
            j = i
            first = tour[j]
            seg_dist = distance_matrix[0, first]
            while j < len(tour):
                if j == i:
                    pass
                else:
                    prev = tour[j-1]
                    curr = tour[j]
                    add = distance_matrix[prev, curr]
                    if seg_dist + add > L + eps:
                        break
                    seg_dist += add
                back_to_depot = distance_matrix[tour[j], 0]
                if seg_dist + back_to_depot > L + eps:
                    if j == i:
                        return False
                    else:
                        break
                j += 1
            if j == i:
                return False
            i = j
        return seg_count <= truck_count
    
    for _ in range(60):
        mid = (low + high) / 2
        if feasible(mid):
            high = mid
        else:
            low = mid
    L_min = high
    
    # Reconstruct routes using L_min greedy
    routes = []
    i = 0
    while i < len(tour):
        j = i
        first = tour[j]
        seg_dist = distance_matrix[0, first]
        while j < len(tour):
            if j == i:
                pass
            else:
                prev = tour[j-1]
                curr = tour[j]
                add = distance_matrix[prev, curr]
                if seg_dist + add > L_min + eps:
                    break
                seg_dist += add
            back = distance_matrix[tour[j], 0]
            if seg_dist + back > L_min + eps:
                if j == i:
                    break
                else:
                    break
            j += 1
        segment = tour[i:j]
        route = [0] + segment + [0]
        routes.append(route)
        i = j
    while len(routes) < truck_count:
        routes.append([0, 0])
    routes = routes[:truck_count]
    
    report_best_vrp(routes)
    return routes