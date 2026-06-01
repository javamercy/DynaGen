import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [distance_matrix[0, 0] * 2 for _ in range(truck_count)]  # actually 0 for empty
    for i in range(truck_count):
        route_dists[i] = 0.0  # route [0,0] has distance 0
    
    def route_distance(route):
        if len(route) == 2:
            return 0.0
        dist = 0.0
        for j in range(len(route)-1):
            dist += distance_matrix[route[j], route[j+1]]
        return dist
    
    # insertion for each customer
    for cust in customers:
        best_max = float('inf')
        best_delta = float('inf')
        best_route = -1
        best_pos = -1
        current_max = max(route_dists) if route_dists else 0.0
        for r_idx in range(truck_count):
            route = routes[r_idx]
            old_dist = route_dists[r_idx]
            # compute current max excluding this route (other routes unchanged)
            other_max = max([route_dists[j] for j in range(truck_count) if j != r_idx] or [0.0])
            # try inserting at each position (1 to len(route)-1, i.e., before depot)
            for pos in range(1, len(route)):
                # compute delta
                a = route[pos-1]
                b = route[pos] if pos < len(route) else 0
                delta = (distance_matrix[a, cust] + distance_matrix[cust, b] - distance_matrix[a, b])
                new_dist = old_dist + delta
                candidate_max = max(other_max, new_dist)
                if candidate_max < best_max or (candidate_max == best_max and delta < best_delta):
                    best_max = candidate_max
                    best_delta = delta
                    best_route = r_idx
                    best_pos = pos
                elif candidate_max == best_max and delta == best_delta:
                    if r_idx < best_route or (r_idx == best_route and pos < best_pos):
                        best_max = candidate_max
                        best_delta = delta
                        best_route = r_idx
                        best_pos = pos
        # insert customer at best position
        routes[best_route].insert(best_pos, cust)
        route_dists[best_route] += best_delta
    
    # update route_dists to be accurate (should be, but just in case)
    for i in range(truck_count):
        route_dists[i] = route_distance(routes[i])
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    
    # Improvement: relocate from the longest route
    max_iter = n * 2
    for _ in range(max_iter):
        # find route with max distance
        max_dist = max(route_dists)
        max_route_idx = route_dists.index(max_dist)
        max_route = routes[max_route_idx]
        if len(max_route) <= 2:  # only depot
            break
        improved = False
        # iterate over customers in max_route (exclude depot)
        for cust in max_route[1:-1]:  # actually loop over indices, but careful: we will remove and insert
            # we need the list of customers in order; we can iterate over copies
            pass
        # better: iterate over customer indices in the route
        cust_indices = list(range(1, len(max_route)-1))
        for idx in cust_indices:
            cust = max_route[idx]
            old_dist = route_dists[max_route_idx]
            # compute delta if removed
            a = max_route[idx-1]
            b = max_route[idx+1]
            removal_delta = distance_matrix[a, cust] + distance_matrix[cust, b] - distance_matrix[a, b]
            new_dist_removed = old_dist - removal_delta
            other_max_without = max([route_dists[j] for j in range(truck_count) if j != max_route_idx] or [0.0])
            # try inserting into other routes
            for r_idx in range(truck_count):
                if r_idx == max_route_idx:
                    continue
                route = routes[r_idx]
                old_route_dist = route_dists[r_idx]
                other_max_without_this = max([route_dists[j] for j in range(truck_count) if j != max_route_idx and j != r_idx] or [0.0])
                for pos in range(1, len(route)):
                    a2 = route[pos-1]
                    b2 = route[pos] if pos < len(route) else 0
                    insertion_delta = distance_matrix[a2, cust] + distance_matrix[cust, b2] - distance_matrix[a2, b2]
                    new_other_dist = old_route_dist + insertion_delta
                    candidate_max = max(new_dist_removed, new_other_dist, other_max_without_this)
                    if candidate_max < best_max:
                        # perform move
                        # remove cust from max_route
                        del max_route[idx]
                        # insert into route r_idx at pos
                        route.insert(pos, cust)
                        # update distances
                        route_dists[max_route_idx] = new_dist_removed
                        route_dists[r_idx] = new_other_dist
                        best_max = candidate_max
                        improved = True
                        # update best_routes
                        best_routes = [list(r) for r in routes]
                        # report
                        from vrp_report import report_best_vrp
                        report_best_vrp(best_routes)
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    # final check: ensure all routes start and end at 0
    for r in routes:
        if r[0] != 0 or r[-1] != 0:
            r[0] = 0
            r[-1] = 0
    return [list(r) for r in routes]