import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # Initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0 for _ in range(truck_count)]
    for r in range(truck_count):
        route_dists[r] = distance_matrix[0, 0] * 2  # actually 0
    
    def compute_route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    for r in range(truck_count):
        route_dists[r] = compute_route_dist(routes[r])
    
    unassigned = set(range(1, n))
    
    def best_insertion(customer):
        best_val = float('inf')
        best_pos = None
        second_val = float('inf')
        for r_idx, route in enumerate(routes):
            cur_dist = route_dists[r_idx]
            for i in range(1, len(route)):
                new_dist = cur_dist - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], customer] + distance_matrix[customer, route[i]]
                other_max = max(route_dists[:r_idx] + route_dists[r_idx+1:], default=0.0)
                cand_max = max(new_dist, other_max)
                if cand_max < best_val:
                    second_val = best_val
                    best_val = cand_max
                    best_pos = (r_idx, i)
                elif cand_max < second_val and cand_max != best_val:
                    second_val = cand_max
        return best_val, second_val, best_pos
    
    # Construction: regret-2 insertion
    while unassigned:
        regrets = []
        for c in unassigned:
            best_val, second_val, _ = best_insertion(c)
            regret = second_val - best_val if second_val != float('inf') else 0.0
            regrets.append((regret, c, best_val, second_val))
        regrets.sort(key=lambda x: (-x[0], x[1]))
        selected = regrets[0][1]
        best_val, _, best_pos = best_insertion(selected)
        r_idx, i = best_pos
        route = routes[r_idx]
        route.insert(i, selected)
        route_dists[r_idx] = compute_route_dist(route)
        unassigned.remove(selected)
    
    current_routes = [list(r) for r in routes]
    current_max = max(route_dists)
    
    def report_best_vrp(routes):
        pass
    
    # Improvement: try to reduce longest route
    max_iters = n * n
    it = 0
    improved = True
    while improved and it < max_iters:
        improved = False
        it += 1
        # Find longest route
        max_dist = max(route_dists)
        if max_dist <= 0:
            break
        # Pick the first route with max distance
        longest_idx = route_dists.index(max_dist)
        longest_route = current_routes[longest_idx]
        # Try relocating each customer in longest route except depot
        for ci in range(1, len(longest_route)-1):
            customer = longest_route[ci]
            # Compute new route without customer
            new_long_route = longest_route[:ci] + longest_route[ci+1:]
            new_long_dist = compute_route_dist(new_long_route)
            # Find best route to insert into (including other routes and maybe same route? Better to try others)
            best_new_max = current_max
            best_target = None
            best_pos = None
            for r_idx, route in enumerate(current_routes):
                if r_idx == longest_idx:
                    continue
                cur_dist = route_dists[r_idx]
                for i in range(1, len(route)):
                    new_dist = cur_dist - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], customer] + distance_matrix[customer, route[i]]
                    new_max = max(new_long_dist, new_dist, max(route_dists[:r_idx] + route_dists[r_idx+1:], default=0.0))
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_target = (r_idx, i)
            if best_target is not None and best_new_max < current_max:
                # Perform relocation
                r_idx, i = best_target
                # Remove from longest route
                current_routes[longest_idx] = new_long_route
                route_dists[longest_idx] = new_long_dist
                # Insert into target route
                route = current_routes[r_idx]
                route.insert(i, customer)
                route_dists[r_idx] = compute_route_dist(route)
                current_max = max(route_dists)
                improved = True
                report_best_vrp(current_routes)
                break
        if improved:
            continue
        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            best_route = list(route)
            best_dist = route_dists[r_idx]
            local_impr = True
            local_it = 0
            while local_impr and local_it < len(route) * 10:
                local_impr = False
                local_it += 1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_dist(new_route)
                        if new_dist < best_dist:
                            best_dist = new_dist
                            best_route = new_route
                            local_impr = True
                            new_max = max(max(route_dists[:r_idx] + route_dists[r_idx+1:], default=0.0), best_dist)
                            if new_max < current_max:
                                current_max = new_max
                                report_best_vrp(current_routes)
                            break
                    if local_impr:
                        break
                if local_impr:
                    route = best_route
            current_routes[r_idx] = best_route
            route_dists[r_idx] = best_dist
        # Inter-route swap
        for r1 in range(truck_count):
            for r2 in range(r1+1, truck_count):
                route1 = current_routes[r1]
                route2 = current_routes[r2]
                for i in range(1, len(route1)-1):
                    for j in range(1, len(route2)-1):
                        new1 = route1[:i] + [route2[j]] + route1[i+1:]
                        new2 = route2[:j] + [route1[i]] + route2[j+1:]
                        new_dist1 = compute_route_dist(new1)
                        new_dist2 = compute_route_dist(new2)
                        new_max = max(max(route_dists[:r1] + route_dists[r1+1:r2] + route_dists[r2+1:], default=0.0), new_dist1, new_dist2)
                        if new_max < current_max:
                            current_routes[r1] = new1
                            current_routes[r2] = new2
                            route_dists[r1] = new_dist1
                            route_dists[r2] = new_dist2
                            current_max = new_max
                            improved = True
                            report_best_vrp(current_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return current_routes