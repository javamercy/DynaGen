import numpy as np
import copy

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # Initialize routes: each route is [0,0]
    routes = [[0, 0] for _ in range(truck_count)]
    # route distances
    route_dist = [0.0] * truck_count
    # customers to assign (1..n-1)
    unassigned = list(range(1, n))
    
    # Helper to compute delta if insert customer between positions i and i+1 in route
    def insertion_delta(route, pos, customer):
        # route is a list, pos is index after which to insert (0 <= pos < len(route)-1)
        # Returns new route length increase
        prev = route[pos]
        next_ = route[pos+1]
        return distance_matrix[prev, customer] + distance_matrix[customer, next_] - distance_matrix[prev, next_]
    
    # Find best insertion position in a route for a customer, return (pos, delta)
    def best_in_route(route, customer):
        best_delta = float('inf')
        best_pos = -1
        for pos in range(len(route)-1):
            delta = insertion_delta(route, pos, customer)
            if delta < best_delta:
                best_delta = delta
                best_pos = pos
        return best_pos, best_delta
    
    # Construction: regret insertion
    while unassigned:
        best_regret_cust = None
        best_regret_val = -float('inf')
        best_route_idx = -1
        best_pos = -1
        best_delta = None
        for cust in unassigned:
            # For each route, compute best insertion delta
            deltas = []
            positions = []
            for ridx, route in enumerate(routes):
                pos, delta = best_in_route(route, cust)
                deltas.append(delta)
                positions.append(pos)
            # Find best and second best delta
            sorted_deltas = sorted(deltas)
            best_delta_val = sorted_deltas[0]
            # second best; if only one route, use inf
            if len(sorted_deltas) > 1:
                second_best = sorted_deltas[1]
            else:
                second_best = float('inf')
            regret = second_best - best_delta_val
            if regret > best_regret_val or (regret == best_regret_val and cust < (best_regret_cust if best_regret_cust is not None else n)):
                best_regret_val = regret
                best_regret_cust = cust
                # Record the best route and position
                best_delta = best_delta_val
                # Find which route gives best delta (first occurrence)
                for ridx, d in enumerate(deltas):
                    if d == best_delta_val:
                        best_route_idx = ridx
                        best_pos = positions[ridx]
                        break
        # Insert the chosen customer
        route = routes[best_route_idx]
        route.insert(best_pos+1, best_regret_cust)
        route_dist[best_route_idx] += best_delta
        # Also update route_dist to actual distance? But delta is correct if no other changes.
        unassigned.remove(best_regret_cust)
    
    # After construction, compute actual route distances
    def compute_route_distance(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist
    for i, route in enumerate(routes):
        route_dist[i] = compute_route_distance(route)
    
    best_routes = copy.deepcopy(routes)
    best_max = max(route_dist)
    report_best_vrp(best_routes)
    
    # Improvement: finite loops
    max_iter = n * 5  # bounded
    for _ in range(max_iter):
        improved = False
        # Intra-route: 2-opt for each route
        for ridx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            found_improve = True
            while found_improve:
                found_improve = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        # before swap: edges (route[i-1], route[i]), (route[j], route[j+1])
                        # after: (route[i-1], route[j]), (route[i], route[j+1])
                        old_dist = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new_dist = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new_dist < old_dist - 1e-9:
                            # apply swap: reverse segment from i to j
                            route[i:j+1] = reversed(route[i:j+1])
                            route_dist[ridx] += (new_dist - old_dist)
                            found_improve = True
                            improved = True
                            break
                    if found_improve:
                        break
        # Inter-route: move customer from longest route to another
        # Find current longest route
        max_dist = max(route_dist)
        longest_idx = route_dist.index(max_dist)
        longest_route = routes[longest_idx]
        # Consider each customer in longest route (excluding depot)
        candidates = list(range(1, len(longest_route)-1))
        found_move = False
        for cust_idx in candidates:
            cust = longest_route[cust_idx]
            # Remove customer from longest route temporarily
            new_longest = longest_route[:cust_idx] + longest_route[cust_idx+1:]
            new_longest_dist = compute_route_distance(new_longest)
            # Find best insertion in other routes
            best_other_delta = float('inf')
            best_other_route = -1
            best_other_pos = -1
            for r_idx, route in enumerate(routes):
                if r_idx == longest_idx:
                    continue
                pos, delta = best_in_route(route, cust)
                if delta < best_other_delta:
                    best_other_delta = delta
                    best_other_route = r_idx
                    best_other_pos = pos
            if best_other_route == -1:
                continue
            # Evaluate new max distance
            new_other = routes[best_other_route][:]
            new_other.insert(best_other_pos+1, cust)
            new_other_dist = compute_route_distance(new_other)
            new_max = max(new_longest_dist, new_other_dist, *[route_dist[t] for t in range(truck_count) if t != longest_idx and t != best_other_route])
            if new_max < max_dist - 1e-9:
                # Accept
                routes[longest_idx] = new_longest
                routes[best_other_route] = new_other
                route_dist[longest_idx] = new_longest_dist
                route_dist[best_other_route] = new_other_dist
                improved = True
                found_move = True
                break
        if improved:
            current_max = max(route_dist)
            if current_max < best_max - 1e-9:
                best_routes = copy.deepcopy(routes)
                best_max = current_max
                report_best_vrp(best_routes)
        else:
            # If no improvement in a full iteration, break to avoid infinite loops
            if not improved:
                break
    # Ensure exactly truck_count routes
    return best_routes