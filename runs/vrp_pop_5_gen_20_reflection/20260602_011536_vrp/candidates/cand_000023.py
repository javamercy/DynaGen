import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()
    
    def route_distance(route):
        if len(route) < 2:
            return 0
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))
    
    def objective(routes):
        return max(route_distance(r) for r in routes)
    
    # Build initial solution via minimax construction (same as parent)
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    while unassigned:
        best_max = float('inf')
        best_total = float('inf')
        best_node = None
        best_route = None
        best_pos = None
        for node in unassigned:
            for r in range(truck_count):
                route = routes[r]
                for pos in range(1, len(route)):
                    new_dist = 0
                    prev = route[0]
                    for k in range(1, len(route)):
                        if k == pos:
                            new_dist += dist[prev][node]
                            prev = node
                        new_dist += dist[prev][route[k]]
                        prev = route[k]
                    current_max = 0
                    for rr in range(truck_count):
                        if rr == r:
                            d = new_dist
                        else:
                            d = route_distance(routes[rr])
                        if d > current_max:
                            current_max = d
                    if current_max < best_max or (current_max == best_max and new_dist < best_total):
                        best_max = current_max
                        best_total = new_dist
                        best_node = node
                        best_route = r
                        best_pos = pos
        routes[best_route].insert(best_pos, best_node)
        unassigned.remove(best_node)
    
    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)
    report_best_vrp(best_routes)
    
    # Ruin and recreate with targeted removal
    max_iter = min(20, n * 2)
    all_customers = list(range(1, n))
    for _ in range(max_iter):
        current_routes = [list(r) for r in routes]
        # Identify routes with highest distance
        route_dists = [route_distance(r) for r in current_routes]
        # Sort routes by distance descending
        sorted_route_indices = sorted(range(truck_count), key=lambda i: route_dists[i], reverse=True)
        # Assign removal probabilities: higher for longer routes
        removal_probs = [0.0]*truck_count
        total_weight = sum(range(1, truck_count+1))
        for rank, idx in enumerate(sorted_route_indices):
            removal_probs[idx] = (truck_count - rank) / total_weight
        # Determine number of customers to remove (50% of all customers)
        remove_count = max(1, int(0.5 * (n-1)))
        # Collect customers based on weighted route selection
        removed_list = []
        candidates = list(range(1, n))
        # Remove customers one by one, preferring routes with higher removal probability
        for _ in range(remove_count):
            if not candidates:
                break
            # Choose route index based on probabilities
            chosen_route_idx = random.choices(range(truck_count), weights=removal_probs, k=1)[0]
            # From that route, pick a customer randomly
            route = current_routes[chosen_route_idx]
            # Exclude depot
            customer_opts = [c for c in route if c != 0]
            if not customer_opts:
                # fallback random route
                valid_routes = [i for i in range(truck_count) if len(current_routes[i]) > 2]
                if not valid_routes:
                    break
                chosen_route_idx = random.choice(valid_routes)
                customer_opts = [c for c in current_routes[chosen_route_idx] if c != 0]
            cust = random.choice(customer_opts)
            # Remove from route
            pos = current_routes[chosen_route_idx].index(cust)
            current_routes[chosen_route_idx] = current_routes[chosen_route_idx][:pos] + current_routes[chosen_route_idx][pos+1:]
            if len(current_routes[chosen_route_idx]) < 2:
                current_routes[chosen_route_idx] = [0, 0]
            removed_list.append(cust)
            candidates.remove(cust)
        # Reconstruct with minimax insertion (random order)
        random.shuffle(removed_list)
        unassigned = removed_list
        while unassigned:
            best_max = float('inf')
            best_total = float('inf')
            best_node = None
            best_route = None
            best_pos = None
            for node in unassigned:
                for r in range(truck_count):
                    route = current_routes[r]
                    for pos in range(1, len(route)):
                        new_dist = 0
                        prev = route[0]
                        for k in range(1, len(route)):
                            if k == pos:
                                new_dist += dist[prev][node]
                                prev = node
                            new_dist += dist[prev][route[k]]
                            prev = route[k]
                        current_max = 0
                        for rr in range(truck_count):
                            if rr == r:
                                d = new_dist
                            else:
                                d = route_distance(current_routes[rr])
                            if d > current_max:
                                current_max = d
                        if current_max < best_max or (current_max == best_max and new_dist < best_total):
                            best_max = current_max
                            best_total = new_dist
                            best_node = node
                            best_route = r
                            best_pos = pos
            current_routes[best_route].insert(best_pos, best_node)
            unassigned.remove(best_node)
        # Apply 2-opt on each route
        for r in range(truck_count):
            route = current_routes[r]
            if len(route) <= 3:
                continue
            improved = True
            local_iter = 0
            while improved and local_iter < 10:
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        old_dist = route_distance(route)
                        if new_dist < old_dist:
                            current_routes[r] = new_route
                            improved = True
                            break
                    if improved:
                        break
                local_iter += 1
        new_obj = objective(current_routes)
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)
        # Always accept new solution
        routes = [list(r) for r in current_routes]
    return best_routes