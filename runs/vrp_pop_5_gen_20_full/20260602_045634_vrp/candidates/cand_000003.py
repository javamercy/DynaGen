import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    all_customers = list(range(1, n))
    unassigned = set(all_customers)
    # Initialize routes: each truck has an empty route [0,0] but we treat as [0,0] with length 0
    routes = [[0, 0] for _ in range(truck_count)]
    # Represent routes as list of nodes, with depot implicit at start and end; internal representation: list of nodes without depot at ends, but we'll keep depots for simplicity
    # Actually we'll store routes as lists with 0 at start and end, but insertion positions are between.
    # For easier insertion, we maintain routes as lists of nodes between depots? We'll use full route lists.
    
    # Helper to compute route distance
    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    # While there are unassigned customers
    while unassigned:
        best_regret = -1
        best_customer = None
        best_insertion = None  # (route_idx, pos) where pos is index to insert (after that index? Better: insert at index pos, meaning before previous node at pos)
        best_best_cost = None
        best_second_best_cost = None
        
        for cust in unassigned:
            # Collect insertion costs across all routes and positions
            insertion_options = []
            for r_idx, route in enumerate(routes):
                # route is list like [0, ..., 0]. Possible positions from index 1 to len(route)-1 (insert before that index)
                for pos in range(1, len(route)):
                    # Compute added distance: remove edge route[pos-1]-route[pos], add edges route[pos-1]-cust and cust-route[pos]
                    prev = route[pos-1]
                    next_node = route[pos]
                    added = distance_matrix[prev, cust] + distance_matrix[cust, next_node] - distance_matrix[prev, next_node]
                    insertion_options.append((added, r_idx, pos))
            # Sort by added cost
            insertion_options.sort()
            if not insertion_options:
                continue
            best_cost = insertion_options[0][0]
            second_best_cost = insertion_options[1][0] if len(insertion_options) > 1 else best_cost
            regret = second_best_cost - best_cost
            # Update best if regret larger, or tie-breaking
            if (regret > best_regret or
                (regret == best_regret and (best_best_cost is None or best_cost > best_best_cost or
                                            (best_cost == best_best_cost and cust < best_customer)))):
                best_regret = regret
                best_customer = cust
                best_best_cost = best_cost
                best_second_best_cost = second_best_cost
                best_insertion = (insertion_options[0][1], insertion_options[0][2])  # route_idx, pos
        
        if best_customer is None:
            break  # should not happen
        # Insert best customer
        r_idx, pos = best_insertion
        routes[r_idx].insert(pos, best_customer)
        unassigned.remove(best_customer)
    
    # Local search: inter-route relocate to reduce max route distance
    # Compute current max distance
    def max_route_dist():
        return max(route_dist(r) for r in routes)
    
    best_routes = [r[:] for r in routes]
    best_max = max_route_dist()
    report_best_vrp(best_routes)
    
    # Limit iterations to avoid infinite loops
    n_cust = n - 1
    max_iter = n_cust * truck_count * 10  # bound
    for _ in range(max_iter):
        improved = False
        for r_idx_src in range(truck_count):
            route_src = routes[r_idx_src]
            if len(route_src) <= 2:
                continue
            # For each customer in source route (excluding depots)
            for pos_src in range(1, len(route_src)-1):
                cust = route_src[pos_src]
                for r_idx_dst in range(truck_count):
                    if r_idx_dst == r_idx_src:
                        continue
                    route_dst = routes[r_idx_dst]
                    # Try insert cust into route_dst at each possible position
                    for pos_dst in range(1, len(route_dst)):
                        # Compute new route distances
                        # Remove cust from source
                        new_src = route_src[:pos_src] + route_src[pos_src+1:]
                        # Insert in destination
                        new_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                        # Compute max of new distances
                        dist_src = route_dist(new_src)
                        dist_dst = route_dist(new_dst)
                        new_max = max(dist_src, dist_dst)
                        # Also consider other routes unchanged
                        # But we only compare to current best_max
                        if new_max < best_max:
                            # Accept move
                            routes[r_idx_src] = new_src
                            routes[r_idx_dst] = new_dst
                            best_routes = [r[:] for r in routes]
                            best_max = new_max
                            improved = True
                            report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    # Ensure exactly truck_count routes
    # Already have truck_count routes
    return best_routes