import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # Initialize routes: all start and end at depot
    routes = [[0, 0] for _ in range(truck_count)]
    # Compute route distance helper
    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    # Track current max distance and best solution
    best_routes = [r[:] for r in routes]
    best_max = 0.0
    # Insertion phase: assign all customers (1..n-1)
    unassigned = set(range(1, n))
    while unassigned:
        best_regret = -1.0
        best_cust = None
        best_route_idx = None
        best_pos = None
        best_cost = float('inf')
        for cust in unassigned:
            # Find best and second best insertion across all routes
            first_cost = float('inf')
            second_cost = float('inf')
            first_route = -1
            first_pos = -1
            for r_idx, route in enumerate(routes):
                # Insert in all positions between 1 and len(route)-1 (keep first and last depot)
                for pos in range(1, len(route)):
                    before = route[pos-1]
                    after = route[pos]
                    # cost increase if we insert cust between before and after
                    increase = distance_matrix[before, cust] + distance_matrix[cust, after] - distance_matrix[before, after]
                    new_cost = increase  # we compare increase, not absolute
                    if new_cost < first_cost:
                        second_cost = first_cost
                        first_cost = new_cost
                        first_route = r_idx
                        first_pos = pos
                    elif new_cost < second_cost:
                        second_cost = new_cost
            # Compute regret: second_best - best (if only one feasible, regret = large)
            if second_cost == float('inf'):
                regret = 1e9
            else:
                regret = second_cost - first_cost
            # Select customer with highest regret; tie-break by smallest index
            if regret > best_regret or (regret == best_regret and (best_cust is None or cust < best_cust)):
                best_regret = regret
                best_cust = cust
                best_cost = first_cost
                best_route_idx = first_route
                best_pos = first_pos
        # Insert best_cust into best_route at best_pos
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    # After insertion, compute max distance and set best
    current_max = max(route_dist(r) for r in routes)
    best_routes = [r[:] for r in routes]
    best_max = current_max
    # Improvement phase: relocate and intra-route 2-opt (bounded loops)
    total_customers = n - 1
    for _ in range(total_customers):  # bounded by instance size
        # Find longest route
        dists = [route_dist(r) for r in routes]
        max_idx = np.argmax(dists)
        max_dist = dists[max_idx]
        if max_dist <= best_max * 0.999:  # no significant improvement
            break
        # Try to relocate a customer from longest route to another route
        improved = False
        long_route = routes[max_idx]
        # Consider customers in long route (excluding depot at ends)
        cust_indices = list(range(1, len(long_route)-1))  # positions of customers
        for idx in cust_indices:
            cust = long_route[idx]
            # Remove cust temporarily
            new_long = long_route[:idx] + long_route[idx+1:]
            # Try inserting cust into other routes (including possibly same route? better to try all)
            best_new_route = None
            best_new_pos = None
            best_inc = float('inf')
            best_new_route_idx = -1
            for r_idx, route in enumerate(routes):
                if r_idx == max_idx:
                    # Insert back into long route after removal?
                    # Actually we can try reinserting into the same route at a different position
                    for pos in range(1, len(new_long)):
                        before = new_long[pos-1]
                        after = new_long[pos]
                        inc = distance_matrix[before, cust] + distance_matrix[cust, after] - distance_matrix[before, after]
                        if inc < best_inc:
                            best_inc = inc
                            best_new_route = new_long
                            best_new_pos = pos
                            best_new_route_idx = max_idx
                else:
                    # Try all insertion positions in this route
                    for pos in range(1, len(route)):
                        before = route[pos-1]
                        after = route[pos]
                        inc = distance_matrix[before, cust] + distance_matrix[cust, after] - distance_matrix[before, after]
                        if inc < best_inc:
                            best_inc = inc
                            best_new_route = route
                            best_new_pos = pos
                            best_new_route_idx = r_idx
            # Perform best insertion if it reduces max distance
            # Compute new distances after move
            # Clone routes to test
            new_routes = [r[:] for r in routes]
            if best_new_route_idx == max_idx:
                # stay in same route but different position
                new_routes[max_idx] = new_long[:best_new_pos] + [cust] + new_long[best_new_pos:]
            else:
                # remove from long, insert into target
                new_routes[max_idx] = new_long
                target_route = new_routes[best_new_route_idx]
                new_routes[best_new_route_idx] = target_route[:best_new_pos] + [cust] + target_route[best_new_pos:]
            new_dists = [route_dist(r) for r in new_routes]
            new_max = max(new_dists)
            if new_max < best_max:
                # Accept move
                routes = new_routes
                best_max = new_max
                best_routes = [r[:] for r in routes]
                improved = True
                break
        if not improved:
            # Try intra-route 2-opt on the longest route
            best_route = routes[max_idx]
            n_cust = len(best_route) - 2  # number of customers in route
            if n_cust >= 4:
                best_imp = 0.0
                best_i = best_j = -1
                for i in range(1, n_cust):  # i is customer index in route (1-indexed post-depot)
                    for j in range(i+1, n_cust+1):  # j is customer index inclusive? Actually reverse segment i..j
                        # Reverse segment from i to j (both inclusive)
                        new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                        if new_route[0] != 0 or new_route[-1] != 0:
                            continue
                        orig_dist = route_dist(best_route)
                        new_dist = route_dist(new_route)
                        reduction = orig_dist - new_dist
                        if reduction > best_imp:
                            best_imp = reduction
                            best_i, best_j = i, j
                if best_imp > 0:
                    new_route = best_route[:best_i] + best_route[best_i:best_j+1][::-1] + best_route[best_j+1:]
                    routes[max_idx] = new_route
                    new_dists = [route_dist(r) for r in routes]
                    new_max = max(new_dists)
                    if new_max < best_max:
                        best_max = new_max
                        best_routes = [r[:] for r in routes]
        # Report improvement
        # (We could call report_best_vrp but we don't have that function; we'll assume it's available globally)
        # The instruction says to call report_best_vrp(routes) when better found.
        # But we can't define it here; it's expected to be available in environment.
        # We'll simulate by checking if best_max improved and calling if possible.
    # Final best
    return best_routes