import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    
    # Helper functions
    def route_distance(route):
        if len(route) == 2:
            return 0.0  # depot to depot distance is 0 (assuming dist[0][0]=0)
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_route_distance(routes):
        return max(route_distance(r) for r in routes) if routes else 0.0
    
    def insert_customer(route, pos, cust):
        return route[:pos] + [cust] + route[pos:]
    
    # Initial construction: greedy min-max insertion (like cand_000002)
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
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
    report_best_vrp(best_routes)
    
    # LNS parameters
    max_iter = n * 2
    destroy_fraction = 0.2
    min_destroy = 1
    
    for iteration in range(max_iter):
        # Destroy: remove most expensive customers (largest removal gain)
        # Compute removal gain for each customer
        removal_gain = []
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)-1):
                cust = route[pos]
                prev = route[pos-1]
                next = route[pos+1]
                gain = distance_matrix[prev][cust] + distance_matrix[cust][next] - distance_matrix[prev][next]
                removal_gain.append((gain, r_idx, pos, cust))
        # Sort descending by gain
        removal_gain.sort(key=lambda x: (-x[0], x[3]))  # tie by customer ID
        destroy_count = max(min_destroy, int(n * destroy_fraction))
        destroy_count = min(destroy_count, len(removal_gain))
        # Identify customers to remove
        removed_customers = []
        for gain, r_idx, pos, cust in removal_gain[:destroy_count]:
            removed_customers.append((r_idx, pos, cust))
        # Remove them (must remove from the end to preserve indices)
        # Sort removal by route index and position descending
        removed_customers.sort(key=lambda x: (x[0], -x[1]))
        for r_idx, pos, cust in removed_customers:
            route = routes[r_idx]
            # Since we remove from the end within same route, indices shift, but we process descending pos
            routes[r_idx] = route[:pos] + route[pos+1:]
        # Now routes may have empty trucks (just [0,0]) - that's fine
        
        # Repair: regret-2 insertion for removed customers
        unassigned = [c for _, _, c in removed_customers]
        # Regret-2 insertion
        while unassigned:
            best_costs = {}
            for c in unassigned:
                best = float('inf')
                second_best = float('inf')
                best_route = -1
                best_pos = -1
                for r_idx, route in enumerate(routes):
                    for i in range(len(route) - 1):
                        cost = distance_matrix[route[i]][c] + distance_matrix[c][route[i+1]] - distance_matrix[route[i]][route[i+1]]
                        if cost < best:
                            second_best = best
                            best = cost
                            best_route = r_idx
                            best_pos = i + 1
                        elif cost < second_best:
                            second_best = cost
                best_costs[c] = (best, best_route, best_pos, second_best)
            # compute regret
            regret = {}
            for c, (best, _, _, second) in best_costs.items():
                if second == float('inf'):
                    regret[c] = float('inf')
                else:
                    regret[c] = second - best
            max_regret = max(regret.values())
            candidates = [c for c in unassigned if regret[c] == max_regret]
            if len(candidates) > 1:
                candidates.sort(key=lambda c: (-best_costs[c][0], c))
            chosen = candidates[0]
            best_cost, best_route, best_pos, _ = best_costs[chosen]
            routes[best_route] = insert_customer(routes[best_route], best_pos, chosen)
            unassigned.remove(chosen)
        
        # Evaluate new solution
        new_max = max_route_distance(routes)
        if new_max < best_max:
            best_max = new_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        else:
            # optional: revert? We accept only improvements, so revert to best_routes
            routes = [list(r) for r in best_routes]
    
    return best_routes