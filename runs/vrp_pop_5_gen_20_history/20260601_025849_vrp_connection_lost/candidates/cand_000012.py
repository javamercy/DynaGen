import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    
    def route_distance(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_route_distance(routes):
        return max(route_distance(r) for r in routes) if routes else 0.0
    
    def insert_customer(route, pos, cust):
        return route[:pos] + [cust] + route[pos:]
    
    # Initial construction: greedy min-max insertion
    for cust in customers:
        best_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = insert_customer(route, pos, cust)
                new_route_dist = route_distance(new_route)
                other_routes = [routes[i] for i in range(truck_count) if i != r_idx]
                other_max = max((route_distance(r) for r in other_routes), default=0.0)
                new_max = max(new_route_dist, other_max)
                if new_max < best_max or (new_max == best_max and (r_idx < best_route_idx or (r_idx == best_route_idx and pos < best_pos))):
                    best_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
        routes[best_route_idx] = insert_customer(routes[best_route_idx], best_pos, cust)
    
    best_routes = [list(r) for r in routes]
    best_max = max_route_distance(best_routes)
    report_best_vrp(best_routes)
    
    # Parameters for improvement
    max_iter = n * 2
    destroy_fraction = 0.2
    min_destroy = 1
    
    for iteration in range(max_iter):
        # Compute removal gain for each customer
        removal_gain = []
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route) - 1):
                cust = route[pos]
                prev = route[pos-1]
                next = route[pos+1]
                gain = distance_matrix[prev][cust] + distance_matrix[cust][next] - distance_matrix[prev][next]
                removal_gain.append((gain, r_idx, pos, cust))
        removal_gain.sort(key=lambda x: (-x[0], x[3]))  # descending gain, tie by customer ID
        destroy_count = max(min_destroy, int(n * destroy_fraction))
        destroy_count = min(destroy_count, len(removal_gain))
        # Identify customers to remove
        removed_info = []
        for gain, r_idx, pos, cust in removal_gain[:destroy_count]:
            removed_info.append((r_idx, pos, cust))
        # Remove them from routes (process in descending position order per route)
        # Group by route index and sort positions descending
        removed_by_route = {}
        for r_idx, pos, cust in removed_info:
            removed_by_route.setdefault(r_idx, []).append((pos, cust))
        for r_idx, items in removed_by_route.items():
            items.sort(key=lambda x: -x[0])
            route = routes[r_idx]
            for pos, cust in items:
                route = route[:pos] + route[pos+1:]
            routes[r_idx] = route
        # Repair with regret-2 insertion
        unassigned = [c for _, _, c in removed_info]
        while unassigned:
            # Compute best and second-best insertion for each unassigned customer
            info = {}
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
                info[c] = (best, best_route, best_pos, second_best)
            # Compute regret
            regret = {}
            for c, (best, _, _, second) in info.items():
                regret[c] = second - best if second != float('inf') else float('inf')
            max_regret = max(regret.values())
            candidates = [c for c in unassigned if regret[c] == max_regret]
            # Tie-breaking: smaller best cost, then customer ID
            candidates.sort(key=lambda c: (info[c][0], c))
            chosen = candidates[0]
            best_cost, best_route, best_pos, _ = info[chosen]
            routes[best_route] = insert_customer(routes[best_route], best_pos, chosen)
            unassigned.remove(chosen)
        # Evaluate new solution
        new_max = max_route_distance(routes)
        if new_max < best_max:
            best_max = new_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        else:
            # Revert to best known solution
            routes = [list(r) for r in best_routes]
            continue
        # Improvement: targeted reduction of longest route and 2-opt
        # 2-opt on each route (first improving move per route)
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        routes[r_idx] = new_route
                        improved = True
                        break
                if improved:
                    break
        # Relocate one customer from the longest route to another
        current_max = max_route_distance(routes)
        longest_indices = [i for i, r in enumerate(routes) if route_distance(r) == current_max]
        if longest_indices:
            r_idx = longest_indices[0]
            route = routes[r_idx]
            if len(route) > 3:
                for pos in range(1, len(route)-1):
                    cust = route[pos]
                    for other_idx, other_route in enumerate(routes):
                        if other_idx == r_idx:
                            continue
                        for other_pos in range(1, len(other_route)):
                            new_other = insert_customer(other_route, other_pos, cust)
                            new_self = route[:pos] + route[pos+1:]
                            new_routes = [list(r) for r in routes]
                            new_routes[r_idx] = new_self
                            new_routes[other_idx] = new_other
                            new_max = max_route_distance(new_routes)
                            if new_max < best_max:
                                best_max = new_max
                                best_routes = [list(r) for r in new_routes]
                                routes = new_routes
                                report_best_vrp(best_routes)
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
    return best_routes