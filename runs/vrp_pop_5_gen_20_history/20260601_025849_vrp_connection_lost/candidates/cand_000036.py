import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix
    
    def route_distance(route):
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += dist[a][b]
        return d
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    # Regret-2 construction (same as parent)
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))
    while unassigned:
        best_info = {}
        for c in unassigned:
            best = float('inf')
            second = float('inf')
            best_r = -1
            best_p = -1
            for r_idx, route in enumerate(routes):
                for i in range(len(route) - 1):
                    cost = dist[route[i]][c] + dist[c][route[i+1]] - dist[route[i]][route[i+1]]
                    if cost < best:
                        second = best
                        best = cost
                        best_r = r_idx
                        best_p = i + 1
                    elif cost < second:
                        second = cost
            best_info[c] = (best, second, best_r, best_p)
        
        candidates = []
        for c, (best, second, r_idx, pos) in best_info.items():
            regret = second - best if second != float('inf') else float('inf')
            new_route = routes[r_idx][:pos] + [c] + routes[r_idx][pos:]
            new_route_dist = route_distance(new_route)
            other_max = 0.0
            if truck_count > 1:
                other_max = max(route_distance(r) for i, r in enumerate(routes) if i != r_idx)
            new_max = max(new_route_dist, other_max)
            candidates.append((-regret, new_max, c, r_idx, pos))
        
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, chosen_c, chosen_r, chosen_p = candidates[0]
        routes[chosen_r].insert(chosen_p, chosen_c)
        unassigned.remove(chosen_c)
    
    best_routes = [list(r) for r in routes]
    best_max = max_distance(best_routes)
    report_best_vrp(best_routes)
    
    # LNS parameters
    max_iter = n * truck_count * 2
    stagnation_limit = max(10, (n - 1) // 10)
    no_improve = 0
    
    for _ in range(max_iter):
        # Destroy: remove a random subset of customers
        all_customers = []
        for route in routes:
            all_customers.extend(route[1:-1])
        if not all_customers:
            break
        removal_count = random.randint(max(1, int(0.1 * len(all_customers))), 
                                       min(len(all_customers), int(0.4 * len(all_customers))))
        removed_set = set(random.sample(all_customers, removal_count))
        
        # Remove customers from routes
        new_routes = []
        for route in routes:
            new_route = [0]
            for cust in route[1:-1]:
                if cust not in removed_set:
                    new_route.append(cust)
            new_route.append(0)
            if len(new_route) == 2:
                new_route = [0, 0]
            new_routes.append(new_route)
        
        # Repair: reinsert removed customers using regret-2
        unassigned = removed_set.copy()
        while unassigned:
            best_info = {}
            for c in unassigned:
                best = float('inf')
                second = float('inf')
                best_r = -1
                best_p = -1
                for r_idx, route in enumerate(new_routes):
                    for i in range(len(route) - 1):
                        cost = dist[route[i]][c] + dist[c][route[i+1]] - dist[route[i]][route[i+1]]
                        if cost < best:
                            second = best
                            best = cost
                            best_r = r_idx
                            best_p = i + 1
                        elif cost < second:
                            second = cost
                best_info[c] = (best, second, best_r, best_p)
            
            candidates = []
            for c, (best, second, r_idx, pos) in best_info.items():
                regret = second - best if second != float('inf') else float('inf')
                new_route = new_routes[r_idx][:pos] + [c] + new_routes[r_idx][pos:]
                new_route_dist = route_distance(new_route)
                other_max = 0.0
                if truck_count > 1:
                    other_max = max(route_distance(r) for i, r in enumerate(new_routes) if i != r_idx)
                new_max = max(new_route_dist, other_max)
                candidates.append((-regret, new_max, c, r_idx, pos))
            
            candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            _, _, chosen_c, chosen_r, chosen_p = candidates[0]
            new_routes[chosen_r].insert(chosen_p, chosen_c)
            unassigned.remove(chosen_c)
        
        # Evaluate
        new_max = max_distance(new_routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in new_routes]
            routes = new_routes
            report_best_vrp(best_routes)
            no_improve = 0
        else:
            routes = [list(r) for r in best_routes]
            no_improve += 1
            if no_improve >= stagnation_limit:
                break
    
    # Ensure exactly truck_count routes
    final_routes = []
    for route in best_routes:
        if len(route) == 2:
            final_routes.append([0, 0])
        else:
            if route[0] != 0:
                route.insert(0, 0)
            if route[-1] != 0:
                route.append(0)
            final_routes.append(route)
    return final_routes