import numpy as np
import random

random.seed(0)

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix
    
    def route_dist(route):
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += dist[a][b]
        return d
    
    def max_dist(routes):
        return max(route_dist(r) for r in routes)
    
    # Minimax construction (from cand_000053)
    def initial_solution():
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            best_info = {}
            for c in unassigned:
                best = float('inf')
                best_r = -1
                best_p = -1
                for r_idx, route in enumerate(routes):
                    for i in range(len(route) - 1):
                        cost = dist[route[i]][c] + dist[c][route[i+1]] - dist[route[i]][route[i+1]]
                        if cost < best:
                            best = cost
                            best_r = r_idx
                            best_p = i + 1
                best_info[c] = (best, best_r, best_p)
            candidates = []
            for c, (best, r_idx, pos) in best_info.items():
                new_route = routes[r_idx][:pos] + [c] + routes[r_idx][pos:]
                new_route_dist = route_dist(new_route)
                other_max = 0.0
                if truck_count > 1:
                    other_max = max(route_dist(r) for i, r in enumerate(routes) if i != r_idx)
                new_max = max(new_route_dist, other_max)
                candidates.append((new_max, c, r_idx, pos))
            candidates.sort(key=lambda x: (x[0], x[1]))
            _, chosen_c, chosen_r, chosen_p = candidates[0]
            routes[chosen_r].insert(chosen_p, chosen_c)
            unassigned.remove(chosen_c)
        return routes
    
    routes = initial_solution()
    best_routes = [list(r) for r in routes]
    best_max = max_dist(best_routes)
    report_best_vrp(best_routes)
    
    # LNS parameters
    max_iter = max(100, n * (n - 1) // 2)  # bounded by instance size
    destroy_size = max(1, (n - 1) // 5)  # remove ~20% customers
    
    def regret3_minimax_insertion(unassigned, routes):
        # returns a new routes list after inserting all unassigned using regret-3
        routes = [list(r) for r in routes]
        while unassigned:
            best_infos = {}
            for c in unassigned:
                costs = []
                for r_idx, route in enumerate(routes):
                    for i in range(len(route) - 1):
                        cost = dist[route[i]][c] + dist[c][route[i+1]] - dist[route[i]][route[i+1]]
                        costs.append((cost, r_idx, i+1))
                costs.sort(key=lambda x: x[0])
                top3 = costs[:3]
                if len(top3) < 3:
                    regret = float('inf')
                else:
                    c1, c2, c3 = top3[0][0], top3[1][0], top3[2][0]
                    regret = (c2 + c3) - 2 * c1
                best_r = top3[0][1]
                best_p = top3[0][2]
                new_route = routes[best_r][:best_p] + [c] + routes[best_r][best_p:]
                new_route_dist = route_dist(new_route)
                other_max = 0.0
                if truck_count > 1:
                    other_max = max(route_dist(r) for i, r in enumerate(routes) if i != best_r)
                new_max = max(new_route_dist, other_max)
                best_infos[c] = (-regret, new_max, c, best_r, best_p)
            candidates = list(best_infos.values())
            candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            _, _, chosen_c, chosen_r, chosen_p = candidates[0]
            routes[chosen_r].insert(chosen_p, chosen_c)
            unassigned.remove(chosen_c)
        return routes
    
    for _ in range(max_iter):
        # Destroy: randomly remove customers (including depot? no)
        # Choose a random set of customers to remove (all customers may be removed)
        customers = list(range(1, n))
        random.shuffle(customers)
        to_remove = customers[:destroy_size]
        # If we remove all customers, solution becomes trivial, so avoid
        if len(to_remove) > n - 2:
            to_remove = customers[:max(1, (n-1)//2)]
        # Remove from routes
        for cust in to_remove:
            for route in routes:
                if cust in route:
                    route.remove(cust)
                    break
        # Repair: insert removed customers using regret-3 minimax
        unassigned = set(to_remove)
        new_routes = regret3_minimax_insertion(unassigned, routes)
        # Evaluate
        new_max = max_dist(new_routes)
        # Accept if better or with small probability (simulated annealing? no, keep it simple)
        if new_max < best_max:
            routes = new_routes
            best_routes = [list(r) for r in routes]
            best_max = new_max
            report_best_vrp(best_routes)
        else:
            # keep previous routes, but maybe still accept to diversify? 
            # For diversification, accept with probability exp(-(new_max - best_max)/temperature) but we skip to keep deterministic? 
            # Use a simple acceptance: if new_max < best_max * 1.1, accept (weak)
            # Actually, let's accept only improvements to maintain finite loops
            pass
    
    final_routes = []
    for route in best_routes:
        if len(route) == 2:
            final_routes.append([0, 0])
        else:
            final_routes.append([0] + route[1:-1] + [0])
    return final_routes