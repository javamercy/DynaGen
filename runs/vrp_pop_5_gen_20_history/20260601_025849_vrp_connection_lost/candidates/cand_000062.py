import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))
    
    def route_dist(route):
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += dist[a][b]
        return d
    
    def max_dist(routes):
        return max(route_dist(r) for r in routes)
    
    # Regret-2 construction
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
            new_route_dist = route_dist(new_route)
            other_max = 0.0
            if truck_count > 1:
                other_max = max(route_dist(r) for i, r in enumerate(routes) if i != r_idx)
            new_max = max(new_route_dist, other_max)
            candidates.append((-regret, new_max, c, r_idx, pos))
        
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, chosen_c, chosen_r, chosen_p = candidates[0]
        routes[chosen_r].insert(chosen_p, chosen_c)
        unassigned.remove(chosen_c)
    
    report_best_vrp(routes)
    best_routes = [list(r) for r in routes]
    best_max = max_dist(best_routes)
    
    # Improvement: focus on longest route, relocate or swap to reduce max distance
    max_iter = n * truck_count * 2
    stagnation_limit = max(10, n // 10)
    no_improve = 0
    
    for _ in range(max_iter):
        improved = False
        # Identify longest route (first if multiple)
        dists = [route_dist(r) for r in best_routes]
        longest_idx = max(range(len(dists)), key=lambda i: dists[i])
        longest_route = best_routes[longest_idx]
        # Try all possible moves involving longest route
        best_move = None
        best_new_max = best_max
        
        # Relocate a customer from longest to another route
        for pos in range(1, len(longest_route)-1):
            cust = longest_route[pos]
            new_long = longest_route[:pos] + longest_route[pos+1:]
            if len(new_long) == 2:
                new_long = [0, 0]
            for other_idx, other_route in enumerate(best_routes):
                if other_idx == longest_idx:
                    continue
                for ins in range(1, len(other_route)):
                    new_other = other_route[:ins] + [cust] + other_route[ins:]
                    # Compute new max distance
                    new_dists = []
                    for idx, r in enumerate(best_routes):
                        if idx == longest_idx:
                            new_dists.append(route_dist(new_long))
                        elif idx == other_idx:
                            new_dists.append(route_dist(new_other))
                        else:
                            new_dists.append(dists[idx])
                    candidate_max = max(new_dists)
                    if candidate_max < best_new_max:
                        best_new_max = candidate_max
                        best_move = ('relocate', longest_idx, other_idx, pos, ins, cust)
        
        # Swap a customer from longest with a customer from another route
        for pos1 in range(1, len(longest_route)-1):
            cust1 = longest_route[pos1]
            for other_idx, other_route in enumerate(best_routes):
                if other_idx == longest_idx:
                    continue
                for pos2 in range(1, len(other_route)-1):
                    cust2 = other_route[pos2]
                    new_long = longest_route[:pos1] + [cust2] + longest_route[pos1+1:]
                    new_other = other_route[:pos2] + [cust1] + other_route[pos2+1:]
                    new_dists = []
                    for idx, r in enumerate(best_routes):
                        if idx == longest_idx:
                            new_dists.append(route_dist(new_long))
                        elif idx == other_idx:
                            new_dists.append(route_dist(new_other))
                        else:
                            new_dists.append(dists[idx])
                    candidate_max = max(new_dists)
                    if candidate_max < best_new_max:
                        best_new_max = candidate_max
                        best_move = ('swap', longest_idx, other_idx, pos1, pos2, cust1, cust2)
        
        if best_move is not None:
            if best_move[0] == 'relocate':
                _, l_idx, o_idx, pos, ins, cust = best_move
                new_routes = [list(r) for r in best_routes]
                new_routes[l_idx] = new_routes[l_idx][:pos] + new_routes[l_idx][pos+1:]
                if len(new_routes[l_idx]) == 2:
                    new_routes[l_idx] = [0, 0]
                new_routes[o_idx] = new_routes[o_idx][:ins] + [cust] + new_routes[o_idx][ins:]
            else:  # swap
                _, l_idx, o_idx, pos1, pos2, cust1, cust2 = best_move
                new_routes = [list(r) for r in best_routes]
                new_routes[l_idx][pos1] = cust2
                new_routes[o_idx][pos2] = cust1
            # Update best if improved
            new_max = max_dist(new_routes)
            if new_max < best_max:
                best_max = new_max
                best_routes = new_routes
                report_best_vrp(best_routes)
                improved = True
                no_improve = 0
        
        if not improved:
            no_improve += 1
            if no_improve >= stagnation_limit:
                break
    
    # Final sanitization: ensure each route starts/ends with depot and empty routes are [0,0]
    final_routes = []
    for route in best_routes:
        if len(route) == 2 and route[0] == 0 and route[1] == 0:
            final_routes.append([0, 0])
        else:
            new_route = [0]
            for node in route:
                if node != 0:
                    new_route.append(node)
            new_route.append(0)
            final_routes.append(new_route)
    return final_routes