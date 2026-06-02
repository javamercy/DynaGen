import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    random.seed(0)
    
    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def construction():
        routes = [[0, 0] for _ in range(truck_count)]
        route_distances = [0.0 for _ in range(truck_count)]
        unassigned = set(range(1, n))
        
        def top_k_insertions(customer, k=3):
            # returns list of (cost, route_idx, position) sorted ascending
            candidates = []
            for r_idx, route in enumerate(routes):
                curr_dist = route_distances[r_idx]
                for i in range(1, len(route)):
                    new_dist = curr_dist - distance_matrix[route[i-1], route[i]] \
                               + distance_matrix[route[i-1], customer] \
                               + distance_matrix[customer, route[i]]
                    other_max = max([route_distances[j] for j in range(truck_count) if j != r_idx] or [0.0])
                    cand_max = max(new_dist, other_max)
                    candidates.append((cand_max, r_idx, i))
            candidates.sort(key=lambda x: x[0])
            return candidates[:k]
        
        while unassigned:
            regrets = []
            for c in unassigned:
                tops = top_k_insertions(c, k=3)
                if len(tops) < 3:
                    # fallback: use regret-2
                    if len(tops) == 2:
                        regret = tops[1][0] - tops[0][0]
                    else:
                        regret = 0.0
                else:
                    regret = tops[2][0] - tops[0][0]
                best_cost = tops[0][0] if tops else float('inf')
                regrets.append((regret, best_cost, c, tops[0]))
            regrets.sort(key=lambda x: (-x[0], -x[1], x[2]))
            selected = regrets[0][2]
            _, r_idx, i = regrets[0][3]
            route = routes[r_idx]
            route.insert(i, selected)
            route_distances[r_idx] = compute_route_distance(route)
            unassigned.remove(selected)
        return routes, route_distances
    
    def local_search(routes, route_distances):
        current_routes = [list(r) for r in routes]
        current_distances = list(route_distances)
        current_max = max(current_distances)
        improved = True
        max_iters = n * n
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            # Intra-route 2-opt (best improvement)
            for r_idx in range(truck_count):
                route = current_routes[r_idx]
                best_new_route = None
                best_new_dist = current_distances[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_distance(new_route)
                        if new_dist < best_new_dist - 1e-12:
                            best_new_dist = new_dist
                            best_new_route = new_route
                if best_new_route is not None:
                    current_routes[r_idx] = best_new_route
                    current_distances[r_idx] = best_new_dist
                    new_max = max(current_distances)
                    if new_max < current_max - 1e-12:
                        current_max = new_max
                        improved = True
            if improved:
                continue
            # Inter-swap (best improvement)
            best_swap = None
            best_swap_max = current_max
            for r1 in range(truck_count):
                for r2 in range(r1+1, truck_count):
                    route1 = current_routes[r1]
                    route2 = current_routes[r2]
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i] + [route2[j]] + route1[i+1:]
                            new2 = route2[:j] + [route1[i]] + route2[j+1:]
                            new_dist1 = compute_route_distance(new1)
                            new_dist2 = compute_route_distance(new2)
                            other_max = max([current_distances[k] for k in range(truck_count) if k not in (r1, r2)] or [0.0])
                            cand_max = max(new_dist1, new_dist2, other_max)
                            if cand_max < best_swap_max - 1e-12:
                                best_swap_max = cand_max
                                best_swap = (r1, r2, i, j, new1, new2, new_dist1, new_dist2)
            if best_swap is not None:
                r1, r2, i, j, new1, new2, new_dist1, new_dist2 = best_swap
                current_routes[r1] = new1
                current_routes[r2] = new2
                current_distances[r1] = new_dist1
                current_distances[r2] = new_dist2
                current_max = best_swap_max
                improved = True
                continue
            # Inter-relocate (best improvement)
            best_reloc = None
            best_reloc_max = current_max
            for r1 in range(truck_count):
                for r2 in range(truck_count):
                    if r1 == r2:
                        continue
                    route1 = current_routes[r1]
                    route2 = current_routes[r2]
                    for i in range(1, len(route1)-1):
                        c = route1[i]
                        new1 = route1[:i] + route1[i+1:]
                        new_dist1 = compute_route_distance(new1)
                        best_new2 = None
                        best_new_dist2 = float('inf')
                        for j in range(1, len(route2)):
                            new2 = route2[:j] + [c] + route2[j:]
                            d2 = compute_route_distance(new2)
                            if d2 < best_new_dist2:
                                best_new_dist2 = d2
                                best_new2 = new2
                        other_max = max([current_distances[k] for k in range(truck_count) if k not in (r1, r2)] or [0.0])
                        cand_max = max(new_dist1, best_new_dist2, other_max)
                        if cand_max < best_reloc_max - 1e-12:
                            best_reloc_max = cand_max
                            best_reloc = (r1, r2, new1, best_new2, new_dist1, best_new_dist2)
            if best_reloc is not None:
                r1, r2, new1, new2, new_dist1, new_dist2 = best_reloc
                current_routes[r1] = new1
                current_routes[r2] = new2
                current_distances[r1] = new_dist1
                current_distances[r2] = new_dist2
                current_max = best_reloc_max
                improved = True
        return current_routes, current_distances, current_max
    
    def ruin_recreate(routes, route_distances, ruin_pct):
        new_routes = [list(r) for r in routes]
        new_distances = list(route_distances)
        max_route_idx = max(range(truck_count), key=lambda i: new_distances[i])
        route = new_routes[max_route_idx]
        if len(route) <= 2:
            return new_routes, new_distances
        num_remove = max(1, int(ruin_pct * (len(route) - 2)))
        removable = list(range(1, len(route)-1))
        random.shuffle(removable)
        to_remove = removable[:num_remove]
        to_remove.sort(reverse=True)
        removed_customers = []
        for idx in to_remove:
            removed_customers.append(route.pop(idx))
        new_distances[max_route_idx] = compute_route_distance(route)
        unassigned = set(removed_customers)
        while unassigned:
            best_customer = None
            best_max_val = float('inf')
            best_pos = None
            for c in unassigned:
                for r_idx, r in enumerate(new_routes):
                    for i in range(1, len(r)):
                        new_dist = new_distances[r_idx] - distance_matrix[r[i-1], r[i]] \
                                   + distance_matrix[r[i-1], c] \
                                   + distance_matrix[c, r[i]]
                        other_max = max([new_distances[j] for j in range(truck_count) if j != r_idx] or [0.0])
                        cand_max = max(new_dist, other_max)
                        if cand_max < best_max_val - 1e-12:
                            best_max_val = cand_max
                            best_customer = c
                            best_pos = (r_idx, i)
            r_idx, i = best_pos
            new_routes[r_idx].insert(i, best_customer)
            new_distances[r_idx] = compute_route_distance(new_routes[r_idx])
            unassigned.remove(best_customer)
        return new_routes, new_distances
    
    best_routes = None
    best_max_val = float('inf')
    max_restarts = min(10, n) if n > 0 else 1
    
    route_distances = [0.0]*truck_count
    routes, route_distances = construction()
    routes, route_distances, current_max = local_search(routes, route_distances)
    if current_max < best_max_val - 1e-12:
        best_max_val = current_max
        best_routes = routes
        report_best_vrp(best_routes)
    
    for restart in range(1, max_restarts):
        ruin_pct = max(0.1, 0.5 - 0.05 * restart)
        routes = [list(r) for r in best_routes]
        route_distances = [compute_route_distance(r) for r in routes]
        routes, route_distances = ruin_recreate(routes, route_distances, ruin_pct)
        routes, route_distances, current_max = local_search(routes, route_distances)
        if current_max < best_max_val - 1e-12:
            best_max_val = current_max
            best_routes = routes
            report_best_vrp(best_routes)
    
    return best_routes