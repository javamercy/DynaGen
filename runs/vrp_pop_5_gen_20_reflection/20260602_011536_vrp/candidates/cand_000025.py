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
    
    # Initial solution via minimax construction (deterministic)
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
                    if (current_max < best_max) or (current_max == best_max and new_dist < best_total):
                        best_max = current_max
                        best_total = new_dist
                        best_node = node
                        best_route = r
                        best_pos = pos
        routes[best_route].insert(best_pos, best_node)
        unassigned.remove(best_node)
    
    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)
    
    # Ruin-and-recreate with stochastic removal
    max_iter = min(50, n * 3)
    for _ in range(max_iter):
        current_routes = [list(r) for r in routes]
        # Compute contribution of each customer
        contribution = {}
        for r_idx, route in enumerate(current_routes):
            if len(route) <= 2:
                continue
            for i in range(1, len(route)-1):
                node = route[i]
                contrib = dist[route[i-1]][node] + dist[node][route[i+1]]
                contribution[node] = contrib
        # Stochastic removal via roulette wheel based on contribution
        all_custs = list(contribution.keys())
        if not all_custs:
            continue
        weights = [contribution[c] for c in all_custs]
        min_remove = max(1, int(0.2 * (n-1)))
        max_remove = max(min_remove+1, int(0.4 * (n-1)))
        remove_count = random.randint(min_remove, min(max_remove, len(all_custs)))
        # sample without replacement using weights
        if sum(weights) == 0:
            to_remove = set(random.sample(all_custs, remove_count))
        else:
            pop = random.choices(all_custs, weights=weights, k=remove_count)
            # ensure no duplicates; if duplicates occur, adjust
            to_remove = set()
            while len(to_remove) < remove_count:
                if len(pop) == 0:
                    pop = random.choices(all_custs, weights=weights, k=remove_count - len(to_remove))
                to_remove.add(pop.pop())
        # Remove customers
        removed_list = []
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            new_route = [route[0]]
            for node in route[1:-1]:
                if node in to_remove:
                    removed_list.append(node)
                else:
                    new_route.append(node)
            new_route.append(0)
            current_routes[r_idx] = new_route
            if len(current_routes[r_idx]) < 2:
                current_routes[r_idx] = [0, 0]
        # Randomize insertion order
        random.shuffle(removed_list)
        unassigned = removed_list
        # Reconstruct via minimax insertion
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
                        if (current_max < best_max) or (current_max == best_max and new_dist < best_total):
                            best_max = current_max
                            best_total = new_dist
                            best_node = node
                            best_route = r
                            best_pos = pos
            current_routes[best_route].insert(best_pos, best_node)
            unassigned.remove(best_node)
        # Apply limited intra-route 2-opt
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            for _ in range(10):
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        old_dist = route_distance(route)
                        if new_dist < old_dist:
                            route = new_route
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            current_routes[r_idx] = route
        new_obj = objective(current_routes)
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [list(r) for r in current_routes]
            # Report best
            try:
                from vrp_utils import report_best_vrp
                report_best_vrp(best_routes)
            except ImportError:
                pass
        # Always accept
        routes = current_routes
    return best_routes