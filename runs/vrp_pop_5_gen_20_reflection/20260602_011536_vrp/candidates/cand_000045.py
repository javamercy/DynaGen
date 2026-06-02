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

    # Initial solution via minimax construction
    routes = [[0,0] for _ in range(truck_count)]
    unassigned = list(range(1,n))
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
                    new_d = 0
                    prev = route[0]
                    for k in range(1,len(route)):
                        if k==pos:
                            new_d += dist[prev][node]
                            prev = node
                        new_d += dist[prev][route[k]]
                        prev = route[k]
                    current_max = max(new_d, max(route_distance(routes[rr]) for rr in range(truck_count) if rr!=r))
                    if current_max < best_max or (current_max==best_max and new_d<best_total):
                        best_max = current_max
                        best_total = new_d
                        best_node = node
                        best_route = r
                        best_pos = pos
        routes[best_route].insert(best_pos, best_node)
        unassigned.remove(best_node)

    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)

    # Streamlined ruin-and-recreate with random removal and deterministic insertion
    max_iter = min(30, 2*n)
    T0 = 2.0
    Tf = 0.01
    for iteration in range(max_iter):
        current_routes = [list(r) for r in routes]
        # Remove a random fraction
        remove_frac = random.uniform(0.15, 0.35)
        remove_count = max(1, int(remove_frac*(n-1)))
        all_cust = list(range(1,n))
        random.shuffle(all_cust)
        to_remove = set(all_cust[:remove_count])
        removed = []
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            new_route = [route[0]]
            for node in route[1:-1]:
                if node in to_remove:
                    removed.append(node)
                else:
                    new_route.append(node)
            new_route.append(0)
            current_routes[r_idx] = new_route
            if len(current_routes[r_idx]) < 2:
                current_routes[r_idx] = [0,0]
        random.shuffle(removed)
        # Reconstruct with minimax insertion, deterministic tie-breaking (first found)
        while removed:
            best_max = float('inf')
            best_total = float('inf')
            best_tuple = None
            for node in removed:
                for r in range(truck_count):
                    route = current_routes[r]
                    for pos in range(1,len(route)):
                        new_d = 0
                        prev = route[0]
                        for k in range(1,len(route)):
                            if k==pos:
                                new_d += dist[prev][node]
                                prev = node
                            new_d += dist[prev][route[k]]
                            prev = route[k]
                        current_max = max(new_d, max(route_distance(current_routes[rr]) for rr in range(truck_count) if rr!=r))
                        if current_max < best_max:
                            best_max = current_max
                            best_total = new_d
                            best_tuple = (node,r,pos)
                        elif current_max == best_max:
                            if new_d < best_total:
                                best_total = new_d
                                best_tuple = (node,r,pos)
            if best_tuple is None:
                break
            node, r, pos = best_tuple
            current_routes[r].insert(pos, node)
            removed.remove(node)

        new_obj = objective(current_routes)
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [list(r) for r in current_routes]
        # SA acceptance
        T = T0 * (Tf/T0)**(iteration/(max_iter-1)) if max_iter>1 else T0
        current_obj = objective(routes)
        delta = new_obj - current_obj
        if delta < 0 or random.random() < np.exp(-delta/T):
            routes = current_routes

    return best_routes