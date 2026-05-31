import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    random.seed(0)
    
    # Initialization
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = list(range(1, n))
    
    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def best_insertion(c, routes, route_dists):
        best = (float('inf'), -1, -1)
        second = (float('inf'), -1, -1)
        for r_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            other_max = 0.0
            for j, d in enumerate(route_dists):
                if j != r_idx and d > other_max:
                    other_max = d
            for pos in range(1, len(route)):
                pred = route[pos-1]
                succ = route[pos]
                new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                new_max = max(other_max, new_dist)
                if new_max < best[0]:
                    best, second = (new_max, r_idx, pos), best
                elif new_max < second[0]:
                    second = (new_max, r_idx, pos)
        return best[0], best[1], best[2], second[0]
    
    # Regret construction
    while unassigned:
        bests = []
        for c in unassigned:
            best_new_max, best_route, best_pos, second_new_max = best_insertion(c, routes, route_dists)
            if best_route == -1:
                continue
            regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
            bests.append((-regret, c, best_route, best_pos, best_new_max))
        bests.sort(key=lambda x: (x[0], x[1]))
        _, c, best_route, best_pos, new_max = bests[0]
        route = routes[best_route]
        route.insert(best_pos, c)
        route_dists[best_route] = route_dist(route)
        unassigned.remove(c)
    
    best_routes = [route[:] for route in routes]
    best_max = max(route_dists)
    current_routes = [route[:] for route in routes]
    current_dists = route_dists[:]
    report_best_vrp(best_routes)
    
    # SA parameters
    max_iter = n * 10
    T0 = best_max * 0.1
    restart_freq = max(1, n // 2)
    no_improve = 0
    
    for it in range(max_iter):
        T = T0 * (1.0 - it / max_iter)
        # Destroy
        num_remove = random.randint(max(1, (n-1)//10), max(1, (n-1)*4//10))
        customers = list(range(1, n))
        random.shuffle(customers)
        to_remove = customers[:num_remove]
        # Remove
        temp_routes = [route[:] for route in current_routes]
        temp_dists = current_dists[:]
        for c in to_remove:
            for r_idx in range(truck_count):
                if c in temp_routes[r_idx]:
                    pos = temp_routes[r_idx].index(c)
                    pred = temp_routes[r_idx][pos-1]
                    succ = temp_routes[r_idx][pos+1]
                    temp_dists[r_idx] += distance_matrix[pred, succ] - distance_matrix[pred, c] - distance_matrix[c, succ]
                    temp_routes[r_idx].pop(pos)
                    break
        # Repair
        unassigned = to_remove[:]
        while unassigned:
            bests = []
            for c in unassigned:
                best_new_max, best_route, best_pos, second_new_max = best_insertion(c, temp_routes, temp_dists)
                if best_route == -1:
                    continue
                regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
                bests.append((-regret, c, best_route, best_pos, best_new_max))
            bests.sort(key=lambda x: (x[0], x[1]))
            _, c, best_route, best_pos, _ = bests[0]
            route = temp_routes[best_route]
            route.insert(best_pos, c)
            temp_dists[best_route] = route_dist(route)
            unassigned.remove(c)
        new_max = max(temp_dists)
        # Acceptance
        delta = new_max - max(current_dists)
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta / T) if T > 0 else False
        if accept:
            current_routes = [route[:] for route in temp_routes]
            current_dists = temp_dists[:]
            if new_max < best_max - 1e-12:
                best_max = new_max
                best_routes = [route[:] for route in current_routes]
                report_best_vrp(best_routes)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        # Restart if stagnation
        if no_improve >= restart_freq:
            # Reset to best and shake
            current_routes = [route[:] for route in best_routes]
            current_dists = [route_dist(route) for route in current_routes]
            # Shake: random swaps between different routes
            num_shakes = max(1, n // 10)
            for _ in range(num_shakes):
                # Pick two distinct routes with at least 2 customers (excluding depots)
                candidates = [r for r in range(truck_count) if len(current_routes[r]) > 3]
                if len(candidates) < 2:
                    continue
                r1, r2 = random.sample(candidates, 2)
                # Pick random internal positions
                pos1 = random.randint(1, len(current_routes[r1])-2)
                pos2 = random.randint(1, len(current_routes[r2])-2)
                # Swap customers
                c1 = current_routes[r1][pos1]
                c2 = current_routes[r2][pos2]
                current_routes[r1][pos1] = c2
                current_routes[r2][pos2] = c1
                # Update distances
                current_dists[r1] = route_dist(current_routes[r1])
                current_dists[r2] = route_dist(current_routes[r2])
            no_improve = 0
    return best_routes