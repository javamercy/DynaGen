import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(12345)
    n = distance_matrix.shape[0]
    # Initialization: each route starts as [0,0]
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0] * truck_count
    
    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    for r in range(truck_count):
        route_distances[r] = compute_route_distance(routes[r])
    
    unassigned = set(range(1, n))
    
    def best_max(customer):
        best_val = float('inf')
        best_pos = None
        second_val = float('inf')
        for r_idx, route in enumerate(routes):
            curr_dist = route_distances[r_idx]
            for i in range(1, len(route)):
                new_dist = curr_dist - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], customer] + distance_matrix[customer, route[i]]
                other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                cand_max = max(new_dist, other_max)
                if cand_max < best_val:
                    second_val = best_val
                    best_val = cand_max
                    best_pos = (r_idx, i)
                elif cand_max < second_val and cand_max != best_val:
                    second_val = cand_max
        return best_val, second_val, best_pos
    
    while unassigned:
        regrets = []
        for c in unassigned:
            best_val, second_val, _ = best_max(c)
            regret = second_val - best_val if second_val != float('inf') else 0
            regrets.append((regret, c, best_val))
        regrets.sort(key=lambda x: (-x[0], x[1]))
        selected = regrets[0][1]
        best_val, _, best_pos = best_max(selected)
        r_idx, i = best_pos
        route = routes[r_idx]
        route.insert(i, selected)
        route_distances[r_idx] = compute_route_distance(route)
        unassigned.remove(selected)
    
    current_routes = [list(r) for r in routes]
    current_max = max(route_distances)
    best_routes = [list(r) for r in routes]
    best_max = current_max
    report_best_vrp(current_routes)
    
    # Adaptive SA parameters
    T0 = 0.1 * current_max
    if T0 == 0:
        T0 = 1.0
    T = T0
    cooling_rate = 0.99
    max_iter = 5000
    # Neighborhood success tracking
    attempts = [1, 1, 1]  # relocate, swap, 2-opt
    successes = [1, 1, 1]
    
    for iteration in range(max_iter):
        if T < 1e-6:
            break
        # Compute selection probabilities
        weights = [s / a for s, a in zip(successes, attempts)]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        nh = random.choices([0, 1, 2], weights=probs, k=1)[0]
        
        if nh == 0:  # relocate
            customers = list(range(1, n))
            cust = random.choice(customers)
            src_idx = None
            src_pos = None
            for idx, route in enumerate(current_routes):
                if cust in route:
                    src_idx = idx
                    src_pos = route.index(cust)
                    break
            if src_idx is None:
                continue
            src_route = current_routes[src_idx]
            if len(src_route) <= 2:
                continue
            new_src = src_route[:src_pos] + src_route[src_pos+1:]
            if len(new_src) == 2:
                new_src = [0,0]
            new_src_dist = compute_route_distance(new_src)
            tgt_idx = random.randint(0, truck_count-1)
            if tgt_idx == src_idx:
                continue
            tgt_route = current_routes[tgt_idx]
            if len(tgt_route) <= 2:
                pos = 1
                new_tgt = [0, cust, 0]
            else:
                pos = random.randint(1, len(tgt_route)-1)
                new_tgt = tgt_route[:pos] + [cust] + tgt_route[pos:]
            new_tgt_dist = compute_route_distance(new_tgt)
            other_max = max([route_distances[i] for i in range(truck_count) if i not in (src_idx, tgt_idx)], default=0.0)
            new_max = max(new_src_dist, new_tgt_dist, other_max)
            delta = new_max - current_max
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_routes[src_idx] = new_src
                current_routes[tgt_idx] = new_tgt
                route_distances[src_idx] = new_src_dist
                route_distances[tgt_idx] = new_tgt_dist
                current_max = new_max
                successes[nh] += 1
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [list(r) for r in current_routes]
                    report_best_vrp(best_routes)
            attempts[nh] += 1
        elif nh == 1:  # swap
            r1 = random.randint(0, truck_count-1)
            r2 = random.randint(0, truck_count-1)
            if r1 == r2:
                continue
            route1 = current_routes[r1]
            route2 = current_routes[r2]
            if len(route1) <= 2 or len(route2) <= 2:
                continue
            i = random.randint(1, len(route1)-2)
            j = random.randint(1, len(route2)-2)
            cust1 = route1[i]
            cust2 = route2[j]
            new_route1 = route1[:i] + [cust2] + route1[i+1:]
            new_route2 = route2[:j] + [cust1] + route2[j+1:]
            new_dist1 = compute_route_distance(new_route1)
            new_dist2 = compute_route_distance(new_route2)
            other_max = max([route_distances[k] for k in range(truck_count) if k not in (r1, r2)], default=0.0)
            new_max = max(new_dist1, new_dist2, other_max)
            delta = new_max - current_max
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_routes[r1] = new_route1
                current_routes[r2] = new_route2
                route_distances[r1] = new_dist1
                route_distances[r2] = new_dist2
                current_max = new_max
                successes[nh] += 1
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [list(r) for r in current_routes]
                    report_best_vrp(best_routes)
            attempts[nh] += 1
        else:  # 2-opt intra-route
            r = random.randint(0, truck_count-1)
            route = current_routes[r]
            if len(route) <= 3:
                continue
            i = random.randint(1, len(route)-3)
            j = random.randint(i+1, len(route)-2)
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            new_dist = compute_route_distance(new_route)
            other_max = max([route_distances[k] for k in range(truck_count) if k != r], default=0.0)
            new_max = max(new_dist, other_max)
            delta = new_max - current_max
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_routes[r] = new_route
                route_distances[r] = new_dist
                current_max = new_max
                successes[nh] += 1
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [list(r) for r in current_routes]
                    report_best_vrp(best_routes)
            attempts[nh] += 1
        T *= cooling_rate
    return best_routes