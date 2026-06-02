import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(12345)
    n = distance_matrix.shape[0]
    
    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    # Greedy min-max construction
    def greedy_minmax_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        route_distances = [0.0] * truck_count
        unassigned = set(range(1, n))
        
        while unassigned:
            best_customer = None
            best_route_idx = None
            best_position = None
            best_inc_max = float('inf')
            for c in unassigned:
                for r_idx, route in enumerate(routes):
                    curr_dist = route_distances[r_idx]
                    for i in range(1, len(route)):
                        new_dist = curr_dist - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], c] + distance_matrix[c, route[i]]
                        other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                        cand_max = max(new_dist, other_max)
                        if cand_max < best_inc_max:
                            best_inc_max = cand_max
                            best_customer = c
                            best_route_idx = r_idx
                            best_position = i
            if best_customer is None:
                break
            route = routes[best_route_idx]
            route.insert(best_position, best_customer)
            route_distances[best_route_idx] = compute_route_distance(route)
            unassigned.remove(best_customer)
        return routes, route_distances, max(route_distances)
    
    initial_routes, initial_distances, initial_max = greedy_minmax_construction()
    best_routes = [list(r) for r in initial_routes]
    best_max = initial_max
    current_routes = [list(r) for r in initial_routes]
    current_distances = list(initial_distances)
    current_max = initial_max
    report_best_vrp(best_routes)
    
    def simulated_annealing(initial_routes, initial_distances, initial_max, max_iter, T0, cooling_rate, seed_offset):
        random.seed(12345 + seed_offset)
        nonlocal best_routes, best_max
        current_routes = [list(r) for r in initial_routes]
        current_distances = list(initial_distances)
        current_max = initial_max
        T = T0
        success_counts = [1, 1, 1]
        total_successes = 3
        nh_names = [0, 1, 2]
        
        for iteration in range(max_iter):
            if T < 1e-6:
                break
            probs = [c / total_successes for c in success_counts]
            nh = random.choices(nh_names, weights=probs)[0]
            accepted = False
            
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
                    new_src = [0, 0]
                new_src_dist = compute_route_distance(new_src)
                tgt_idx = random.randint(0, truck_count-1)
                if tgt_idx == src_idx:
                    continue
                tgt_route = current_routes[tgt_idx]
                if len(tgt_route) <= 2:
                    new_tgt = [0, cust, 0]
                else:
                    pos = random.randint(1, len(tgt_route)-1)
                    new_tgt = tgt_route[:pos] + [cust] + tgt_route[pos:]
                new_tgt_dist = compute_route_distance(new_tgt)
                other_max = max([current_distances[i] for i in range(truck_count) if i not in (src_idx, tgt_idx)], default=0.0)
                new_max = max(new_src_dist, new_tgt_dist, other_max)
                delta = new_max - current_max
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_routes[src_idx] = new_src
                    current_routes[tgt_idx] = new_tgt
                    current_distances[src_idx] = new_src_dist
                    current_distances[tgt_idx] = new_tgt_dist
                    current_max = new_max
                    accepted = True
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
                other_max = max([current_distances[k] for k in range(truck_count) if k not in (r1, r2)], default=0.0)
                new_max = max(new_dist1, new_dist2, other_max)
                delta = new_max - current_max
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_routes[r1] = new_route1
                    current_routes[r2] = new_route2
                    current_distances[r1] = new_dist1
                    current_distances[r2] = new_dist2
                    current_max = new_max
                    accepted = True
            else:  # 2-opt intra-route
                r = random.randint(0, truck_count-1)
                route = current_routes[r]
                if len(route) <= 3:
                    continue
                i = random.randint(1, len(route)-3)
                j = random.randint(i+1, len(route)-2)
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_dist = compute_route_distance(new_route)
                other_max = max([current_distances[k] for k in range(truck_count) if k != r], default=0.0)
                new_max = max(new_dist, other_max)
                delta = new_max - current_max
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_routes[r] = new_route
                    current_distances[r] = new_dist
                    current_max = new_max
                    accepted = True
            
            if accepted:
                success_counts[nh] += 1
                total_successes += 1
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [list(r) for r in current_routes]
                    report_best_vrp(best_routes)
            else:
                success_counts[nh] = max(1, success_counts[nh] - 0.1)
                total_successes = sum(success_counts)
            
            T *= cooling_rate
        return current_routes, current_distances, current_max
    
    # Phase 1
    T0 = 0.1 * initial_max if initial_max > 0 else 1.0
    cooling_rate = 0.99
    max_iter1 = 1000
    simulated_annealing(initial_routes, initial_distances, initial_max, max_iter1, T0, cooling_rate, 0)
    
    # Phase 2: restart from best solution with perturbation
    num_perturb = max(1, int(0.2 * (n-1)))
    perturb_routes = [list(r) for r in best_routes]
    perturb_distances = [compute_route_distance(r) for r in perturb_routes]
    for _ in range(num_perturb):
        customers = [c for route in perturb_routes for c in route if c != 0]
        if not customers:
            break
        cust = random.choice(customers)
        src_idx = None
        src_pos = None
        for idx, route in enumerate(perturb_routes):
            if cust in route:
                src_idx = idx
                src_pos = route.index(cust)
                break
        if src_idx is None:
            continue
        src_route = perturb_routes[src_idx]
        if len(src_route) <= 2:
            continue
        new_src = src_route[:src_pos] + src_route[src_pos+1:]
        if len(new_src) == 2:
            new_src = [0, 0]
        tgt_idx = random.randint(0, truck_count-1)
        if tgt_idx == src_idx:
            tgt_idx = (tgt_idx + 1) % truck_count
        tgt_route = perturb_routes[tgt_idx]
        if len(tgt_route) <= 2:
            new_tgt = [0, cust, 0]
        else:
            pos = random.randint(1, len(tgt_route)-1)
            new_tgt = tgt_route[:pos] + [cust] + tgt_route[pos:]
        perturb_routes[src_idx] = new_src
        perturb_routes[tgt_idx] = new_tgt
        perturb_distances[src_idx] = compute_route_distance(new_src)
        perturb_distances[tgt_idx] = compute_route_distance(new_tgt)
    perturb_max = max(perturb_distances)
    T0_2 = 0.05 * perturb_max if perturb_max > 0 else 1.0
    cooling_rate_2 = 0.98
    max_iter2 = 800
    simulated_annealing(perturb_routes, perturb_distances, perturb_max, max_iter2, T0_2, cooling_rate_2, 1)
    
    return best_routes