import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    def max_route_len(routes):
        return max(route_length(r) for r in routes)

    best_routes = None
    best_max = float('inf')
    max_attempts = max(1, n // 5)

    for attempt in range(max_attempts):
        # Construction: probabilistic regret with more randomness
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        random.shuffle(unassigned)
        while unassigned:
            candidates = []
            for cust in unassigned:
                costs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        costs.append((cost, r_idx, pos))
                costs.sort(key=lambda x: x[0])
                best_cost = costs[0][0]
                second_cost = costs[1][0] if len(costs) > 1 else best_cost + 1e9
                regret = second_cost - best_cost
                candidates.append((cust, regret, best_cost, costs[0][1], costs[0][2]))
            candidates.sort(key=lambda x: -x[1])
            # Higher exploration: 30% random pick
            if random.random() < 0.3 and len(candidates) > 1:
                chosen = random.choice(candidates)
            else:
                k = min(3, len(candidates))
                chosen = random.choice(candidates[:k])
            cust, _, _, r_idx, pos = chosen
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)

        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

        # Local search with simulated annealing acceptance
        improved = True
        iter_count = 0
        max_iter = n * truck_count * 2
        no_improve_count = 0
        perturbation_threshold = max(1, n // 4)
        # Simulated annealing parameters
        temp = 0.1 * current_max  # initial temperature
        cooling = 0.99

        while iter_count < max_iter:
            iter_count += 1
            improved = False
            lengths = [route_length(r) for r in routes]
            max_idx = np.argmax(lengths)
            max_route = routes[max_idx]
            
            # Inter-route relocate from longest route
            if len(max_route) > 2:
                best_delta = 0
                best_move = None
                for cust in max_route[1:-1]:
                    new_max_route = [x for x in max_route if x != cust]
                    new_max_len = route_length(new_max_route)
                    for r_idx in range(truck_count):
                        if r_idx == max_idx:
                            continue
                        other_route = routes[r_idx]
                        for pos in range(1, len(other_route)):
                            new_other = other_route[:pos] + [cust] + other_route[pos:]
                            new_other_len = route_length(new_other)
                            new_max_candidate = max(new_max_len, new_other_len, *[lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)])
                            if new_max_candidate < current_max:
                                delta = current_max - new_max_candidate
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = (cust, max_idx, r_idx, pos)
                if best_move:
                    cust, from_idx, to_idx, pos = best_move
                    routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                    routes[to_idx].insert(pos, cust)
                    current_max = current_max - best_delta
                    improved = True
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
                else:
                    # Random move with SA acceptance
                    if random.random() < 0.2:
                        cust = random.choice(max_route[1:-1])
                        new_max_route = [x for x in max_route if x != cust]
                        new_max_len = route_length(new_max_route)
                        r_idx = random.randrange(truck_count)
                        if r_idx == max_idx:
                            r_idx = (r_idx + 1) % truck_count
                        other_route = routes[r_idx]
                        pos = random.randrange(1, len(other_route)+1)
                        new_other = other_route[:pos] + [cust] + other_route[pos:]
                        new_other_len = route_length(new_other)
                        new_max_candidate = max(new_max_len, new_other_len, *[lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)])
                        delta = current_max - new_max_candidate
                        if delta > 0 or (temp > 0 and random.random() < np.exp(delta/temp)):
                            routes[from_idx] = [x for x in routes[from_idx] if x != cust]  # note: from_idx is max_idx
                            routes[r_idx].insert(pos, cust)
                            current_max = new_max_candidate
                            improved = True
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
            
            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for k in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        new_len = route_length(new_route)
                        old_len = route_length(route)
                        if new_len < old_len:
                            route[:] = new_route
                            improved = True
                            new_max = max_route_len(routes)
                            if new_max < current_max:
                                current_max = new_max
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(routes)
                            break
                    if improved:
                        break
            
            # Cooling
            temp *= cooling
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= perturbation_threshold:
                    # Aggressive perturbation: relocate 40% customers randomly
                    customers = list(range(1, n))
                    random.shuffle(customers)
                    num_perturb = max(1, n // 2)
                    for cust in customers[:num_perturb]:
                        for r_idx, route in enumerate(routes):
                            if cust in route:
                                routes[r_idx] = [x for x in route if x != cust]
                                break
                        r_idx = random.randrange(truck_count)
                        pos = random.randrange(1, len(routes[r_idx]))
                        routes[r_idx].insert(pos, cust)
                    current_max = max_route_len(routes)
                    no_improve_count = 0
                    improved = True
                    temp = 0.1 * current_max  # reset temperature
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
            else:
                no_improve_count = 0
        # End local search
    
    if best_routes is None:
        best_routes = routes
    return best_routes