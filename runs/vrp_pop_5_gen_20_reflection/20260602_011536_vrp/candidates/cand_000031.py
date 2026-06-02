import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()
    random.seed(0)  # for reproducibility

    def route_distance(route):
        if len(route) < 2:
            return 0
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))

    def objective(routes):
        return max(route_distance(r) for r in routes)

    # ---- initial minimax construction ----
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    unassigned.sort(key=lambda x: dist[0][x], reverse=True)  # deterministic order
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
                    if current_max < best_max or (current_max == best_max and new_dist < best_total):
                        best_max = current_max
                        best_total = new_dist
                        best_node = node
                        best_route = r
                        best_pos = pos
        routes[best_route].insert(best_pos, best_node)
        unassigned.remove(best_node)

    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)
    report_best_vrp(best_routes)

    current_routes = [list(r) for r in routes]
    current_obj = objective(current_routes)

    # ---- SA parameters ----
    max_iter = min(50, n * 2)
    T0 = 0.1 * current_obj  # initial temperature
    if T0 == 0:
        T0 = 1.0
    removal_fraction = 0.3

    all_customers = list(range(1, n))
    for iteration in range(max_iter):
        # compute route distances for weighting
        route_dists = [route_distance(r) for r in current_routes]
        total_dist = sum(route_dists)
        if total_dist == 0:
            # all routes empty (unlikely)
            weights = [1.0 / truck_count] * truck_count
        else:
            weights = [d / total_dist for d in route_dists]

        # assign removal probability to each customer: proportional to its route weight
        cust_probs = []
        for r_idx, route in enumerate(current_routes):
            w = weights[r_idx]
            for cust in route[1:-1]:  # exclude depot
                cust_probs.append((cust, r_idx, w))
        if not cust_probs:
            break
        # normalize probabilities
        total_w = sum(p[2] for p in cust_probs)
        probs = [p[2]/total_w for p in cust_probs]
        # sample customers to remove without replacement
        remove_count = max(1, int(removal_fraction * len(cust_probs)))
        # ensure we don't sample more than available
        if remove_count > len(cust_probs):
            remove_count = len(cust_probs)
        sampled_indices = random.choices(range(len(cust_probs)), weights=probs, k=remove_count)
        # use set to avoid duplicates (though choices may repeat, but we want unique)
        to_remove = set()
        for idx in sampled_indices:
            to_remove.add(cust_probs[idx][0])
        # if we got fewer due to duplicates, fill up randomly
        while len(to_remove) < remove_count:
            remaining = [c for c in all_customers if c not in to_remove]
            if not remaining:
                break
            to_remove.add(random.choice(remaining))
        # remove customers from a copy of current_routes
        new_routes = [list(r) for r in current_routes]
        removed_list = []
        for node in to_remove:
            for r, route in enumerate(new_routes):
                if node in route:
                    pos = route.index(node)
                    new_routes[r] = route[:pos] + route[pos+1:]
                    if len(new_routes[r]) < 2:
                        new_routes[r] = [0, 0]
                    removed_list.append(node)
                    break
        # reconstruct using minimax insertion on removed customers (shuffled)
        random.shuffle(removed_list)
        unassigned = removed_list
        while unassigned:
            best_max = float('inf')
            best_total = float('inf')
            best_node = None
            best_route = None
            best_pos = None
            for node in unassigned:
                for r in range(truck_count):
                    route = new_routes[r]
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
                                d = route_distance(new_routes[rr])
                            if d > current_max:
                                current_max = d
                        if current_max < best_max or (current_max == best_max and new_dist < best_total):
                            best_max = current_max
                            best_total = new_dist
                            best_node = node
                            best_route = r
                            best_pos = pos
            new_routes[best_route].insert(best_pos, best_node)
            unassigned.remove(best_node)
        # apply 2-opt on each route (limited iterations)
        for r in range(truck_count):
            route = new_routes[r]
            if len(route) <= 3:
                continue
            improved = True
            local_iter = 0
            while improved and local_iter < 5:
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        old_dist = route_distance(route)
                        if new_dist < old_dist:
                            new_routes[r] = new_route
                            route = new_route
                            improved = True
                            break
                    if improved:
                        break
                local_iter += 1
        new_obj = objective(new_routes)
        # SA acceptance
        if new_obj < current_obj:
            accept = True
        else:
            T = T0 * (0.95 ** iteration)
            if T > 0:
                accept_prob = math.exp((current_obj - new_obj) / T)  # negative delta
                # Actually prob = exp(-(new_obj - current_obj)/T) = exp((current_obj-new_obj)/T)
                accept = random.random() < accept_prob
            else:
                accept = False
        if accept:
            current_routes = new_routes
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_routes = [list(r) for r in new_routes]
                report_best_vrp(best_routes)

    return best_routes