import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()
    random.seed(0)

    def route_distance(route):
        if len(route) < 2:
            return 0
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))

    def objective(routes):
        return max(route_distance(r) for r in routes)

    # ---- initial minimax construction ----
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    unassigned.sort(key=lambda x: dist[0][x], reverse=True)
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
                    new_route = route[:pos] + [node] + route[pos:]
                    new_routes = [list(routes[i]) for i in range(truck_count)]
                    new_routes[r] = new_route
                    new_max = 0
                    for rr in range(truck_count):
                        d = route_distance(new_routes[rr])
                        if d > new_max:
                            new_max = d
                    total = sum(route_distance(rr) for rr in new_routes)
                    if new_max < best_max or (new_max == best_max and total < best_total):
                        best_max = new_max
                        best_total = total
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

    # ---- SA parameters (simplified) ----
    max_iter = min(50, n * 2)
    removal_fraction = 0.3
    all_customers = list(range(1, n))

    for iteration in range(max_iter):
        # compute route distances for weighting
        route_dists = [route_distance(r) for r in current_routes]
        total_dist = sum(route_dists)
        if total_dist == 0:
            weights = [1.0 / truck_count] * truck_count
        else:
            weights = [d / total_dist for d in route_dists]
        # assign removal probability to each customer: proportional to its route weight
        cust_probs = []
        for r_idx, route in enumerate(current_routes):
            w = weights[r_idx]
            for cust in route[1:-1]:
                cust_probs.append((cust, r_idx, w))
        if not cust_probs:
            break
        total_w = sum(p[2] for p in cust_probs)
        probs = [p[2]/total_w for p in cust_probs]
        remove_count = max(1, int(removal_fraction * len(cust_probs)))
        if remove_count > len(cust_probs):
            remove_count = len(cust_probs)
        sampled_indices = random.choices(range(len(cust_probs)), weights=probs, k=remove_count)
        to_remove = set()
        for idx in sampled_indices:
            to_remove.add(cust_probs[idx][0])
        while len(to_remove) < remove_count and len(to_remove) < len(all_customers):
            remaining = [c for c in all_customers if c not in to_remove]
            if not remaining:
                break
            to_remove.add(random.choice(remaining))
        # remove customers
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
        # reconstruct with minimax insertion
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
                        new_route = route[:pos] + [node] + route[pos:]
                        new_routes_temp = [list(new_routes[i]) for i in range(truck_count)]
                        new_routes_temp[r] = new_route
                        new_max = 0
                        for rr in range(truck_count):
                            d = route_distance(new_routes_temp[rr])
                            if d > new_max:
                                new_max = d
                        total = sum(route_distance(rr) for rr in new_routes_temp)
                        if new_max < best_max or (new_max == best_max and total < best_total):
                            best_max = new_max
                            best_total = total
                            best_node = node
                            best_route = r
                            best_pos = pos
            new_routes[best_route].insert(best_pos, best_node)
            unassigned.remove(best_node)

        # ---- inter-route moves to balance ----
        # limited iterations
        for _ in range(n):
            # find longest and shortest routes (by distance)
            dists = [route_distance(r) for r in new_routes]
            max_idx = dists.index(max(dists))
            min_idx = dists.index(min(dists))
            if max_idx == min_idx:
                break
            # try relocate: move a customer from longest to shortest
            route_long = new_routes[max_idx]
            route_short = new_routes[min_idx]
            best_delta = float('inf')
            best_move = None
            for cust in route_long[1:-1]:  # exclude depots
                # remove cust from long
                new_long = route_long[:]
                pos_long = new_long.index(cust)
                new_long.pop(pos_long)
                if len(new_long) < 2:
                    new_long = [0, 0]
                long_dist = route_distance(new_long)
                # try inserting into short
                for pos in range(1, len(route_short)):
                    new_short = route_short[:pos] + [cust] + route_short[pos:]
                    short_dist = route_distance(new_short)
                    # compute new max
                    temp_routes = [list(r) for r in new_routes]
                    temp_routes[max_idx] = new_long
                    temp_routes[min_idx] = new_short
                    new_max = max(route_distance(r) for r in temp_routes)
                    if new_max < best_delta:
                        best_delta = new_max
                        best_move = (max_idx, min_idx, cust, pos)
            if best_move is None or best_delta >= objective(new_routes):
                # try swap between longest and shortest
                for cust1 in route_long[1:-1]:
                    for cust2 in route_short[1:-1]:
                        if cust1 == cust2:
                            continue
                        new_long = [0] + [c for c in route_long[1:-1] if c != cust1] + [cust2] + [0]
                        new_short = [0] + [c for c in route_short[1:-1] if c != cust2] + [cust1] + [0]
                        temp_routes = [list(r) for r in new_routes]
                        temp_routes[max_idx] = new_long
                        temp_routes[min_idx] = new_short
                        new_max = max(route_distance(r) for r in temp_routes)
                        if new_max < best_delta:
                            best_delta = new_max
                            best_move = ('swap', max_idx, min_idx, cust1, cust2)
                if best_move and best_delta < objective(new_routes):
                    if best_move[0] == 'swap':
                        _, mi, mj, c1, c2 = best_move
                        new_long = [0] + [c for c in new_routes[mi][1:-1] if c != c1] + [c2] + [0]
                        new_short = [0] + [c for c in new_routes[mj][1:-1] if c != c2] + [c1] + [0]
                        new_routes[mi] = new_long
                        new_routes[mj] = new_short
                else:
                    break
            else:
                # apply relocate
                _, mi, mj, cust, pos = best_move
                new_long = [c for c in new_routes[mi] if c != cust]
                if len(new_long) < 2:
                    new_long = [0, 0]
                new_short = new_routes[mj][:pos] + [cust] + new_routes[mj][pos:]
                new_routes[mi] = new_long
                new_routes[mj] = new_short

        # 2-opt on each route
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
                        if route_distance(new_route) < route_distance(route):
                            new_routes[r] = new_route
                            route = new_route
                            improved = True
                            break
                    if improved:
                        break
                local_iter += 1

        new_obj = objective(new_routes)
        # simple acceptance: always accept if better or equal, else with probability
        if new_obj < current_obj:
            accept = True
        else:
            T = 0.1 * current_obj * (0.95 ** iteration)
            if T > 0:
                accept_prob = math.exp((current_obj - new_obj) / T)
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