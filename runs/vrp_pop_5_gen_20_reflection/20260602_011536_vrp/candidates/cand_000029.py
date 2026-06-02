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

    max_iter = min(50, 2 * n)
    T_start = 5.0
    T_end = 0.1
    for iteration in range(max_iter):
        current_routes = [list(r) for r in routes]
        # Compute contribution of each customer
        contribution = {}
        for r_idx, route in enumerate(current_routes):
            if len(route) <= 2:
                continue
            for i in range(1, len(route)-1):
                node = route[i]
                contrib = dist[route[i-1]][node] + dist[node][route[i+1]]
                contribution[node] = contribution.get(node, 0) + contrib
        sorted_cust = sorted(contribution.items(), key=lambda x: -x[1])
        # Adaptive removal fraction: linearly from 0.2 to 0.4
        remove_frac = 0.2 + 0.2 * iteration / max_iter
        remove_count = max(1, int(remove_frac * (n-1)))
        total_contrib = sum(c[1] for c in sorted_cust)
        if total_contrib == 0:
            to_remove = set(c[0] for c in sorted_cust[:remove_count])
        else:
            weights = [c[1]/total_contrib for c in sorted_cust]
            cum_weights = []
            s = 0
            for w in weights:
                s += w
                cum_weights.append(s)
            selected = set()
            while len(selected) < remove_count and len(selected) < len(sorted_cust):
                r = random.random()
                lo, hi = 0, len(cum_weights)-1
                while lo < hi:
                    mid = (lo+hi)//2
                    if cum_weights[mid] < r:
                        lo = mid+1
                    else:
                        hi = mid
                idx = lo
                node = sorted_cust[idx][0]
                if node not in selected:
                    selected.add(node)
            to_remove = selected

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

        random.shuffle(removed_list)
        unassigned = removed_list
        while unassigned:
            best_candidates = []
            best_max = float('inf')
            best_total = float('inf')
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
                        if current_max < best_max:
                            best_max = current_max
                            best_total = new_dist
                            best_candidates = [(node, r, pos)]
                        elif current_max == best_max:
                            if new_dist < best_total:
                                best_total = new_dist
                                best_candidates = [(node, r, pos)]
                            elif new_dist == best_total:
                                best_candidates.append((node, r, pos))
            if not best_candidates:
                break
            chosen = random.choice(best_candidates)
            node, best_route, best_pos = chosen
            current_routes[best_route].insert(best_pos, node)
            unassigned.remove(node)

        # Intra-route 2-opt with dynamic iterations
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            max_2opt_iter = min(5, len(route))
            for _ in range(max_2opt_iter):
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_d = route_distance(new_route)
                        old_d = route_distance(route)
                        if new_d < old_d:
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
        else:
            # Exponential temperature decay
            T = T_start * (T_end / T_start) ** (iteration / max_iter)
            delta = new_obj - objective(routes)
            if delta > 0 and random.random() < np.exp(-delta / T):
                pass
            else:
                continue
        routes = current_routes
        routes_objective = new_obj

    return best_routes