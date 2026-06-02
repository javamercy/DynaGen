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
    current_routes = [list(r) for r in routes]
    current_obj = best_obj

    # Parameters
    max_iter = min(50, 2 * n)
    T_start = 5.0
    T_end = 0.1

    for iteration in range(max_iter):
        # Determine removal strategy
        route_lengths = [route_distance(r) for r in current_routes]
        max_len_idx = max(range(truck_count), key=lambda i: route_lengths[i])

        # Compute contribution of each customer
        contribution = {}
        for r_idx, route in enumerate(current_routes):
            if len(route) <= 2:
                continue
            for i in range(1, len(route)-1):
                node = route[i]
                contrib = dist[route[i-1]][node] + dist[node][route[i+1]]
                if r_idx == max_len_idx:
                    contrib *= 2.0
                contribution[node] = contribution.get(node, 0) + contrib

        remove_frac = random.uniform(0.2, 0.4)
        remove_count = max(1, int(remove_frac * (n-1)))

        # Weighted random selection without replacement
        if not contribution:
            selected = set()
        else:
            total_contrib = sum(contribution.values())
            if total_contrib == 0:
                selected = set(list(contribution.keys())[:remove_count])
            else:
                nodes = list(contribution.keys())
                weights = [contribution[node] / total_contrib for node in nodes]
                cum_weights = []
                s = 0
                for w in weights:
                    s += w
                    cum_weights.append(s)
                selected = set()
                while len(selected) < remove_count and len(selected) < len(nodes):
                    r = random.random()
                    lo, hi = 0, len(cum_weights)-1
                    while lo < hi:
                        mid = (lo+hi)//2
                        if cum_weights[mid] < r:
                            lo = mid+1
                        else:
                            hi = mid
                    idx = lo
                    node = nodes[idx]
                    if node not in selected:
                        selected.add(node)

        # Remove customers
        removed_list = []
        new_routes = []
        for r_idx, route in enumerate(current_routes):
            new_route = [route[0]]
            for node in route[1:-1]:
                if node in selected:
                    removed_list.append(node)
                else:
                    new_route.append(node)
            new_route.append(0)
            new_routes.append(new_route)
            if len(new_routes[-1]) < 2:
                new_routes[-1] = [0, 0]

        random.shuffle(removed_list)

        # Reconstruct via minimax insertion with tie-breaking preferring routes not currently longest
        # Record current route distances before insertion
        current_dists = [route_distance(r) for r in new_routes]
        current_max_dist = max(current_dists)
        unassigned = removed_list
        while unassigned:
            best_candidates = []
            best_max = float('inf')
            best_total = float('inf')
            for node in unassigned:
                for r in range(truck_count):
                    route = new_routes[r]
                    for pos in range(1, len(route)):
                        # Compute new distance for route r after inserting node at pos
                        new_dist = 0
                        prev = route[0]
                        for k in range(1, len(route)):
                            if k == pos:
                                new_dist += dist[prev][node]
                                prev = node
                            new_dist += dist[prev][route[k]]
                            prev = route[k]
                        # new route distance for r
                        current_route_dist = current_dists[r]
                        # Compute new_max
                        new_max = current_max_dist
                        # Only need to update if r's new distance becomes larger than current max
                        if new_dist > new_max:
                            new_max = new_dist
                        # Tie-breaking: check if insertion is into the route that currently has max distance
                        is_current_max_route = (current_dists[r] == current_max_dist)
                        # We'll evaluate based on (new_max, is_current_max_route, new_dist)
                        if new_max < best_max:
                            best_max = new_max
                            best_total = new_dist
                            best_candidates = [(node, r, pos, is_current_max_route)]
                        elif new_max == best_max:
                            # Prefer candidate that is NOT inserting into current max route
                            if is_current_max_route:
                                # Current candidate inserts into max route
                                # Check if all existing best_candidates also insert into max route
                                all_max = all(c[3] for c in best_candidates)
                                if all_max:
                                    # All same, then compare total distance
                                    if new_dist < best_total:
                                        best_total = new_dist
                                        best_candidates = [(node, r, pos, is_current_max_route)]
                                    elif new_dist == best_total:
                                        best_candidates.append((node, r, pos, is_current_max_route))
                                else:
                                    # Current candidate is worse because it inserts into max route
                                    pass
                            else:
                                # Current candidate inserts into non-max route
                                # Check if any existing best_candidate is also non-max
                                non_max_exists = any(not c[3] for c in best_candidates)
                                if non_max_exists:
                                    # Compare total distance among non-max candidates
                                    # Only consider candidates that are non-max for best_total comparison
                                    # For simplicity, we just compare new_dist with best_total among current_set
                                    # But we need to update best_total accordingly
                                    # Since we have best_candidates, we can filter those with not is_current_max_route
                                    pass
                                else:
                                    # All existing are max route, so current candidate is better
                                    best_total = new_dist
                                    best_candidates = [(node, r, pos, is_current_max_route)]
            if not best_candidates:
                break
            # Choose candidate: prefer those not inserting into current max route
            # Filter candidates where is_current_max_route is False
            non_max_candidates = [c for c in best_candidates if not c[3]]
            if non_max_candidates:
                # Among non-max, choose by total distance (best_total already reflects min total?)
                # best_total might be from other candidates; we select by min total among non_max_candidates
                min_total = min(c[2] for c in non_max_candidates)
                final_candidates = [c for c in non_max_candidates if c[2] == min_total]
                chosen = random.choice(final_candidates)
            else:
                # All are max route, select by total distance
                min_total = min(c[2] for c in best_candidates)
                final_candidates = [c for c in best_candidates if c[2] == min_total]
                chosen = random.choice(final_candidates)
            node, best_route, best_pos, _ = chosen
            new_routes[best_route].insert(best_pos, node)
            # Update current_dists and current_max_dist
            new_route_dist = route_distance(new_routes[best_route])
            current_dists[best_route] = new_route_dist
            current_max_dist = max(current_dists)
            unassigned.remove(node)

        # Intra-route 2-opt limited
        for r_idx in range(truck_count):
            route = new_routes[r_idx]
            if len(route) <= 3:
                continue
            for _ in range(10):
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
            new_routes[r_idx] = route

        new_obj = objective(new_routes)
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [list(r) for r in new_routes]
            # report_best_vrp(best_routes)  # internal call

        # Simulated annealing acceptance
        T = T_start - (T_start - T_end) * (iteration / max_iter)
        delta = new_obj - current_obj
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_routes = new_routes
            current_obj = new_obj

    return best_routes