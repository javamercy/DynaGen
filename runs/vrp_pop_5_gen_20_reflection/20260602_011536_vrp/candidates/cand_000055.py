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
        best_node = None
        best_route = None
        best_pos = None
        best_new_route_dist = float('inf')
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
                    if (current_max < best_max) or (current_max == best_max and new_dist < best_new_route_dist):
                        best_max = current_max
                        best_node = node
                        best_route = r
                        best_pos = pos
                        best_new_route_dist = new_dist
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
        route_lengths = [route_distance(r) for r in current_routes]
        max_len_idx = max(range(truck_count), key=lambda i: route_lengths[i])

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

        # Reconstruction: minimax insertion with tie-breaking by new route distance (prefer smaller)
        unassigned = removed_list
        while unassigned:
            best_candidates = []
            best_max = float('inf')
            best_route_dist = float('inf')
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
                        if current_max < best_max:
                            best_max = current_max
                            best_route_dist = new_dist
                            best_candidates = [(node, r, pos)]
                        elif current_max == best_max:
                            if new_dist < best_route_dist:
                                best_route_dist = new_dist
                                best_candidates = [(node, r, pos)]
                            elif new_dist == best_route_dist:
                                best_candidates.append((node, r, pos))
            if not best_candidates:
                break
            chosen = random.choice(best_candidates)
            node, best_route, best_pos = chosen
            new_routes[best_route].insert(best_pos, node)
            unassigned.remove(node)

        # Inter-route relocate: move customers from longest route to shorter routes to reduce max
        route_lengths = [route_distance(r) for r in new_routes]
        max_route_idx = max(range(truck_count), key=lambda i: route_lengths[i])
        max_route = new_routes[max_route_idx]
        if len(max_route) > 2:
            # Try to relocate each customer from longest route to other routes
            for i in range(1, len(max_route)-1):
                node = max_route[i]
                best_delta = float('inf')
                best_route_idx = None
                best_pos = None
                for r in range(truck_count):
                    if r == max_route_idx:
                        continue
                    route = new_routes[r]
                    for pos in range(1, len(route)):
                        # Temporarily insert node into route r
                        new_route_r = route[:pos] + [node] + route[pos:]
                        new_dist_r = route_distance(new_route_r)
                        old_dist_r = route_distance(route)
                        # Remove node from max route
                        new_max_route = max_route[:i] + max_route[i+1:]
                        new_dist_max = 0
                        if len(new_max_route) < 2:
                            new_max_route = [0, 0]
                            new_dist_max = 0
                        else:
                            new_dist_max = route_distance(new_max_route)
                        old_dist_max = route_distance(max_route)
                        new_max_overall = max(new_dist_r, new_dist_max, max(route_distance(rr) for rr in new_routes if rr is not route and rr is not max_route))
                        old_max_overall = max(route_lengths)
                        delta = new_max_overall - old_max_overall
                        if delta < best_delta:
                            best_delta = delta
                            best_route_idx = r
                            best_pos = pos
                if best_delta < 0:
                    # Perform the move
                    node = max_route.pop(i)
                    # If after removal max_route becomes [0,], restore to [0,0]
                    if len(max_route) == 1:
                        max_route = [0, 0]
                        new_routes[max_route_idx] = max_route
                        max_route_idx = max(range(truck_count), key=lambda i: route_distance(new_routes[i]))
                        max_route = new_routes[max_route_idx]
                        i = 0  # restart scanning from beginning (simplified)
                        break
                    new_routes[best_route_idx].insert(best_pos, node)
                    # Update max_route and lengths
                    route_lengths = [route_distance(r) for r in new_routes]
                    max_route_idx = max(range(truck_count), key=lambda i: route_lengths[i])
                    max_route = new_routes[max_route_idx]
                    i = 0  # restart from beginning of new longest route
                    continue
                # else continue to next customer

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

        T = T_start - (T_start - T_end) * (iteration / max_iter)
        delta = new_obj - current_obj
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_routes = new_routes
            current_obj = new_obj

    return best_routes