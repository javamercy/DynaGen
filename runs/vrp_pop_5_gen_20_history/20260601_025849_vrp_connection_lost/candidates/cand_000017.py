import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))

    def route_dist(route):
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += dist[a][b]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes) if routes else 0.0

    # Regret-2 construction with tie-breaking by max distance impact
    while unassigned:
        best_info = {}
        for c in unassigned:
            best = float('inf')
            second = float('inf')
            best_r = -1
            best_p = -1
            for r_idx, route in enumerate(routes):
                for i in range(len(route) - 1):
                    cost = dist[route[i]][c] + dist[c][route[i+1]] - dist[route[i]][route[i+1]]
                    if cost < best:
                        second = best
                        best = cost
                        best_r = r_idx
                        best_p = i + 1
                    elif cost < second:
                        second = cost
            best_info[c] = (best, second, best_r, best_p)

        candidates = []
        for c, (best, second, r_idx, pos) in best_info.items():
            regret = second - best if second != float('inf') else float('inf')
            # Compute new max if inserted at best position
            new_route = routes[r_idx][:pos] + [c] + routes[r_idx][pos:]
            new_route_dist = route_dist(new_route)
            other_max = max(route_dist(r) for i, r in enumerate(routes) if i != r_idx) if truck_count > 1 else 0.0
            new_max = max(new_route_dist, other_max)
            candidates.append((-regret, new_max, c, r_idx, pos))

        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, chosen_c, chosen_r, chosen_p = candidates[0]
        routes[chosen_r].insert(chosen_p, chosen_c)
        unassigned.remove(chosen_c)

    report_best_vrp(routes)
    best_routes = [list(r) for r in routes]
    best_max = max_dist(best_routes)

    max_iter = n * truck_count * 2
    for _ in range(max_iter):
        improved = False
        # 2-opt on each route (first improvement)
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_dist(new_route) < route_dist(route):
                        routes[r_idx] = new_route
                        improved = True
                        current_max = max_dist(routes)
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Remove customer with highest gain from longest route and reinsert
        current_max = max_dist(routes)
        longest_indices = [i for i, r in enumerate(routes) if route_dist(r) == current_max]
        if longest_indices:
            r_idx = longest_indices[0]
            route = routes[r_idx]
            if len(route) > 3:
                best_gain = -float('inf')
                best_pos = -1
                best_cust = -1
                for pos in range(1, len(route)-1):
                    cust = route[pos]
                    prev = route[pos-1]
                    next = route[pos+1]
                    gain = dist[prev][cust] + dist[cust][next] - dist[prev][next]
                    if gain > best_gain:
                        best_gain = gain
                        best_pos = pos
                        best_cust = cust
                # Remove best_cust
                removed_route = route[:best_pos] + route[best_pos+1:]
                # Best insertion across all routes (including the modified longest route)
                best_ins_cost = float('inf')
                best_ins_route = -1
                best_ins_pos = -1
                for other_idx, other_route in enumerate(routes):
                    if other_idx == r_idx:
                        target = removed_route
                    else:
                        target = other_route
                    for i in range(1, len(target)):
                        cost = dist[target[i-1]][best_cust] + dist[best_cust][target[i]] - dist[target[i-1]][target[i]]
                        if cost < best_ins_cost:
                            best_ins_cost = cost
                            best_ins_route = other_idx
                            best_ins_pos = i
                # Build new routes
                new_routes = [list(r) for r in routes]
                if best_ins_route == r_idx:
                    new_route = removed_route[:best_ins_pos] + [best_cust] + removed_route[best_ins_pos:]
                    new_routes[r_idx] = new_route
                else:
                    new_routes[r_idx] = removed_route
                    new_routes[best_ins_route] = new_routes[best_ins_route][:best_ins_pos] + [best_cust] + new_routes[best_ins_route][best_ins_pos:]
                new_max = max_dist(new_routes)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [list(r) for r in new_routes]
                    routes = new_routes
                    improved = True
                    report_best_vrp(best_routes)
        if not improved:
            break

    # Ensure each route has exactly [0,0] if empty
    final_routes = []
    for route in best_routes:
        if len(route) == 2:
            final_routes.append([0, 0])
        else:
            final_routes.append([0] + route[1:-1] + [0])
    return final_routes