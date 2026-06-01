import numpy as np


def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix

    # Helper functions
    def route_distance(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += dist[route[i], route[i + 1]]
        return d

    def max_route_distance(routes):
        return max(route_distance(r) for r in routes) if routes else 0.0

    def insert_customer(route, pos, cust):
        return route[:pos] + [cust] + route[pos:]

    # Initial construction: regret-2 with tie-breaking on max distance impact
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))

    while unassigned:
        best_info = {}  # customer -> (best_cost, second_best, best_route_idx, best_pos)
        for c in unassigned:
            best = float('inf')
            second = float('inf')
            best_r = -1
            best_p = -1
            for r_idx, route in enumerate(routes):
                for i in range(len(route) - 1):
                    cost = dist[route[i]][c] + dist[c][route[i + 1]] - dist[route[i]][route[i + 1]]
                    if cost < best:
                        second = best
                        best = cost
                        best_r = r_idx
                        best_p = i + 1
                    elif cost < second:
                        second = cost
            best_info[c] = (best, second, best_r, best_p)

        # Compute regret and new max distance if inserted at best position
        candidates = []
        for c, (best, second, r_idx, pos) in best_info.items():
            regret = second - best if second != float('inf') else float('inf')
            new_route = insert_customer(routes[r_idx], pos, c)
            new_route_dist = route_distance(new_route)
            other_routes = [routes[i] for i in range(truck_count) if i != r_idx]
            other_max = max((route_distance(r) for r in other_routes), default=0.0)
            new_max = max(new_route_dist, other_max)
            candidates.append((-regret, new_max, c, r_idx, pos))

        # Sort: higher regret first (negated), then smaller new_max, then smaller customer ID
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, chosen_c, chosen_r, chosen_p = candidates[0]
        routes[chosen_r] = insert_customer(routes[chosen_r], chosen_p, chosen_c)
        unassigned.remove(chosen_c)

    # Initial best solution
    best_routes = [list(r) for r in routes]
    best_max = max_route_distance(best_routes)
    report_best_vrp(best_routes)

    # Improvement parameters
    max_iter = n * 2
    destroy_fraction = 0.2
    min_destroy = 1

    for iteration in range(max_iter):
        improved = False
        # --- LNS: destroy most expensive customers ---
        removal_gain = []
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route) - 1):
                cust = route[pos]
                prev = route[pos - 1]
                next = route[pos + 1]
                gain = dist[prev][cust] + dist[cust][next] - dist[prev][next]
                removal_gain.append((gain, r_idx, pos, cust))
        removal_gain.sort(key=lambda x: (-x[0], x[3]))  # descending gain, tie by customer ID

        destroy_count = max(min_destroy, int(n * destroy_fraction))
        destroy_count = min(destroy_count, len(removal_gain))
        removed_info = []
        for gain, r_idx, pos, cust in removal_gain[:destroy_count]:
            removed_info.append((r_idx, pos, cust))

        # Remove customers
        removed_by_route = {}
        for r_idx, pos, cust in removed_info:
            removed_by_route.setdefault(r_idx, []).append((pos, cust))
        for r_idx, items in removed_by_route.items():
            items.sort(key=lambda x: -x[0])
            route = routes[r_idx]
            for pos, cust in items:
                route = route[:pos] + route[pos + 1:]
            routes[r_idx] = route

        # Repair with regret-2 (same as construction but only for removed customers)
        unassigned = [c for _, _, c in removed_info]
        while unassigned:
            best_info = {}
            for c in unassigned:
                best = float('inf')
                second = float('inf')
                best_r = -1
                best_p = -1
                for r_idx, route in enumerate(routes):
                    for i in range(len(route) - 1):
                        cost = dist[route[i]][c] + dist[c][route[i + 1]] - dist[route[i]][route[i + 1]]
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
                candidates.append((-regret, len(unassigned), c, r_idx, pos))  # tie-breaking: simpler
            candidates.sort(key=lambda x: (x[0], x[2]))
            _, _, chosen_c, chosen_r, chosen_p = candidates[0]
            routes[chosen_r] = insert_customer(routes[chosen_r], chosen_p, chosen_c)
            unassigned.remove(chosen_c)

        # Evaluate new solution
        new_max = max_route_distance(routes)
        if new_max < best_max:
            best_max = new_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
            improved = True
        else:
            # Revert to best
            routes = [list(r) for r in best_routes]
            continue

        # --- 2-opt on each route ---
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            found_improvement = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    if route_distance(new_route) < route_distance(route):
                        routes[r_idx] = new_route
                        current_max = max_route_distance(routes)
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                        found_improvement = True
                        break
                if found_improvement:
                    break

        # --- Relocate from longest route ---
        current_max = max_route_distance(routes)
        longest_indices = [i for i, r in enumerate(routes) if route_distance(r) == current_max]
        if longest_indices:
            r_idx = longest_indices[0]
            route = routes[r_idx]
            if len(route) > 3:
                relocate_improved = False
                for pos in range(1, len(route) - 1):
                    cust = route[pos]
                    new_self = route[:pos] + route[pos + 1:]
                    for other_idx, other_route in enumerate(routes):
                        if other_idx == r_idx:
                            continue
                        for ins_pos in range(1, len(other_route)):
                            new_other = insert_customer(other_route, ins_pos, cust)
                            new_routes = [list(r) for r in routes]
                            new_routes[r_idx] = new_self
                            new_routes[other_idx] = new_other
                            new_max = max_route_distance(new_routes)
                            if new_max < best_max:
                                best_max = new_max
                                best_routes = [list(r) for r in new_routes]
                                routes = new_routes
                                report_best_vrp(best_routes)
                                relocate_improved = True
                                improved = True
                                break
                        if relocate_improved:
                            break
                    if relocate_improved:
                        break

        if not improved:
            break

    # Ensure exactly truck_count routes with depot endpoints
    final_routes = []
    for route in best_routes:
        if len(route) == 2:
            final_routes.append([0, 0])
        else:
            final_routes.append([0] + route[1:-1] + [0])
    return final_routes