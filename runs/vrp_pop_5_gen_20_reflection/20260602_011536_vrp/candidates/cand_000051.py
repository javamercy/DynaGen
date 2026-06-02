import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()

    def route_distance(route):
        if len(route) < 2:
            return 0
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))

    def objective(routes):
        return max(route_distance(r) for r in routes)

    # Initial construction: greedy insertion minimizing max distance, then total distance
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    while unassigned:
        best_max = float('inf')
        best_total = float('inf')
        chosen = None
        for node in unassigned:
            for r in range(truck_count):
                route = routes[r]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [node] + route[pos:]
                    new_dist = route_distance(new_route)
                    current_max = max(route_distance(routes[rr]) if rr != r else new_dist for rr in range(truck_count))
                    if (current_max < best_max) or (current_max == best_max and new_dist < best_total):
                        best_max = current_max
                        best_total = new_dist
                        chosen = (node, r, pos)
        if chosen is None:
            break
        node, r, pos = chosen
        routes[r].insert(pos, node)
        unassigned.remove(node)

    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)
    # report_best_vrp(best_routes)

    # Simulated annealing parameters
    max_iter = min(50, 2 * n)
    T_start = 5.0
    T_end = 0.1
    current_routes = [list(r) for r in routes]
    current_obj = best_obj

    for it in range(max_iter):
        # Ruin: remove customers from routes with distance above average
        route_dists = [route_distance(r) for r in current_routes]
        avg_dist = sum(route_dists) / truck_count if truck_count > 0 else 0
        # Choose routes to ruin: those with distance > avg_dist
        candidates = []
        for r_idx in range(truck_count):
            if route_dists[r_idx] > avg_dist and len(current_routes[r_idx]) > 2:
                candidates.append(r_idx)
        if not candidates:
            # Fallback: longest route
            max_idx = max(range(truck_count), key=lambda i: route_dists[i])
            if len(current_routes[max_idx]) > 2:
                candidates = [max_idx]
        if not candidates:
            break  # no customers to remove

        # Select customers to remove from each candidate route
        removed = []
        for r_idx in candidates:
            route = current_routes[r_idx]
            if len(route) <= 2:
                continue
            # remove random fraction between 0.2 and 0.4 of customers in that route
            cust_count = len(route) - 2  # exclude depots
            if cust_count == 0:
                continue
            remove_frac = random.uniform(0.2, 0.4)
            remove_num = max(1, int(cust_count * remove_frac))
            # Randomly choose positions (1..len-2)
            positions = list(range(1, len(route)-1))
            random.shuffle(positions)
            chosen_pos = sorted(positions[:remove_num])
            # Remove from end to avoid index issues
            for pos in reversed(chosen_pos):
                removed.append(route.pop(pos))
            # Ensure route is [0,0] if empty
            if len(route) < 2:
                route = [0, 0]
                current_routes[r_idx] = route

        random.shuffle(removed)

        # Reconstruct with minimax insertion
        unassigned = removed
        while unassigned:
            best_max = float('inf')
            best_total = float('inf')
            chosen = None
            for node in unassigned:
                for r in range(truck_count):
                    route = current_routes[r]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [node] + route[pos:]
                        new_dist = route_distance(new_route)
                        new_max = max(route_distance(current_routes[rr]) if rr != r else new_dist for rr in range(truck_count))
                        if (new_max < best_max) or (new_max == best_max and new_dist < best_total):
                            best_max = new_max
                            best_total = new_dist
                            chosen = (node, r, pos)
            if chosen is None:
                break
            node, r, pos = chosen
            current_routes[r].insert(pos, node)
            unassigned.remove(node)

        # Local search: inter-route relocate from longest route to shorter ones
        for _ in range(10 * truck_count):
            improved = False
            current_max = objective(current_routes)
            # Find longest route
            route_dists = [route_distance(r) for r in current_routes]
            longest_idx = max(range(truck_count), key=lambda i: route_dists[i])
            if len(current_routes[longest_idx]) <= 2:
                break
            # Try relocate each customer from longest route to some other route
            best_delta = 0
            best_move = None
            for ci_idx in range(1, len(current_routes[longest_idx])-1):
                ci = current_routes[longest_idx][ci_idx]
                new_route_i = current_routes[longest_idx][:ci_idx] + current_routes[longest_idx][ci_idx+1:]
                for j in range(truck_count):
                    if j == longest_idx:
                        continue
                    for cj_idx in range(1, len(current_routes[j])):
                        new_route_j = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx:]
                        new_max = max(route_distance(new_route_i), route_distance(new_route_j))
                        for k in range(truck_count):
                            if k != longest_idx and k != j:
                                new_max = max(new_max, route_distance(current_routes[k]))
                        if new_max < current_max:
                            delta = current_max - new_max
                            if delta > best_delta:
                                best_delta = delta
                                best_move = (longest_idx, ci_idx, j, cj_idx)
            if best_move is not None:
                i, ci_idx, j, cj_idx = best_move
                ci = current_routes[i][ci_idx]
                del current_routes[i][ci_idx]
                if len(current_routes[i]) < 2:
                    current_routes[i] = [0, 0]
                current_routes[j].insert(cj_idx, ci)
                improved = True
            if not improved:
                break

        # Intra-route 2-opt (limited)
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            for _ in range(10):
                improved_opt = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_d = route_distance(new_route)
                        old_d = route_distance(route)
                        if new_d < old_d:
                            route = new_route
                            improved_opt = True
                            break
                    if improved_opt:
                        break
                if not improved_opt:
                    break
            current_routes[r_idx] = route

        new_obj = objective(current_routes)
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [list(r) for r in current_routes]
            # report_best_vrp(best_routes)
        # Simulated annealing acceptance
        T = T_start * (T_end / T_start) ** (it / max_iter)
        delta = new_obj - current_obj
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_routes = [list(r) for r in current_routes]
            current_obj = new_obj

    return best_routes