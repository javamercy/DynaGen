import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def compute_max_dist(routes):
        return max(route_distance(r) for r in routes)

    def build_solution(random_tie=False):
        routes = [[0, 0] for _ in range(truck_count)]
        route_distances = [0.0 for _ in range(truck_count)]
        unassigned = set(customers)
        while unassigned:
            candidates = []
            for cust in unassigned:
                costs = []
                for r in range(truck_count):
                    route = routes[r]
                    for pos in range(1, len(route)):
                        delta = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        new_dist = route_distances[r] + delta
                        costs.append((new_dist, r, pos))
                costs.sort(key=lambda x: x[0])
                best = costs[0]
                second = costs[1] if len(costs) > 1 else (float('inf'), -1, -1)
                regret = second[0] - best[0]
                candidates.append((regret, cust, best[1], best[2], best[0]))
            if random_tie:
                max_regret = max(c[0] for c in candidates)
                tied = [c for c in candidates if c[0] == max_regret]
                chosen = random.choice(tied)
            else:
                candidates.sort(key=lambda x: (-x[0], x[1]))
                chosen = candidates[0]
            regret, cust, r_idx, pos, new_dist = chosen
            routes[r_idx].insert(pos, cust)
            route_distances[r_idx] = new_dist
            unassigned.remove(cust)
        return routes, route_distances

    # Initial construction deterministic
    routes, route_distances = build_solution(random_tie=False)
    best_routes = [r[:] for r in routes]
    best_max = compute_max_dist(routes)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    current_routes = [r[:] for r in routes]
    current_max = best_max
    deviation = 0.05 * current_max  # smaller initial threshold
    total_iter = 400
    stagnation_counter = 0
    restart_flag = False

    def or_opt(route):
        # Try relocating a segment of length 1..3 within route
        if len(route) <= 3:
            return route, 0.0
        best_route = route[:]
        best_dist = route_distance(route)
        improved = False
        for seg_len in [2, 3]:
            if len(route) - 2 < seg_len:
                continue
            for i in range(1, len(route) - seg_len):
                segment = route[i:i+seg_len]
                remaining = route[:i] + route[i+seg_len:]
                for j in range(1, len(remaining)):
                    new_route = remaining[:j] + segment + remaining[j:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist - 1e-9:
                        best_dist = new_dist
                        best_route = new_route
                        improved = True
        return best_route, best_dist

    def cross_exchange(routes, route_distances):
        best_move = None
        best_new_max = current_max + deviation + 1
        for r1 in range(truck_count):
            if len(routes[r1]) <= 2:
                continue
            for r2 in range(r1+1, truck_count):
                if len(routes[r2]) <= 2:
                    continue
                route1 = routes[r1]
                route2 = routes[r2]
                # Try swapping segments of length up to 2
                for i in range(1, len(route1)-2):
                    for j in range(i+1, min(len(route1)-1, i+3)):
                        seg1 = route1[i:j+1]
                        for p in range(1, len(route2)-2):
                            for q in range(p+1, min(len(route2)-1, p+3)):
                                seg2 = route2[p:q+1]
                                new_route1 = route1[:i] + seg2 + route1[j+1:]
                                new_route2 = route2[:p] + seg1 + route2[q+1:]
                                new_dist1 = route_distance(new_route1)
                                new_dist2 = route_distance(new_route2)
                                candidate_max = max(new_dist1, new_dist2)
                                for rr in range(truck_count):
                                    if rr != r1 and rr != r2:
                                        candidate_max = max(candidate_max, route_distances[rr])
                                if candidate_max < best_new_max:
                                    best_new_max = candidate_max
                                    best_move = (r1, i, j+1, r2, p, q+1, new_route1, new_route2, new_dist1, new_dist2)
        return best_move, best_new_max

    for it in range(total_iter):
        improved = False

        # Apply Or-opt on each route
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            new_route, new_dist = or_opt(route)
            if new_dist < route_distances[r_idx] - 1e-9:
                current_routes[r_idx] = new_route
                route_distances[r_idx] = new_dist
                current_max = compute_max_dist(current_routes)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in current_routes]
                    try:
                        report_best_vrp(best_routes)
                    except NameError:
                        pass
                improved = True

        # Intra-route 2-opt (best improvement)
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            best_i = None
            best_j = None
            best_dist = route_distances[r_idx]
            for i in range(1, len(route)-2):
                for j in range(i+2, len(route)-1):
                    new_route = route[:i+1] + route[j:i:-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist - 1e-9:
                        best_dist = new_dist
                        best_i = i
                        best_j = j
            if best_i is not None:
                new_route = current_routes[r_idx][:best_i+1] + current_routes[r_idx][best_j:best_i:-1] + current_routes[r_idx][best_j+1:]
                current_routes[r_idx] = new_route
                route_distances[r_idx] = best_dist
                new_max = compute_max_dist(current_routes)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [r[:] for r in current_routes]
                    try:
                        report_best_vrp(best_routes)
                    except NameError:
                        pass
                improved = True

        # Inter-route relocate (best improvement)
        best_move = None
        best_new_max = current_max + deviation + 1
        for r_from in range(truck_count):
            if len(current_routes[r_from]) <= 2:
                continue
            for pos_from in range(1, len(current_routes[r_from])-1):
                cust = current_routes[r_from][pos_from]
                prev = current_routes[r_from][pos_from-1]
                nxt = current_routes[r_from][pos_from+1]
                delta_from = distance_matrix[prev, nxt] - distance_matrix[prev, cust] - distance_matrix[cust, nxt]
                new_from_dist = route_distances[r_from] + delta_from
                for r_to in range(truck_count):
                    if r_to == r_from:
                        continue
                    route_to = current_routes[r_to]
                    for pos_to in range(1, len(route_to)):
                        prev_to = route_to[pos_to-1]
                        next_to = route_to[pos_to]
                        delta_to = distance_matrix[prev_to, cust] + distance_matrix[cust, next_to] - distance_matrix[prev_to, next_to]
                        new_to_dist = route_distances[r_to] + delta_to
                        candidate_max = max(new_from_dist, new_to_dist)
                        for rr in range(truck_count):
                            if rr != r_from and rr != r_to:
                                candidate_max = max(candidate_max, route_distances[rr])
                        if candidate_max < best_new_max:
                            best_new_max = candidate_max
                            best_move = (r_from, pos_from, r_to, pos_to, new_from_dist, new_to_dist)
        if best_move is not None and best_new_max <= current_max + deviation:
            r_from, pos_from, r_to, pos_to, new_from_dist, new_to_dist = best_move
            cust = current_routes[r_from].pop(pos_from)
            current_routes[r_to].insert(pos_to, cust)
            route_distances[r_from] = new_from_dist
            route_distances[r_to] = new_to_dist
            current_max = compute_max_dist(current_routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
            improved = True

        # Inter-route swap (best improvement)
        best_move = None
        best_new_max = current_max + deviation + 1
        for r1 in range(truck_count):
            if len(current_routes[r1]) <= 2:
                continue
            for pos1 in range(1, len(current_routes[r1])-1):
                cust1 = current_routes[r1][pos1]
                prev1 = current_routes[r1][pos1-1]
                next1 = current_routes[r1][pos1+1]
                for r2 in range(r1+1, truck_count):
                    if len(current_routes[r2]) <= 2:
                        continue
                    for pos2 in range(1, len(current_routes[r2])-1):
                        cust2 = current_routes[r2][pos2]
                        prev2 = current_routes[r2][pos2-1]
                        next2 = current_routes[r2][pos2+1]
                        delta1 = distance_matrix[prev1, cust2] + distance_matrix[cust2, next1] - distance_matrix[prev1, cust1] - distance_matrix[cust1, next1]
                        new_dist1 = route_distances[r1] + delta1
                        delta2 = distance_matrix[prev2, cust1] + distance_matrix[cust1, next2] - distance_matrix[prev2, cust2] - distance_matrix[cust2, next2]
                        new_dist2 = route_distances[r2] + delta2
                        candidate_max = max(new_dist1, new_dist2)
                        for rr in range(truck_count):
                            if rr != r1 and rr != r2:
                                candidate_max = max(candidate_max, route_distances[rr])
                        if candidate_max < best_new_max:
                            best_new_max = candidate_max
                            best_move = (r1, pos1, r2, pos2, new_dist1, new_dist2)
        if best_move is not None and best_new_max <= current_max + deviation:
            r1, pos1, r2, pos2, new_dist1, new_dist2 = best_move
            cust1 = current_routes[r1][pos1]
            cust2 = current_routes[r2][pos2]
            current_routes[r1][pos1] = cust2
            current_routes[r2][pos2] = cust1
            route_distances[r1] = new_dist1
            route_distances[r2] = new_dist2
            current_max = compute_max_dist(current_routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
            improved = True

        # Cross-exchange (best improvement)
        best_move, best_new_max = cross_exchange(current_routes, route_distances)
        if best_move is not None and best_new_max <= current_max + deviation:
            r1, i, j, r2, p, q, new_route1, new_route2, new_dist1, new_dist2 = best_move
            current_routes[r1] = new_route1
            current_routes[r2] = new_route2
            route_distances[r1] = new_dist1
            route_distances[r2] = new_dist2
            current_max = compute_max_dist(current_routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
            improved = True

        # Update deviation
        if improved:
            stagnation_counter = 0
            deviation *= 0.99  # decay slowly but keep small
        else:
            stagnation_counter += 1
            deviation *= 0.999

        # Restart if stagnation for 100 iterations (longer than parent)
        if stagnation_counter >= 100:
            new_routes, new_distances = build_solution(random_tie=restart_flag)
            current_routes = [r[:] for r in new_routes]
            route_distances = new_distances[:]
            current_max = compute_max_dist(current_routes)
            deviation = 0.05 * current_max
            stagnation_counter = 0
            restart_flag = not restart_flag

    # Final steepest descent on longest route
    for _ in range(200):  # bounded
        max_dist = -1
        max_idx = -1
        for r in range(truck_count):
            d = route_distances[r]
            if d > max_dist:
                max_dist = d
                max_idx = r
        # Try all moves that involve the longest route
        improved = False
        # Relocate customers out of longest route
        r_from = max_idx
        if len(current_routes[r_from]) > 2:
            for pos_from in range(1, len(current_routes[r_from])-1):
                cust = current_routes[r_from][pos_from]
                prev = current_routes[r_from][pos_from-1]
                nxt = current_routes[r_from][pos_from+1]
                delta_from = distance_matrix[prev, nxt] - distance_matrix[prev, cust] - distance_matrix[cust, nxt]
                new_from_dist = route_distances[r_from] + delta_from
                for r_to in range(truck_count):
                    if r_to == r_from:
                        continue
                    route_to = current_routes[r_to]
                    for pos_to in range(1, len(route_to)):
                        prev_to = route_to[pos_to-1]
                        next_to = route_to[pos_to]
                        delta_to = distance_matrix[prev_to, cust] + distance_matrix[cust, next_to] - distance_matrix[prev_to, next_to]
                        new_to_dist = route_distances[r_to] + delta_to
                        new_max = max(new_from_dist, new_to_dist)
                        for rr in range(truck_count):
                            if rr != r_from and rr != r_to:
                                new_max = max(new_max, route_distances[rr])
                        if new_max < best_max - 1e-9:
                            # Apply move
                            new_routes = [r[:] for r in current_routes]
                            new_routes[r_from].pop(pos_from)
                            new_routes[r_to].insert(pos_to, cust)
                            new_dists = route_distances[:]
                            new_dists[r_from] = new_from_dist
                            new_dists[r_to] = new_to_dist
                            current_routes = new_routes
                            route_distances = new_dists
                            best_max = new_max
                            best_routes = [r[:] for r in current_routes]
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        # If no improvement from relocate, try swaps involving longest route
        if not improved and len(current_routes[max_idx]) > 2:
            r1 = max_idx
            for pos1 in range(1, len(current_routes[r1])-1):
                cust1 = current_routes[r1][pos1]
                prev1 = current_routes[r1][pos1-1]
                next1 = current_routes[r1][pos1+1]
                for r2 in range(truck_count):
                    if r2 == r1:
                        continue
                    if len(current_routes[r2]) <= 2:
                        continue
                    for pos2 in range(1, len(current_routes[r2])-1):
                        cust2 = current_routes[r2][pos2]
                        prev2 = current_routes[r2][pos2-1]
                        next2 = current_routes[r2][pos2+1]
                        delta1 = distance_matrix[prev1, cust2] + distance_matrix[cust2, next1] - distance_matrix[prev1, cust1] - distance_matrix[cust1, next1]
                        new_dist1 = route_distances[r1] + delta1
                        delta2 = distance_matrix[prev2, cust1] + distance_matrix[cust1, next2] - distance_matrix[prev2, cust2] - distance_matrix[cust2, next2]
                        new_dist2 = route_distances[r2] + delta2
                        new_max = max(new_dist1, new_dist2)
                        for rr in range(truck_count):
                            if rr != r1 and rr != r2:
                                new_max = max(new_max, route_distances[rr])
                        if new_max < best_max - 1e-9:
                            new_routes = [r[:] for r in current_routes]
                            new_routes[r1][pos1] = cust2
                            new_routes[r2][pos2] = cust1
                            new_dists = route_distances[:]
                            new_dists[r1] = new_dist1
                            new_dists[r2] = new_dist2
                            current_routes = new_routes
                            route_distances = new_dists
                            best_max = new_max
                            best_routes = [r[:] for r in current_routes]
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break

    return best_routes