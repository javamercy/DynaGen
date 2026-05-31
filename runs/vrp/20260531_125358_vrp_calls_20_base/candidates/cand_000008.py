import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # Edge case: each customer gets its own truck
    if truck_count >= n - 1:
        routes = [[0, cust, 0] for cust in customers] + [[0, 0]] * (truck_count - len(customers))
        # call report_best_vrp if available (will be defined in execution environment)
        try:
            report_best_vrp(routes)
        except NameError:
            pass
        return routes

    # --- Clustering (farthest-first seeding) ---
    sorted_cust = sorted(customers, key=lambda c: distance_matrix[0][c], reverse=True)
    seeds = sorted_cust[:truck_count]
    clusters = [[s] for s in seeds]
    remaining = [c for c in customers if c not in seeds]
    for cust in remaining:
        best_cluster = 0
        best_dist = distance_matrix[cust][seeds[0]]
        for i in range(1, truck_count):
            d = distance_matrix[cust][seeds[i]]
            if d < best_dist:
                best_dist = d
                best_cluster = i
            elif d == best_dist and i < best_cluster:
                best_cluster = i
        clusters[best_cluster].append(cust)

    # --- Build initial routes (nearest neighbor per cluster) ---
    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total

    def build_route(cluster):
        if not cluster:
            return [0, 0]
        route = [0]
        unvisited = set(cluster)
        current = 0
        while unvisited:
            next_node = min(unvisited, key=lambda x: (distance_matrix[current][x], x))
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        route.append(0)
        return route

    routes = [build_route(clusters[i]) for i in range(truck_count)]
    lengths = [route_length(r) for r in routes]
    best_routes = [r[:] for r in routes]
    best_max = max(lengths)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    # --- Local search (improve max route distance) ---
    max_iter = n * 10
    for _ in range(max_iter):
        improved = False

        # Relocate move
        for t1 in range(truck_count):
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            for idx1 in range(1, len(route1)-1):
                cust = route1[idx1]
                pred1 = route1[idx1-1]
                succ1 = route1[idx1+1]
                delta1 = -distance_matrix[pred1][cust] - distance_matrix[cust][succ1] + distance_matrix[pred1][succ1]
                for t2 in range(truck_count):
                    if t2 == t1:
                        continue
                    route2 = routes[t2]
                    for pos2 in range(1, len(route2)):
                        pred2 = route2[pos2-1]
                        succ2 = route2[pos2]
                        delta2 = -distance_matrix[pred2][succ2] + distance_matrix[pred2][cust] + distance_matrix[cust][succ2]
                        new_len1 = lengths[t1] + delta1
                        new_len2 = lengths[t2] + delta2
                        other_lengths = [lengths[t] for t in range(truck_count) if t not in (t1, t2)]
                        new_max = max(new_len1, new_len2, max(other_lengths) if other_lengths else 0)
                        if new_max < best_max - 1e-9:
                            # Apply move
                            route1.pop(idx1)
                            route2.insert(pos2, cust)
                            lengths[t1] = new_len1
                            lengths[t2] = new_len2
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
                            improved = True
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Swap move (inter-route only)
        for t1 in range(truck_count):
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            for idx1 in range(1, len(route1)-1):
                cust1 = route1[idx1]
                for t2 in range(t1+1, truck_count):
                    route2 = routes[t2]
                    if len(route2) <= 2:
                        continue
                    for idx2 in range(1, len(route2)-1):
                        cust2 = route2[idx2]
                        # delta for route1: remove cust1, insert cust2 at idx1
                        pred1 = route1[idx1-1]
                        succ1 = route1[idx1+1]
                        delta1_rem = -distance_matrix[pred1][cust1] - distance_matrix[cust1][succ1] + distance_matrix[pred1][succ1]
                        delta1_ins = -distance_matrix[pred1][succ1] + distance_matrix[pred1][cust2] + distance_matrix[cust2][succ1]
                        delta1 = delta1_rem + delta1_ins
                        # delta for route2: remove cust2, insert cust1 at idx2
                        pred2 = route2[idx2-1]
                        succ2 = route2[idx2+1]
                        delta2_rem = -distance_matrix[pred2][cust2] - distance_matrix[cust2][succ2] + distance_matrix[pred2][succ2]
                        delta2_ins = -distance_matrix[pred2][succ2] + distance_matrix[pred2][cust1] + distance_matrix[cust1][succ2]
                        delta2 = delta2_rem + delta2_ins
                        new_len1 = lengths[t1] + delta1
                        new_len2 = lengths[t2] + delta2
                        other_lengths = [lengths[t] for t in range(truck_count) if t not in (t1, t2)]
                        new_max = max(new_len1, new_len2, max(other_lengths) if other_lengths else 0)
                        if new_max < best_max - 1e-9:
                            # Apply swap
                            route1[idx1] = cust2
                            route2[idx2] = cust1
                            lengths[t1] = new_len1
                            lengths[t2] = new_len2
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
                            improved = True
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # 2-opt within route
        for t in range(truck_count):
            route = routes[t]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    pred = route[i-1]
                    succ = route[j+1]
                    old_edges = distance_matrix[pred][route[i]] + distance_matrix[route[j]][succ]
                    new_edges = distance_matrix[pred][route[j]] + distance_matrix[route[i]][succ]
                    delta = new_edges - old_edges
                    if delta < -1e-9:
                        new_len = lengths[t] + delta
                        other_lengths = [lengths[t2] for t2 in range(truck_count) if t2 != t]
                        new_max = max(new_len, max(other_lengths) if other_lengths else 0)
                        if new_max < best_max - 1e-9:
                            route[i:j+1] = reversed(route[i:j+1])
                            lengths[t] = new_len
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
                            improved = True
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Cross-route 2-opt* (tail exchange)
        for t1 in range(truck_count):
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            for t2 in range(t1+1, truck_count):
                route2 = routes[t2]
                if len(route2) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    for j in range(1, len(route2)-1):
                        old1 = distance_matrix[route1[i]][route1[i+1]]
                        old2 = distance_matrix[route2[j]][route2[j+1]]
                        new1 = distance_matrix[route1[i]][route2[j+1]]
                        new2 = distance_matrix[route2[j]][route1[i+1]]
                        delta1 = new1 - old1
                        delta2 = new2 - old2
                        new_len1 = lengths[t1] + delta1
                        new_len2 = lengths[t2] + delta2
                        other_lengths = [lengths[t] for t in range(truck_count) if t not in (t1, t2)]
                        new_max = max(new_len1, new_len2, max(other_lengths) if other_lengths else 0)
                        if new_max < best_max - 1e-9:
                            tail1 = route1[i+1:-1]
                            tail2 = route2[j+1:-1]
                            new_route1 = route1[:i+1] + tail2 + [0]
                            new_route2 = route2[:j+1] + tail1 + [0]
                            routes[t1] = new_route1
                            routes[t2] = new_route2
                            lengths[t1] = new_len1
                            lengths[t2] = new_len2
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
                            improved = True
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    try:
        report_best_vrp(best_routes)
    except NameError:
        pass
    return best_routes