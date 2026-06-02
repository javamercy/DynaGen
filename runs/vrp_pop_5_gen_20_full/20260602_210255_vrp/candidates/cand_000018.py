import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    # ---------- construction: greedy min-max insertion ----------
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    unassigned.sort(key=lambda c: (-distance_matrix[0][c], c))
    def compute_route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d
    route_dists = [compute_route_dist(r) for r in routes]
    for cust in unassigned:
        best_new_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                prev = route[pos-1]
                succ = route[pos]
                increase = distance_matrix[prev][cust] + distance_matrix[cust][succ] - distance_matrix[prev][succ]
                new_route_dist = route_dists[r_idx] + increase
                if new_route_dist < best_new_max:
                    new_max = new_route_dist
                    for other_idx, d in enumerate(route_dists):
                        if other_idx != r_idx and d > new_max:
                            new_max = d
                    if new_max < best_new_max or (new_max == best_new_max and r_idx < best_route_idx):
                        best_new_max = new_max
                        best_route_idx = r_idx
                        best_pos = pos
                elif new_route_dist == best_new_max:
                    if r_idx < best_route_idx:
                        best_new_max = new_max
                        best_route_idx = r_idx
                        best_pos = pos
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
        route_dists[best_route_idx] = compute_route_dist(route)
    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)
    # ---------- improvement ----------
    max_iter = n * truck_count * 10
    for _ in range(max_iter):
        current_max = max(route_dists)
        max_routes = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
        improved = False
        # 1. relocate: move the customer with largest distance contribution from a max route
        for r_idx in max_routes:
            route = routes[r_idx]
            if len(route) <= 2:
                continue
            best_contrib = -1.0
            best_pos_in = -1
            best_cust = -1
            for pos in range(1, len(route)-1):
                cust = route[pos]
                prev = route[pos-1]
                succ = route[pos+1]
                contribution = distance_matrix[prev][cust] + distance_matrix[cust][succ] - distance_matrix[prev][succ]
                if contribution > best_contrib + 1e-12:
                    best_contrib = contribution
                    best_pos_in = pos
                    best_cust = cust
            if best_cust == -1:
                continue
            for other_idx in range(truck_count):
                if other_idx == r_idx:
                    continue
                other_route = routes[other_idx]
                for pos in range(1, len(other_route)):
                    prev = other_route[pos-1]
                    succ = other_route[pos]
                    increase = distance_matrix[prev][best_cust] + distance_matrix[best_cust][succ] - distance_matrix[prev][succ]
                    new_dist_other = route_dists[other_idx] + increase
                    new_dist_r = route_dists[r_idx] - best_contrib
                    new_max = max(new_dist_r, new_dist_other)
                    for idx, d in enumerate(route_dists):
                        if idx != r_idx and idx != other_idx and d > new_max:
                            new_max = d
                    if new_max < best_max - 1e-12:
                        routes[r_idx].pop(best_pos_in)
                        route_dists[r_idx] = new_dist_r
                        routes[other_idx].insert(pos, best_cust)
                        route_dists[other_idx] = new_dist_other
                        best_max = new_max
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 2. relocate (standard): move a customer from a max route to another
        for r_idx in max_routes:
            route = routes[r_idx]
            for pos in range(1, len(route)-1):
                cust = route[pos]
                prev = route[pos-1]
                succ = route[pos+1]
                removal_change = distance_matrix[prev][succ] - (distance_matrix[prev][cust] + distance_matrix[cust][succ])
                new_dist_r = route_dists[r_idx] + removal_change
                for other_idx in range(truck_count):
                    if other_idx == r_idx:
                        continue
                    other_route = routes[other_idx]
                    for insert_pos in range(1, len(other_route)):
                        prev2 = other_route[insert_pos-1]
                        succ2 = other_route[insert_pos]
                        insertion_change = distance_matrix[prev2][cust] + distance_matrix[cust][succ2] - distance_matrix[prev2][succ2]
                        new_dist_other = route_dists[other_idx] + insertion_change
                        new_max = max(new_dist_r, new_dist_other)
                        for idx, d in enumerate(route_dists):
                            if idx != r_idx and idx != other_idx and d > new_max:
                                new_max = d
                        if new_max < best_max - 1e-12:
                            routes[r_idx].pop(pos)
                            route_dists[r_idx] = new_dist_r
                            routes[other_idx].insert(insert_pos, cust)
                            route_dists[other_idx] = new_dist_other
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 3. swap: two customers from different routes (one must be from max route)
        for r1 in max_routes:
            route1 = routes[r1]
            for pos1 in range(1, len(route1)-1):
                cust1 = route1[pos1]
                for r2 in range(truck_count):
                    if r2 == r1:
                        continue
                    route2 = routes[r2]
                    for pos2 in range(1, len(route2)-1):
                        cust2 = route2[pos2]
                        prev1 = route1[pos1-1]
                        succ1 = route1[pos1+1]
                        remove1 = distance_matrix[prev1][succ1] - (distance_matrix[prev1][cust1] + distance_matrix[cust1][succ1])
                        prev2 = route2[pos2-1]
                        succ2 = route2[pos2+1]
                        remove2 = distance_matrix[prev2][succ2] - (distance_matrix[prev2][cust2] + distance_matrix[cust2][succ2])
                        insert1 = distance_matrix[prev1][cust2] + distance_matrix[cust2][succ1] - distance_matrix[prev1][succ1]
                        insert2 = distance_matrix[prev2][cust1] + distance_matrix[cust1][succ2] - distance_matrix[prev2][succ2]
                        new_dist_r1 = route_dists[r1] + remove1 + insert1
                        new_dist_r2 = route_dists[r2] + remove2 + insert2
                        new_max = max(new_dist_r1, new_dist_r2)
                        for idx, d in enumerate(route_dists):
                            if idx != r1 and idx != r2 and d > new_max:
                                new_max = d
                        if new_max < best_max - 1e-12:
                            route1[pos1] = cust2
                            route2[pos2] = cust1
                            route_dists[r1] = new_dist_r1
                            route_dists[r2] = new_dist_r2
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 4. 2-opt within max route
        for r_idx in max_routes:
            route = routes[r_idx]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old_edges = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                    new_edges = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                    change = new_edges - old_edges
                    new_dist = route_dists[r_idx] + change
                    if new_dist < best_max - 1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        route_dists[r_idx] = compute_route_dist(route)
                        best_max = max(route_dists)
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 5. cross-route exchange: between max route and another
        for r1 in max_routes:
            route1 = routes[r1]
            for pos1 in range(0, len(route1)-1):
                if pos1 == len(route1)-1:
                    continue
                for r2 in range(truck_count):
                    if r2 == r1:
                        continue
                    route2 = routes[r2]
                    for pos2 in range(0, len(route2)-1):
                        if pos2 == len(route2)-1:
                            continue
                        delta1 = distance_matrix[route1[pos1]][route2[pos2+1]] - distance_matrix[route1[pos1]][route1[pos1+1]]
                        delta2 = distance_matrix[route2[pos2]][route1[pos1+1]] - distance_matrix[route2[pos2]][route2[pos2+1]]
                        new_dist_r1 = route_dists[r1] + delta1
                        new_dist_r2 = route_dists[r2] + delta2
                        new_max = max(new_dist_r1, new_dist_r2)
                        for idx, d in enumerate(route_dists):
                            if idx != r1 and idx != r2 and d > new_max:
                                new_max = d
                        if new_max < best_max - 1e-12:
                            tail1 = route1[pos1+1:]
                            tail2 = route2[pos2+1:]
                            route1 = route1[:pos1+1] + tail2
                            route2 = route2[:pos2+1] + tail1
                            routes[r1] = route1
                            routes[r2] = route2
                            route_dists[r1] = compute_route_dist(route1)
                            route_dists[r2] = compute_route_dist(route2)
                            best_max = max(route_dists)
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 6. shaking: swap customer with largest contribution from longest route with customer with smallest contribution from shortest route
        longest_idx = max(range(truck_count), key=lambda i: route_dists[i])
        short_idx = min(range(truck_count), key=lambda i: route_dists[i])
        if longest_idx == short_idx or len(routes[longest_idx]) <= 2 or len(routes[short_idx]) <= 2:
            break
        route_long = routes[longest_idx]
        route_short = routes[short_idx]
        # find customer with largest contribution in longest route
        best_contrib = -1.0
        best_pos_long = -1
        best_cust_long = -1
        for pos in range(1, len(route_long)-1):
            cust = route_long[pos]
            prev = route_long[pos-1]
            succ = route_long[pos+1]
            contrib = distance_matrix[prev][cust] + distance_matrix[cust][succ] - distance_matrix[prev][succ]
            if contrib > best_contrib + 1e-12:
                best_contrib = contrib
                best_pos_long = pos
                best_cust_long = cust
        if best_cust_long == -1:
            break
        # find customer with smallest contribution in shortest route
        worst_contrib = float('inf')
        best_pos_short = -1
        best_cust_short = -1
        for pos in range(1, len(route_short)-1):
            cust = route_short[pos]
            prev = route_short[pos-1]
            succ = route_short[pos+1]
            contrib = distance_matrix[prev][cust] + distance_matrix[cust][succ] - distance_matrix[prev][succ]
            if contrib < worst_contrib - 1e-12:
                worst_contrib = contrib
                best_pos_short = pos
                best_cust_short = cust
        if best_cust_short == -1:
            break
        # swap
        route_long[best_pos_long] = best_cust_short
        route_short[best_pos_short] = best_cust_long
        route_dists[longest_idx] = compute_route_dist(route_long)
        route_dists[short_idx] = compute_route_dist(route_short)
        new_max = max(route_dists)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        improved = True
        if not improved:
            break
    return best_routes