import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    unrouted = set(range(1, n))

    # Greedy insertion
    while unrouted:
        best_customer = None
        best_route_idx = None
        best_pos = None
        best_cost = float('inf')
        for cust in sorted(unrouted):
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if cost < best_cost or (cost == best_cost and (cust < best_customer or (cust == best_customer and r_idx < best_route_idx))):
                        best_cost = cost
                        best_customer = cust
                        best_route_idx = r_idx
                        best_pos = pos
        routes[best_route_idx].insert(best_pos, best_customer)
        unrouted.remove(best_customer)

    def route_dist(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist

    dists = [route_dist(r) for r in routes]
    best_max = max(dists)
    best_routes = [r[:] for r in routes]
    # report initial best
    # report_best_vrp(best_routes)  # uncomment if report function available

    max_iter = n * truck_count
    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1

        # Intra-route 2-opt
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            old_dist = route_dist(route)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < old_dist - 1e-9:
                        routes[r_idx] = new_route
                        dists[r_idx] = new_dist
                        old_dist = new_dist
                        # Check if overall max improved
                        new_max = max(dists)
                        if new_max < best_max - 1e-9:
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
                            # report_best_vrp(best_routes)
                        improved = True
                        # break out of loops to restart from first route
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # Inter-route relocate from max route
        max_idx = max(range(truck_count), key=lambda i: dists[i])
        max_route = routes[max_idx]
        if len(max_route) <= 2:
            continue
        old_max = dists[max_idx]
        old_dists = dists[:]
        # Try moving each customer from max route
        for cust_pos in range(1, len(max_route)-1):
            cust = max_route[cust_pos]
            temp_route = max_route[:cust_pos] + max_route[cust_pos+1:]
            for r_idx in range(truck_count):
                if r_idx == max_idx:
                    continue
                route_other = routes[r_idx]
                for pos in range(1, len(route_other)):
                    new_route_other = route_other[:pos] + [cust] + route_other[pos:]
                    new_max_candidate = max(
                        route_dist(temp_route),
                        route_dist(new_route_other),
                        max(dist for i, dist in enumerate(old_dists) if i not in (max_idx, r_idx))
                    )
                    if new_max_candidate < old_max - 1e-9:
                        # accept
                        routes[max_idx] = temp_route
                        routes[r_idx] = new_route_other
                        dists[max_idx] = route_dist(temp_route)
                        dists[r_idx] = route_dist(new_route_other)
                        new_max = max(dists)
                        if new_max < best_max - 1e-9:
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
                            # report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Final sanity: ensure all customers appear
    seen = set()
    for r in best_routes:
        for c in r[1:-1]:
            seen.add(c)
    for c in range(1, n):
        if c not in seen:
            best_routes[-1].insert(-1, c)

    return best_routes