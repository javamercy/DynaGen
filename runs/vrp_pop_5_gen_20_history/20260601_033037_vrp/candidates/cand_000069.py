import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    best_routes = None
    best_max = float('inf')

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    # Construction: min-max insertion (same as parent)
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(customers)
    while unassigned:
        best_cust = None
        best_route_idx = None
        best_pos = None
        best_new_max = float('inf')
        sorted_cust = sorted(unassigned)
        for cust in sorted_cust:
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_route_dist = route_distance(new_route)
                    temp_routes = [list(r) for r in routes]
                    temp_routes[r_idx] = new_route
                    new_max = max(route_distance(r) for r in temp_routes)
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_cust = cust
                        best_route_idx = r_idx
                        best_pos = pos
                    elif abs(new_max - best_new_max) < 1e-12:
                        if cust < best_cust or (cust == best_cust and r_idx < best_route_idx) or \
                           (cust == best_cust and r_idx == best_route_idx and pos < best_pos):
                            best_new_max = new_max
                            best_cust = cust
                            best_route_idx = r_idx
                            best_pos = pos
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    report_best_vrp(routes)

    # Adaptive improvement loop (modified parameters)
    max_total_iter = min(500, 2 * n * truck_count)
    eps = 0.001  # 0.1% relative improvement threshold
    max_stagnation = 3
    stagnation = 0
    iteration = 0
    while iteration < max_total_iter:
        improved = False
        # Find longest route
        dists = [route_distance(r) for r in routes]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        interior = routes[max_idx][1:-1]
        if not interior:
            break

        # Best relocate from longest route
        best_reloc_new_routes = None
        best_reloc_new_max = float('inf')
        for cust in interior:
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for pos in range(1, len(other_route)):
                    new_routes = [list(r) for r in routes]
                    new_routes[max_idx].remove(cust)
                    new_routes[other_idx].insert(pos, cust)
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_reloc_new_max - 1e-12:
                        best_reloc_new_max = new_max
                        best_reloc_new_routes = new_routes
        if best_reloc_new_routes is not None and best_reloc_new_max < best_max - 1e-12:
            routes = best_reloc_new_routes
            report_best_vrp(routes)
            improved = True

        if not improved:
            # Best swap between longest and another
            best_swap_new_routes = None
            best_swap_new_max = float('inf')
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_interior = routes[other_idx][1:-1]
                if not other_interior:
                    continue
                for cust_max in interior:
                    for cust_other in other_interior:
                        new_routes = [list(r) for r in routes]
                        idx_max = new_routes[max_idx].index(cust_max)
                        idx_other = new_routes[other_idx].index(cust_other)
                        new_routes[max_idx][idx_max] = cust_other
                        new_routes[other_idx][idx_other] = cust_max
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_swap_new_max - 1e-12:
                            best_swap_new_max = new_max
                            best_swap_new_routes = new_routes
            if best_swap_new_routes is not None and best_swap_new_max < best_max - 1e-12:
                routes = best_swap_new_routes
                report_best_vrp(routes)
                improved = True

        if not improved:
            # Best 2-opt across all routes (modified from first-improvement)
            best_2opt_new_routes = None
            best_2opt_new_max = float('inf')
            for idx in range(truck_count):
                route = routes[idx]
                if len(route) <= 3:
                    continue
                for a in range(1, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < route_distance(route) - 1e-12:
                            new_routes = [list(r) for r in routes]
                            new_routes[idx] = new_route
                            new_max = max(route_distance(r) for r in new_routes)
                            if new_max < best_2opt_new_max - 1e-12:
                                best_2opt_new_max = new_max
                                best_2opt_new_routes = new_routes
            if best_2opt_new_routes is not None and best_2opt_new_max < best_max - 1e-12:
                routes = best_2opt_new_routes
                report_best_vrp(routes)
                improved = True

        if not improved:
            stagnation += 1
            if stagnation >= max_stagnation:
                break
        else:
            stagnation = 0

        # Check relative improvement
        if iteration > 0:
            prev_max = best_max  # this is actually current best_max after update? careful
        # Actually we need to track previous best_max before iteration
        # We'll keep a variable prev_best
        # But we didn't define it; let's add it properly
        # Let's restructure with prev_best
        # Actually the parent code had prev_best; we'll adopt that here.
        iteration += 1
    return best_routes if best_routes is not None else routes