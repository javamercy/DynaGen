import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total

    best_routes = None
    best_max = float('inf')

    num_restarts = min(5, max(1, n-1))

    for restart in range(num_restarts):
        random.seed(restart)
        # Initialize routes: each customer gets its own route
        routes = [[0, c, 0] for c in customers]
        # Add empty routes if truck_count > number of customers
        while len(routes) < truck_count:
            routes.append([0, 0])
        # Merge until we have exactly truck_count non-empty routes (if more)
        while len([r for r in routes if len(r) > 2]) > truck_count:
            # Compute best merge among all pairs of non-empty routes
            best_i = best_j = -1
            best_new_max = float('inf')
            best_merged = None
            route_indices = [i for i, r in enumerate(routes) if len(r) > 2]
            for idx_a in range(len(route_indices)):
                i = route_indices[idx_a]
                for idx_b in range(idx_a+1, len(route_indices)):
                    j = route_indices[idx_b]
                    # Compute two possible merges: connect end of i to start of j, or end of j to start of i
                    ri = routes[i]
                    rj = routes[j]
                    # Option 1: ... ri[-2] (last customer) to rj[1] (first customer)
                    saving1 = distance_matrix[ri[-2]][0] + distance_matrix[0][rj[1]] - distance_matrix[ri[-2]][rj[1]]
                    merged1 = ri[:-1] + rj[1:]
                    dist1 = route_distance(ri) + route_distance(rj) - saving1
                    # Option 2: opposite orientation
                    saving2 = distance_matrix[rj[-2]][0] + distance_matrix[0][ri[1]] - distance_matrix[rj[-2]][ri[1]]
                    merged2 = rj[:-1] + ri[1:]
                    dist2 = route_distance(ri) + route_distance(rj) - saving2
                    if dist1 <= dist2:
                        merged = merged1
                        new_route_dist = dist1
                    else:
                        merged = merged2
                        new_route_dist = dist2
                    # Compute new max across all routes after this merge
                    other_max = 0.0
                    for k in range(len(routes)):
                        if k != i and k != j and len(routes[k]) > 2:
                            d = route_distance(routes[k])
                            if d > other_max:
                                other_max = d
                    new_max = max(new_route_dist, other_max)
                    # Add small noise to encourage diversity
                    noise = 1.0 + random.uniform(-0.1, 0.1)
                    noisy_max = new_max * noise
                    if noisy_max < best_new_max - 1e-9:
                        best_new_max = noisy_max
                        best_i = i
                        best_j = j
                        best_merged = merged
            if best_i == -1:
                break
            # Perform merge
            # Remove the two routes in descending order of index to avoid shifting
            i, j = best_i, best_j
            if i > j:
                i, j = j, i
            routes.pop(j)
            routes.pop(i)
            routes.append(best_merged)
            # Try to keep exactly truck_count routes, but we may have more after loop
        # After merging, ensure exactly truck_count routes: truncate or add empties
        nonempty = [r for r in routes if len(r) > 2]
        empties = [r for r in routes if len(r) == 2]
        if len(nonempty) > truck_count:
            # Need to further merge? Should not happen if loop ran correctly, but just in case merge arbitrarily
            while len(nonempty) > truck_count:
                # merge two shortest routes? For simplicity, merge any two
                r1 = nonempty.pop(0)
                r2 = nonempty.pop(0)
                # merge them arbitrarily (connect end of r1 to start of r2)
                merged = r1[:-1] + r2[1:]
                nonempty.append(merged)
            routes = nonempty
        elif len(nonempty) < truck_count:
            # add empty routes
            while len(nonempty) < truck_count:
                nonempty.append([0, 0])
            routes = nonempty
        else:
            routes = nonempty
        # Local search: intra-route 2-opt and inter-route relocate
        current_max = max(route_distance(r) for r in routes)
        for iteration in range(n * 2):
            improved = False
            # Intra 2-opt
            for i in range(truck_count):
                route = routes[i]
                if len(route) <= 3:
                    continue
                for a in range(1, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        new_dist = route_distance(new_route)
                        other_max = 0.0
                        for k in range(truck_count):
                            if k != i:
                                d = route_distance(routes[k])
                                if d > other_max:
                                    other_max = d
                        new_max = max(new_dist, other_max)
                        if new_max < current_max - 1e-9:
                            routes[i] = new_route
                            current_max = new_max
                            improved = True
                            try:
                                report_best_vrp(routes)
                            except:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Inter-route relocate: move a customer from longest route to another
            # Find longest route
            dists = [route_distance(r) for r in routes]
            longest_idx = max(range(truck_count), key=lambda x: dists[x])
            route_long = routes[longest_idx]
            if len(route_long) <= 2:
                continue
            for cust_idx in range(1, len(route_long)-1):
                cust = route_long[cust_idx]
                new_long = route_long[:cust_idx] + route_long[cust_idx+1:]
                for dest_idx in range(truck_count):
                    if dest_idx == longest_idx:
                        continue
                    dest_route = routes[dest_idx]
                    for pos in range(1, len(dest_route)):
                        new_dest = dest_route[:pos] + [cust] + dest_route[pos:]
                        new_long_dist = route_distance(new_long)
                        new_dest_dist = route_distance(new_dest)
                        other_max = 0.0
                        for k in range(truck_count):
                            if k not in (longest_idx, dest_idx):
                                d = route_distance(routes[k])
                                if d > other_max:
                                    other_max = d
                        new_max = max(new_long_dist, new_dest_dist, other_max)
                        if new_max < current_max - 1e-9:
                            routes[longest_idx] = new_long
                            routes[dest_idx] = new_dest
                            current_max = new_max
                            improved = True
                            try:
                                report_best_vrp(routes)
                            except:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        # Update best
        current_max = max(route_distance(r) for r in routes)
        if current_max < best_max - 1e-9:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            try:
                report_best_vrp(best_routes)
            except:
                pass

    if best_routes is None:
        best_routes = [[0, 0] for _ in range(truck_count)]
    # Ensure all customers are covered
    covered = set()
    for r in best_routes:
        for c in r:
            if c != 0:
                covered.add(c)
    missing = [c for c in customers if c not in covered]
    if missing:
        # Add missing customers to the shortest route (by distance) to minimize impact
        dists = [route_distance(r) for r in best_routes]
        idx = dists.index(min(dists))
        route = best_routes[idx]
        # Insert missing at the end of the route (before depot)
        for c in missing:
            route = route[:-1] + [c] + [0]
        best_routes[idx] = route
    # Ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    return best_routes