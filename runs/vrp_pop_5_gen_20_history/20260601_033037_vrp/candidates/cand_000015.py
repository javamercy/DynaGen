import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    num_cust = len(customers)
    if truck_count >= num_cust:
        routes = [[0, c, 0] for c in customers]
        for _ in range(truck_count - num_cust):
            routes.append([0, 0])
        return routes
    
    # Initialize routes as list of customer lists (no depot)
    routes = [[c] for c in customers]
    customer_to_route = {c: idx for idx, c in enumerate(customers)}
    first = {idx: c for idx, c in enumerate(customers)}
    last = {idx: c for idx, c in enumerate(customers)}
    
    # Compute savings
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            if s > 0:
                savings.append((s, i, j))
    savings.sort(key=lambda x: -x[0])
    
    # Merge routes
    for s, i, j in savings:
        if len(routes) <= truck_count:
            break
        if customer_to_route[i] == customer_to_route[j]:
            continue
        ri = customer_to_route[i]
        rj = customer_to_route[j]
        # Check if i and j are ends
        if not ((first[ri] == i and last[rj] == j) or (last[ri] == i and first[rj] == j) or
                (first[ri] == i and first[rj] == j) or (last[ri] == i and last[rj] == j)):
            continue
        route_i = routes[ri]
        route_j = routes[rj]
        merged = None
        for a in (route_i, route_i[::-1]):
            for b in (route_j, route_j[::-1]):
                if a[-1] == i and b[0] == j:
                    merged = a + b
                    break
                elif a[-1] == j and b[0] == i:
                    merged = a + b
                    break
            if merged is not None:
                break
        if merged is None:
            continue
        routes[ri] = merged
        routes.pop(rj)
        # Rebuild mappings
        customer_to_route.clear()
        first.clear()
        last.clear()
        for idx, r in enumerate(routes):
            for c in r:
                customer_to_route[c] = idx
            first[idx] = r[0]
            last[idx] = r[-1]
    
    # Fallback: merge if still too many routes
    while len(routes) > truck_count:
        route_i = routes[0]
        route_j = routes[1]
        merged = route_i + route_j
        routes[0] = merged
        routes.pop(1)
        customer_to_route.clear()
        first.clear()
        last.clear()
        for idx, r in enumerate(routes):
            for c in r:
                customer_to_route[c] = idx
            first[idx] = r[0]
            last[idx] = r[-1]
    
    # Build full routes with depot
    full_routes = [[0] + r + [0] for r in routes]
    
    # Compute initial max distance
    def compute_max(routes):
        return max(sum(distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1)) for r in routes)
    
    best_routes = [r[:] for r in full_routes]
    best_max = compute_max(best_routes)
    report_best_vrp(best_routes)  # initial incumbent
    
    # Local search: first-improvement relocate and swap
    max_iters = 10 * num_cust
    for _ in range(max_iters):
        improved = False
        # Relocate
        for cust in customers:
            src_idx = customer_to_route[cust]
            src_route = full_routes[src_idx]
            pos = src_route.index(cust)
            new_src = src_route[:pos] + src_route[pos+1:]
            for dst_idx in range(len(full_routes)):
                if dst_idx == src_idx:
                    continue
                dst_route = full_routes[dst_idx]
                for p in range(1, len(dst_route)):
                    new_dst = dst_route[:p] + [cust] + dst_route[p:]
                    # Ensure depot integrity
                    if new_dst[0] != 0 or new_dst[-1] != 0:
                        continue
                    new_routes = [r[:] for r in full_routes]
                    new_routes[src_idx] = new_src
                    new_routes[dst_idx] = new_dst
                    new_max = compute_max(new_routes)
                    if new_max < best_max:
                        best_max = new_max
                        best_routes = [r[:] for r in new_routes]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            full_routes = [r[:] for r in best_routes]
            # Update mapping
            customer_to_route.clear()
            for idx, r in enumerate(full_routes):
                for c in r[1:-1]:
                    customer_to_route[c] = idx
            report_best_vrp(best_routes)
            continue
        
        # Swap
        for cust1 in customers:
            for cust2 in customers:
                if cust1 >= cust2:
                    continue
                r1_idx = customer_to_route[cust1]
                r2_idx = customer_to_route[cust2]
                if r1_idx == r2_idx:
                    continue
                r1 = full_routes[r1_idx]
                r2 = full_routes[r2_idx]
                p1 = r1.index(cust1)
                p2 = r2.index(cust2)
                new_r1 = r1[:]
                new_r2 = r2[:]
                new_r1[p1] = cust2
                new_r2[p2] = cust1
                new_routes = [r[:] for r in full_routes]
                new_routes[r1_idx] = new_r1
                new_routes[r2_idx] = new_r2
                new_max = compute_max(new_routes)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [r[:] for r in new_routes]
                    improved = True
                    break
            if improved:
                break
        if improved:
            full_routes = [r[:] for r in best_routes]
            customer_to_route.clear()
            for idx, r in enumerate(full_routes):
                for c in r[1:-1]:
                    customer_to_route[c] = idx
            report_best_vrp(best_routes)
            continue
        if not improved:
            break
    
    # Ensure exactly truck_count routes
    while len(best_routes) > truck_count:
        dists = [(sum(distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1)), idx) for idx, r in enumerate(best_routes)]
        dists.sort()
        i = dists[0][1]
        j = dists[1][1]
        if i > j:
            i, j = j, i
        new_route = best_routes[i][:-1] + best_routes[j][1:]
        best_routes[i] = new_route
        best_routes.pop(j)
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    return best_routes