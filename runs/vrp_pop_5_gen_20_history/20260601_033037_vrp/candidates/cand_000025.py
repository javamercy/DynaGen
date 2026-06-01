import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    num_customers = len(customers)
    # If enough trucks, each customer alone
    if truck_count >= num_customers:
        routes = [[0, c, 0] for c in customers]
        for _ in range(truck_count - num_customers):
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    # Initialize each customer as a route (list of customers, no depot)
    route_list = [[c] for c in customers]
    first = {i: c for i, c in enumerate(customers)}
    last = {i: c for i, c in enumerate(customers)}
    # Compute savings
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            if s > 0:
                savings.append((s, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))
    # Map customer to route index
    customer_to_route = {c: idx for idx, c in enumerate(customers)}
    route_count = num_customers
    # Merge routes
    for s, i, j in savings:
        if route_count <= truck_count:
            break
        ri = customer_to_route[i]
        rj = customer_to_route[j]
        if ri == rj:
            continue
        # Try to merge: must match end-start
        route_a = route_list[ri]
        route_b = route_list[rj]
        merged = None
        # Check orientations: a_end = i or last[ri], b_start = j or first[rj]
        if last[ri] == i and first[rj] == j:
            merged = route_a + route_b
        elif last[ri] == i and last[rj] == j:
            merged = route_a + route_b[::-1]
        elif first[ri] == i and first[rj] == j:
            merged = route_a[::-1] + route_b
        elif first[ri] == i and last[rj] == j:
            merged = route_a[::-1] + route_b[::-1]
        if merged is None:
            continue
        # Merge: keep ri, remove rj
        route_list[ri] = merged
        route_list.pop(rj)
        # Update mappings
        for cust in merged:
            customer_to_route[cust] = ri
        first[ri] = merged[0]
        last[ri] = merged[-1]
        del first[rj]
        del last[rj]
        # Shift indices for routes after rj
        for key in list(customer_to_route.keys()):
            if customer_to_route[key] > rj:
                customer_to_route[key] -= 1
        for key in list(first.keys()):
            if key > rj:
                first[key-1] = first.pop(key)
                last[key-1] = last.pop(key)
        route_count -= 1
    # If still too many routes, merge the two smallest
    while route_count > truck_count:
        # Find two routes with fewest customers
        sizes = [(len(route_list[i]), i) for i in range(route_count)]
        sizes.sort()
        i = sizes[0][1]
        j = sizes[1][1]
        # Merge i and j: just concatenate (order may not matter for feasibility, but try to keep original order?)
        merged = route_list[i] + route_list[j]
        route_list[i] = merged
        route_list.pop(j)
        for cust in merged:
            customer_to_route[cust] = i
        first[i] = merged[0]
        last[i] = merged[-1]
        del first[j]
        del last[j]
        for key in list(customer_to_route.keys()):
            if customer_to_route[key] > j:
                customer_to_route[key] -= 1
        for key in list(first.keys()):
            if key > j:
                first[key-1] = first.pop(key)
                last[key-1] = last.pop(key)
        route_count -= 1
    # Convert to full routes with depot
    full_routes = [[0] + r + [0] for r in route_list]
    best_routes = [r[:] for r in full_routes]
    best_max = max(sum(distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1)) for r in full_routes)
    report_best_vrp(best_routes)
    # Local search: relocate and swap
    max_iters = 5 * num_customers
    for _ in range(max_iters):
        improved = False
        # Relocate each customer to a better route
        for cust in customers:
            src_idx = None
            src_pos = None
            for idx, r in enumerate(full_routes):
                if cust in r:
                    src_idx = idx
                    src_pos = r.index(cust)
                    break
            if src_idx is None:
                continue
            src_route = full_routes[src_idx]
            # Remove cust from src_route
            temp_src = src_route[:]
            temp_src.pop(src_pos)
            best_new_max = best_max
            best_move = None
            # Try inserting into every other route at every valid position
            for dst_idx in range(len(full_routes)):
                if dst_idx == src_idx:
                    continue
                dst_route = full_routes[dst_idx]
                # Positions from 1 to len(dst_route)-1 (inclusive) to keep depot at ends
                for pos in range(1, len(dst_route)):
                    new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                    # Compute new max
                    new_routes = full_routes[:]
                    new_routes[src_idx] = temp_src
                    new_routes[dst_idx] = new_dst
                    new_max = max(sum(distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1)) for r in new_routes)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = (src_idx, dst_idx, temp_src, new_dst)
            if best_move is not None:
                src_idx, dst_idx, new_src, new_dst = best_move
                full_routes[src_idx] = new_src
                full_routes[dst_idx] = new_dst
                best_max = best_new_max
                best_routes = [r[:] for r in full_routes]
                report_best_vrp(best_routes)
                improved = True
                break
        if improved:
            continue
        # Swap pairs of customers from different routes
        for cust1 in customers:
            for cust2 in customers:
                if cust1 >= cust2:
                    continue
                r1_idx = None
                r2_idx = None
                for idx, r in enumerate(full_routes):
                    if cust1 in r:
                        r1_idx = idx
                    if cust2 in r:
                        r2_idx = idx
                if r1_idx is None or r2_idx is None or r1_idx == r2_idx:
                    continue
                r1 = full_routes[r1_idx]
                r2 = full_routes[r2_idx]
                p1 = r1.index(cust1)
                p2 = r2.index(cust2)
                new_r1 = r1[:]
                new_r2 = r2[:]
                new_r1[p1] = cust2
                new_r2[p2] = cust1
                new_routes = full_routes[:]
                new_routes[r1_idx] = new_r1
                new_routes[r2_idx] = new_r2
                new_max = max(sum(distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1)) for r in new_routes)
                if new_max < best_max:
                    full_routes = new_routes
                    best_max = new_max
                    best_routes = [r[:] for r in full_routes]
                    report_best_vrp(best_routes)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    # Ensure exactly truck_count routes (should already be)
    while len(best_routes) > truck_count:
        # Merge two shortest (by distance) routes
        dists = []
        for idx, r in enumerate(best_routes):
            d = sum(distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1))
            dists.append((d, idx))
        dists.sort()
        i = dists[0][1]
        j = dists[1][1]
        new_route = best_routes[i][:-1] + best_routes[j][1:]
        best_routes[i] = new_route
        best_routes.pop(j)
        report_best_vrp(best_routes)
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    return best_routes