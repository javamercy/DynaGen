import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # Initialize routes: each customer alone
    routes = [[0, i, 0] for i in range(1, n)]
    route_lengths = [distance_matrix[0][i] + distance_matrix[i][0] for i in range(1, n)]
    # While more routes than trucks, merge
    while len(routes) > truck_count:
        # Compute savings for all endpoint pairs across different routes
        savings = []
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                # endpoints of route i: first internal and last internal
                ri = routes[i]
                rj = routes[j]
                # endpoints in ri: ri[1] and ri[-2]
                # endpoints in rj: rj[1] and rj[-2]
                for a in (ri[1], ri[-2]):
                    if a == 0:
                        continue
                    for b in (rj[1], rj[-2]):
                        if b == 0:
                            continue
                        sav = distance_matrix[0][a] + distance_matrix[0][b] - distance_matrix[a][b]
                        # Check if merging is possible (a and b are true endpoints of their routes)
                        # also ensure that they are the endpoints (first or last internal)
                        if (ri[1] == a or ri[-2] == a) and (rj[1] == b or rj[-2] == b):
                            savings.append((sav, a, b, i, j))
        # Sort descending by savings, then by customer indices for tie-breaking
        savings.sort(key=lambda x: (-x[0], x[1], x[2]))
        merged = False
        for sav, a, b, i, j in savings:
            if i >= len(routes) or j >= len(routes):
                continue
            ri = routes[i]
            rj = routes[j]
            # Check if a and b are still endpoints (they might have changed due to previous merges)
            if (ri[1] == a or ri[-2] == a) and (rj[1] == b or rj[-2] == b):
                # Merge: connect a to b, ensure correct order
                # We need to reverse ri if a is not the first internal? Actually, we need to orient correctly.
                # Simplified: always orient ri so that a is at end, and rj so that b is at start.
                # Determine orientation for ri: if a == ri[1], then forward; else reverse.
                if a == ri[1]:
                    ri_seq = ri[1:-1]  # internal nodes from start
                else:
                    ri_seq = list(reversed(ri[1:-1]))
                # Similarly for rj: we want b at the start of the segment to connect to a
                if b == rj[1]:
                    rj_seq = rj[1:-1]
                else:
                    rj_seq = list(reversed(rj[1:-1]))
                # New route: [0] + ri_seq + rj_seq + [0]
                new_route = [0] + ri_seq + rj_seq + [0]
                # Remove old routes and add new
                routes.pop(max(i,j))
                routes.pop(min(i,j))
                routes.append(new_route)
                # Update lengths (optional, not needed further)
                merged = True
                break
        if not merged:
            # fallback: force merge any two routes (e.g., first two)
            if len(routes) >= 2:
                r1 = routes.pop(0)
                r2 = routes.pop(0)
                # combine by concatenating internal nodes
                new_route = [0] + r1[1:-1] + r2[1:-1] + [0]
                routes.append(new_route)
            else:
                break
    # Ensure exactly truck_count routes (add empty if less)
    while len(routes) < truck_count:
        routes.append([0,0])
    # Compute initial max distance
    def route_length(route):
        if len(route) == 2:
            return 0.0
        length = 0.0
        for k in range(len(route)-1):
            length += distance_matrix[route[k]][route[k+1]]
        return length
    current_max = max(route_length(r) for r in routes)
    # Report initial best
    report_best_vrp(routes)
    # Improvement: move customer from longest route to another
    improved = True
    max_iter = 100 * n  # bounded
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        # Identify longest route
        lengths = [route_length(r) for r in routes]
        max_idx = max(range(len(routes)), key=lambda i: lengths[i])
        longest_route = routes[max_idx]
        if len(longest_route) <= 2:
            break
        # Consider moving each internal customer in longest route to another route
        for c in longest_route[1:-1]:
            # Try to insert into other routes
            for other_idx in range(len(routes)):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                # Try all insertion positions
                for pos in range(1, len(other_route)):
                    new_route = other_route[:pos] + [c] + other_route[pos:]
                    new_length = route_length(new_route)
                    # New longest route after removal: we remove c from longest_route
                    new_longest_route = [x for x in longest_route if x != c]  # preserve order? Actually we need to maintain order, but remove one node.
                    # Reconstruct longest route after removal (should be Hamiltonian path still)
                    # Since we remove a node, we need to connect its neighbors.
                    # Easiest: recompute the route after removal by removing the node and connecting adjacent nodes.
                    # But we don't have adjacency, we need to ensure it's valid. Better approach: we'll just simulate moving by creating a copy of routes.
                    # For simplicity, we'll compute max distance if we perform the move:
                    # Create modified routes list
                    new_routes = [list(r) for r in routes]
                    # Remove c from new_routes[max_idx]
                    idx_c = new_routes[max_idx].index(c)
                    new_routes[max_idx].pop(idx_c)
                    # Insert c into new_routes[other_idx] at best position? We already have new_route for other_idx, but we need to update it.
                    # Actually simpler: we compute new max length after removal and insertion separately.
                    # This is getting complex. We'll use swap heuristic instead.
                    pass
        # Simple: for each pair of routes, try swapping two customers
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                ri = routes[i]
                rj = routes[j]
                if len(ri) <= 2 or len(rj) <= 2:
                    continue
                for ci in ri[1:-1]:
                    for cj in rj[1:-1]:
                        # swap ci and cj
                        # Create copies
                        new_ri = list(ri)
                        new_rj = list(rj)
                        # Replace ci with cj in ri, and cj with ci in rj
                        idx_ci = new_ri.index(ci)
                        idx_cj = new_rj.index(cj)
                        new_ri[idx_ci] = cj
                        new_rj[idx_cj] = ci
                        new_lengths = [route_length(new_ri), route_length(new_rj)]
                        for k in range(len(routes)):
                            if k != i and k != j:
                                new_lengths.append(route_length(routes[k]))
                        new_max = max(new_lengths)
                        if new_max < current_max:
                            # accept swap
                            routes[i] = new_ri
                            routes[j] = new_rj
                            current_max = new_max
                            improved = True
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return routes