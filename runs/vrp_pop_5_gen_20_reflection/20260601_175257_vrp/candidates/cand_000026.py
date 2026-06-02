import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    max_restarts = min(100, 5 * n)
    best_routes = None
    best_max = float('inf')

    for restart in range(max_restarts):
        # --- Noisy savings construction ---
        savings = []
        for i in range(1, n):
            for j in range(i+1, n):
                s = (distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]) * (1 + random.uniform(-0.1, 0.1))
                savings.append((s, i, j))
        savings.sort(key=lambda x: -x[0])

        # Initialize each customer as a separate route: [0, customer, 0]
        routes = [[0, c, 0] for c in range(1, n)]
        route_lengths = [2 * distance_matrix[0, c] for c in range(1, n)]
        active = list(range(len(routes)))  # indices of active routes

        # Merge until we have exactly truck_count routes
        while len(active) > truck_count:
            # Precompute first and last customer of each active route
            first_cust = {}
            last_cust = {}
            for idx in active:
                r = routes[idx]
                if len(r) == 3:
                    first_cust[idx] = r[1]
                    last_cust[idx] = r[1]
                else:
                    first_cust[idx] = r[1]
                    last_cust[idx] = r[-2]

            merged = False
            for s, i, j in savings:
                # Find routes where i is last and j is first
                i_route = None
                j_route = None
                for idx in active:
                    if last_cust[idx] == i:
                        i_route = idx
                    if first_cust[idx] == j:
                        j_route = idx
                if i_route is not None and j_route is not None and i_route != j_route:
                    # Merge routes: connect i_route's end with j_route's start
                    route_i = routes[i_route]
                    route_j = routes[j_route]
                    # new route = route_i[:-1] + route_j[1:]
                    new_route = route_i[:-1] + route_j[1:]
                    new_len = sum(distance_matrix[new_route[k], new_route[k+1]] for k in range(len(new_route)-1))
                    # Remove the two old routes, add the new one
                    active.remove(i_route)
                    active.remove(j_route)
                    # Replace with new route at some index (e.g., i_route)
                    routes[i_route] = new_route
                    route_lengths[i_route] = new_len
                    active.append(i_route)
                    # Mark j_route as inactive (we will not use its old data)
                    merged = True
                    break
            if not merged:
                break  # Should not happen, but safe

        # After merging, routes and route_lengths are updated; active contains the indices of the truck_count routes
        # Collect these routes into a list
        current_routes = [routes[idx] for idx in active]
        current_lengths = [route_lengths[idx] for idx in active]

        # Ensure exactly truck_count routes (if fewer, duplicate? Shouldn't happen)
        while len(current_routes) < truck_count:
            current_routes.append([0, 0])
            current_lengths.append(0.0)

        # --- Local search ---
        def local_search(routes, lengths):
            improved = True
            max_iter = 10 * n * truck_count
            while improved and max_iter > 0:
                improved = False
                max_iter -= 1
                # Intra-route 2-opt
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 3:
                        continue
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                            new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                            if new < old - 1e-9:
                                route[i:j+1] = reversed(route[i:j+1])
                                lengths[r_idx] -= old - new
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue

                # Inter-route relocate
                for r_from in range(truck_count):
                    route_from = routes[r_from]
                    if len(route_from) <= 2:
                        continue
                    for idx_c in range(1, len(route_from)-1):
                        c = route_from[idx_c]
                        prev = route_from[idx_c-1]
                        nxt = route_from[idx_c+1]
                        cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                        new_len_from = lengths[r_from] - cost_remove
                        for r_to in range(truck_count):
                            if r_to == r_from:
                                continue
                            route_to = routes[r_to]
                            for pos in range(1, len(route_to)):
                                prev_to = route_to[pos-1]
                                nxt_to = route_to[pos]
                                cost_insert = distance_matrix[prev_to, c] + distance_matrix[c, nxt_to] - distance_matrix[prev_to, nxt_to]
                                new_len_to = lengths[r_to] + cost_insert
                                new_max = max(lengths[:r_from] + [new_len_from] + lengths[r_from+1:r_to] + [new_len_to] + lengths[r_to+1:])
                                current_max = max(lengths)
                                if new_max < current_max - 1e-9:
                                    route_from.pop(idx_c)
                                    lengths[r_from] = new_len_from
                                    route_to.insert(pos, c)
                                    lengths[r_to] = new_len_to
                                    improved = True
                                    report_best_vrp(routes)
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue

                # Inter-route swap
                for r1 in range(truck_count):
                    route1 = routes[r1]
                    if len(route1) <= 2:
                        continue
                    for idx1 in range(1, len(route1)-1):
                        c1 = route1[idx1]
                        prev1 = route1[idx1-1]
                        nxt1 = route1[idx1+1]
                        cost_remove1 = distance_matrix[prev1, c1] + distance_matrix[c1, nxt1] - distance_matrix[prev1, nxt1]
                        for r2 in range(r1+1, truck_count):
                            route2 = routes[r2]
                            if len(route2) <= 2:
                                continue
                            for idx2 in range(1, len(route2)-1):
                                c2 = route2[idx2]
                                prev2 = route2[idx2-1]
                                nxt2 = route2[idx2+1]
                                cost_remove2 = distance_matrix[prev2, c2] + distance_matrix[c2, nxt2] - distance_matrix[prev2, nxt2]
                                cost_insert1 = distance_matrix[prev1, c2] + distance_matrix[c2, nxt1] - distance_matrix[prev1, nxt1]
                                new_len1 = lengths[r1] - cost_remove1 + cost_insert1
                                cost_insert2 = distance_matrix[prev2, c1] + distance_matrix[c1, nxt2] - distance_matrix[prev2, nxt2]
                                new_len2 = lengths[r2] - cost_remove2 + cost_insert2
                                new_max = max(lengths[:r1] + [new_len1] + lengths[r1+1:r2] + [new_len2] + lengths[r2+1:])
                                current_max = max(lengths)
                                if new_max < current_max - 1e-9:
                                    del route1[idx1]
                                    del route2[idx2]
                                    route1.insert(idx1, c2)
                                    route2.insert(idx2, c1)
                                    lengths[r1] = new_len1
                                    lengths[r2] = new_len2
                                    improved = True
                                    report_best_vrp(routes)
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            return routes, lengths

        # Apply local search
        current_routes, current_lengths = local_search(current_routes, current_lengths)
        current_max = max(current_lengths)

        if current_max < best_max:
            best_max = current_max
            best_routes = [route[:] for route in current_routes]

        # --- Perturbation (if not improved at restart level, but we do it always for exploration) ---
        for _ in range(10):
            # Choose a random route with at least one customer
            non_empty = [i for i, r in enumerate(current_routes) if len(r) > 2]
            if not non_empty:
                break
            r_idx = random.choice(non_empty)
            route = current_routes[r_idx]
            idx = random.randint(1, len(route)-2)
            c = route[idx]
            # Remove c
            prev = route[idx-1]
            nxt = route[idx+1]
            cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
            route.pop(idx)
            current_lengths[r_idx] -= cost_remove
            # Find best insertion position among all routes
            best_route = -1
            best_pos = -1
            best_new_max = max(current_lengths)
            for r_to in range(truck_count):
                rt = current_routes[r_to]
                for pos in range(1, len(rt)):
                    prev_to = rt[pos-1]
                    nxt_to = rt[pos]
                    cost_insert = distance_matrix[prev_to, c] + distance_matrix[c, nxt_to] - distance_matrix[prev_to, nxt_to]
                    new_len_to = current_lengths[r_to] + cost_insert
                    new_max = max(current_lengths[:r_to] + [new_len_to] + current_lengths[r_to+1:])
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_route = r_to
                        best_pos = pos
            # Reinsert at best position
            if best_route != -1:
                current_routes[best_route].insert(best_pos, c)
                # Recompute route length properly
                r = current_routes[best_route]
                current_lengths[best_route] = sum(distance_matrix[r[i], r[i+1]] for i in range(len(r)-1))
                # Run local search again
                current_routes, current_lengths = local_search(current_routes, current_lengths)
                new_max = max(current_lengths)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [route[:] for route in current_routes]
            else:
                # Reinsert back to original position
                route.insert(idx, c)
                current_lengths[r_idx] += cost_remove

    if best_routes is None:
        best_routes = current_routes  # fallback
    return best_routes