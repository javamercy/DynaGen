import numpy as np
import math
import random
import heapq
import itertools
import collections
import time

def report_best_vrp(routes):
    pass

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    customers = list(range(1, n))
    best_overall = None
    best_overall_max = float('inf')

    def compute_route_length(route):
        length = 0.0
        for i in range(len(route)-1):
            length += distance_matrix[route[i]][route[i+1]]
        return length

    def clarke_wright_initial():
        # Initialize each customer as its own route
        routes = [[0, c, 0] for c in customers]
        route_lengths = [distance_matrix[0][c] + distance_matrix[c][0] for c in customers]
        # Compute savings
        savings = []
        for i in range(len(customers)):
            for j in range(i+1, len(customers)):
                ci = customers[i]
                cj = customers[j]
                save = distance_matrix[0][ci] + distance_matrix[0][cj] - distance_matrix[ci][cj]
                savings.append((save, ci, cj))
        savings.sort(reverse=True, key=lambda x: x[0])
        # Merging
        # Track first and last customer of each route (since routes are [0, ..., 0])
        # We'll maintain a mapping from customer to its route index
        cust_to_route = {c: i for i, c in enumerate(customers)}
        # Merge until we have truck_count routes
        while len(routes) > truck_count:
            merged = False
            for (save, ci, cj) in savings:
                if ci not in cust_to_route or cj not in cust_to_route:
                    continue
                ri = cust_to_route[ci]
                rj = cust_to_route[cj]
                if ri == rj:
                    continue
                route_i = routes[ri]
                route_j = routes[rj]
                # Check if ci and cj are endpoints (first or last customer, excluding depot)
                # Since routes have [0, ..., 0], endpoints are route_i[1] and route_i[-2]
                if (route_i[1] == ci and route_j[-2] == cj) or (route_i[1] == ci and route_j[1] == cj) or (route_i[-2] == ci and route_j[-2] == cj) or (route_i[-2] == ci and route_j[1] == cj):
                    # Merge: combine routes by connecting ci and cj appropriately
                    # Need to keep depot at both ends. Determine orientation.
                    if route_i[1] == ci and route_j[-2] == cj:
                        # route_i: 0...ci...0, route_j: 0...cj...0
                        # merge as 0...ci + cj...0 (reverse route_j if needed? Actually need to keep orientation correct
                        # Better to build new route from scratch
                        # Remove depot ends for merging
                        inner_i = route_i[1:-1]
                        inner_j = route_j[1:-1]
                        if inner_i[-1] == ci and inner_j[0] == cj:
                            new_inner = inner_i + inner_j
                        elif inner_i[0] == ci and inner_j[0] == cj:
                            new_inner = inner_i[::-1] + inner_j
                        elif inner_i[-1] == ci and inner_j[-1] == cj:
                            new_inner = inner_i + inner_j[::-1]
                        elif inner_i[0] == ci and inner_j[-1] == cj:
                            new_inner = inner_i[::-1] + inner_j[::-1]
                        else:
                            # Should not happen
                            continue
                        new_route = [0] + new_inner + [0]
                        new_len = compute_route_length(new_route)
                        # Remove old routes and add merged
                        # Update cust_to_route for all customers in new_route interior
                        for cust in new_inner:
                            cust_to_route[cust] = ri  # use index ri, later we will adjust
                        # Remove route j
                        routes.pop(rj)
                        route_lengths.pop(rj)
                        # Update route i
                        routes[ri] = new_route
                        route_lengths[ri] = new_len
                        # Update indices for routes after rj
                        for k in range(rj, len(routes)):
                            for cust in routes[k][1:-1]:
                                cust_to_route[cust] = k
                        merged = True
                        break
                    # Similarly handle other orientations... but the above pattern covers most cases?
            if not merged:
                break
        # If we still have too many routes, we need to assign remaining customers to some routes? Actually should not happen with proper merging.
        return routes, route_lengths

    # But above merge logic is incomplete and buggy; we need a simpler approach.
    # Let's just use a standard savings merging: maintain a list of routes as sequences without depot for interior, then add depot at end.
    # We'll implement a more straightforward version:

    def savings_initial():
        # Each customer as a route: [0, c, 0]
        routes = [[0, c, 0] for c in customers]
        route_lengths = [distance_matrix[0][c] + distance_matrix[c][0] for c in customers]
        # Precompute savings
        savings = []
        for i in range(len(customers)):
            for j in range(i+1, len(customers)):
                ci = customers[i]
                cj = customers[j]
                save = distance_matrix[0][ci] + distance_matrix[0][cj] - distance_matrix[ci][cj]
                savings.append((save, ci, cj, i, j))
        savings.sort(reverse=True, key=lambda x: x[0])
        # For quick lookup of route index for a customer
        cust_route_idx = {c: i for i, c in enumerate(customers)}
        # To track if customer is an endpoint (first or last interior)
        # We'll maintain for each route its first and last interior customer
        first_last = {i: (customers[i], customers[i]) for i in range(len(customers))}
        while len(routes) > truck_count:
            merged = False
            for save, ci, cj, idx_i, idx_j in savings:
                if ci not in cust_route_idx or cj not in cust_route_idx:
                    continue
                ri = cust_route_idx[ci]
                rj = cust_route_idx[cj]
                if ri == rj:
                    continue
                # Check if ci is first or last in route ri, and similarly for cj
                first_i, last_i = first_last[ri]
                first_j, last_j = first_last[rj]
                if (ci == first_i and cj == first_j) or (ci == first_i and cj == last_j) or (ci == last_i and cj == first_j) or (ci == last_i and cj == last_j):
                    # Merge
                    # Extract interior sequences
                    seq_i = routes[ri][1:-1]
                    seq_j = routes[rj][1:-1]
                    # Determine new sequence based on orientation
                    if ci == first_i and cj == first_j:
                        new_seq = seq_i[::-1] + [cj] + seq_j[1:]? Actually need to handle properly
                    # Too complex; use a simpler alternative: just concatenate with appropriate reversal
                    # We'll use a standard trick: represent routes as lists of customers (without depot) and later wrap with depot
                    # Let's simplify: use lists of customers for routes without depot for easier manipulation
                    # Actually it's easier to just use a random restart or insertion heuristic than buggy savings.
            # If no merge possible, break.
            break
        # Fallback: if not enough merges, just use the initial routes (each customer alone) which may be too many routes? But truck_count is fixed; we need exactly truck_count routes. So we must enforce merging.
        # For simplicity, we'll fallback to random insertion if savings fails.
        return None

    # Since implementing full savings is tricky under time constraint, we'll use a simple nearest neighbor insertion for construction, but that may not be enough.
    # Better to use a random insertion with tie-breaking like in cand_000027, but improve by using multiple starts and better perturbation.
    # Alternatively, use a greedy insertion that minimizes max route distance at each step (like in cand_000027 but deterministic?)
    # We'll adopt the random insertion from cand_000027 but add a savings-based initialization for restart seeds.
    # Given the complexity, we'll rely on the proven random insertion and focus on improvement loops.
    # So we'll reuse the random insertion from parent with minor modifications.

    # For simplicity, we'll implement a solver that uses the same structure as cand_000027 but with increased restarts and adaptive perturbation.
    # We'll also add a simple nearest-neighbor construction as alternative initial solution with higher quality.

    def nearest_neighbor_initial():
        # Build routes one by one, for each route start from depot and repeatedly go to nearest unvisited customer until no more customers.
        # Then assign remaining customers to routes? Actually we need exactly truck_count routes.
        # We'll use a simple approach: first create an ordering of customers, then balance them across routes.
        # Better: use a cyclic insertion: each route takes turns picking the closest customer to its last node.
        unvisited = set(customers)
        routes = [[0] for _ in range(truck_count)]
        route_ends = [0]*truck_count
        while unvisited:
            for ri in range(truck_count):
                if not unvisited:
                    break
                last = route_ends[ri]
                # find nearest unvisited customer
                best = None
                best_dist = float('inf')
                for c in unvisited:
                    d = distance_matrix[last][c]
                    if d < best_dist:
                        best_dist = d
                        best = c
                if best is not None:
                    routes[ri].append(best)
                    route_ends[ri] = best
                    unvisited.remove(best)
        for ri in range(truck_count):
            routes[ri].append(0)
        route_lengths = [compute_route_length(r) for r in routes]
        return routes, route_lengths

    def random_insertion(seed):
        random.seed(seed)
        shuffled = customers[:]
        random.shuffle(shuffled)
        routes = [[0, 0] for _ in range(truck_count)]
        route_lengths = [0.0] * truck_count
        for cust in shuffled:
            best_max = float('inf')
            best_route = None
            best_pos = None
            best_len = None
            for ri, route in enumerate(routes):
                cur_len = route_lengths[ri]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    add = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                    new_len = cur_len + add
                    new_max = max(route_lengths[:ri] + [new_len] + route_lengths[ri+1:])
                    if new_max < best_max or (new_max == best_max and (best_len is None or new_len < best_len)):
                        best_max = new_max
                        best_route = ri
                        best_pos = pos
                        best_len = new_len
            route = routes[best_route]
            prev = route[best_pos-1]
            nxt = route[best_pos]
            route.insert(best_pos, cust)
            route_lengths[best_route] += distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
        return routes, route_lengths

    def perturb(routes, route_lengths, strength=0.2):
        n_cust = sum(len(r)-2 for r in routes)
        num_move = max(1, int(n_cust * strength))
        all_custs = []
        cust_route = {}
        for i, r in enumerate(routes):
            for c in r[1:-1]:
                all_custs.append(c)
                cust_route[c] = i
        to_move = random.sample(all_custs, min(num_move, len(all_custs)))
        for c in to_move:
            ri = cust_route[c]
            r = routes[ri]
            pos = r.index(c)
            prev = r[pos-1]
            nxt = r[pos+1]
            removal_delta = distance_matrix[prev][nxt] - distance_matrix[prev][c] - distance_matrix[c][nxt]
            route_lengths[ri] += removal_delta
            r.pop(pos)
        random.shuffle(to_move)
        for cust in to_move:
            best_max = float('inf')
            best_route = None
            best_pos = None
            best_len = None
            for ri, route in enumerate(routes):
                cur_len = route_lengths[ri]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    add = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                    new_len = cur_len + add
                    new_max = max(route_lengths[:ri] + [new_len] + route_lengths[ri+1:])
                    if new_max < best_max or (new_max == best_max and (best_len is None or new_len < best_len)):
                        best_max = new_max
                        best_route = ri
                        best_pos = pos
                        best_len = new_len
            route = routes[best_route]
            prev = route[best_pos-1]
            nxt = route[best_pos]
            route.insert(best_pos, cust)
            route_lengths[best_route] += distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]

    def local_search(routes, route_lengths, best_max):
        max_passes = min(100, n * truck_count)
        improved_global = False
        for _ in range(max_passes):
            improved = False
            # Inter-route relocate
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for pos_i in range(1, len(route_i)-1):
                    cust = route_i[pos_i]
                    prev_i = route_i[pos_i-1]
                    next_i = route_i[pos_i+1]
                    removal_delta = distance_matrix[prev_i][next_i] - distance_matrix[prev_i][cust] - distance_matrix[cust][next_i]
                    new_len_i = route_lengths[i] + removal_delta
                    for j in range(truck_count):
                        if j == i:
                            continue
                        route_j = routes[j]
                        for pos_j in range(1, len(route_j)):
                            prev_j = route_j[pos_j-1]
                            next_j = route_j[pos_j]
                            insert_delta = distance_matrix[prev_j][cust] + distance_matrix[cust][next_j] - distance_matrix[prev_j][next_j]
                            new_len_j = route_lengths[j] + insert_delta
                            new_max = max(new_len_i, new_len_j, max(route_lengths[k] for k in range(truck_count) if k not in (i, j)))
                            if new_max < best_max:
                                route_i.pop(pos_i)
                                route_j.insert(pos_j, cust)
                                route_lengths[i] = new_len_i
                                route_lengths[j] = new_len_j
                                best_max = new_max
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                improved_global = True
                continue

            # Inter-route swap
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for pos_i in range(1, len(route_i)-1):
                    cust_i = route_i[pos_i]
                    prev_i = route_i[pos_i-1]
                    next_i = route_i[pos_i+1]
                    delta_i_rem = distance_matrix[prev_i][next_i] - distance_matrix[prev_i][cust_i] - distance_matrix[cust_i][next_i]
                    for j in range(i+1, truck_count):
                        route_j = routes[j]
                        if len(route_j) <= 2:
                            continue
                        for pos_j in range(1, len(route_j)-1):
                            cust_j = route_j[pos_j]
                            prev_j = route_j[pos_j-1]
                            next_j = route_j[pos_j+1]
                            delta_j_rem = distance_matrix[prev_j][next_j] - distance_matrix[prev_j][cust_j] - distance_matrix[cust_j][next_j]
                            add_i = distance_matrix[prev_i][cust_j] + distance_matrix[cust_j][next_i] - distance_matrix[prev_i][next_i]
                            add_j = distance_matrix[prev_j][cust_i] + distance_matrix[cust_i][next_j] - distance_matrix[prev_j][next_j]
                            new_len_i = route_lengths[i] + delta_i_rem + add_i
                            new_len_j = route_lengths[j] + delta_j_rem + add_j
                            new_max = max(new_len_i, new_len_j, max(route_lengths[k] for k in range(truck_count) if k not in (i, j)))
                            if new_max < best_max:
                                route_i.pop(pos_i)
                                route_j.pop(pos_j)
                                route_i.insert(pos_i, cust_j)
                                route_j.insert(pos_j, cust_i)
                                route_lengths[i] = new_len_i
                                route_lengths[j] = new_len_j
                                best_max = new_max
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                improved_global = True
                continue

            # 2-opt within routes
            for i in range(truck_count):
                route = routes[i]
                if len(route) <= 3:
                    continue
                for a in range(0, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        delta = distance_matrix[route[a]][route[b]] + distance_matrix[route[a+1]][route[b+1]] - distance_matrix[route[a]][route[a+1]] - distance_matrix[route[b]][route[b+1]]
                        new_len = route_lengths[i] + delta
                        if new_len < best_max:
                            new_max = max(new_len, max(route_lengths[k] for k in range(truck_count) if k != i))
                            if new_max < best_max:
                                route[a+1:b+1] = reversed(route[a+1:b+1])
                                route_lengths[i] = new_len
                                best_max = new_max
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return best_max, improved_global

    max_restarts = max(5, n * truck_count // 10)  # more restarts
    for restart in range(max_restarts):
        # Alternate between nearest neighbor and random insertion for diversity
        if restart % 2 == 0:
            routes, route_lengths = nearest_neighbor_initial()
        else:
            routes, route_lengths = random_insertion(restart)
        best_max_local = max(route_lengths)
        best_routes_local = [list(r) for r in routes]
        report_best_vrp(best_routes_local)
        new_max, improved = local_search(routes, route_lengths, best_max_local)
        if improved:
            best_max_local = new_max
            best_routes_local = [list(r) for r in routes]
            report_best_vrp(best_routes_local)
        # Perturbation cycles with adaptive strength
        for cycle in range(3):
            strength = 0.1 + 0.05 * cycle  # adaptive: increase strength
            perturb(routes, route_lengths, strength)
            new_max, improved = local_search(routes, route_lengths, best_max_local)
            if improved:
                best_max_local = new_max
                best_routes_local = [list(r) for r in routes]
                report_best_vrp(best_routes_local)
        if best_max_local < best_overall_max:
            best_overall = [list(r) for r in best_routes_local]
            best_overall_max = best_max_local

    if best_overall is None:
        best_overall = [[0, 0] for _ in range(truck_count)]
    return best_overall