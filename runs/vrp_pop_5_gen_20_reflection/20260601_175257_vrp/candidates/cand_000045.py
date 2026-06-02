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

    def clarke_wright_construction(seed):
        random.seed(seed)
        # Initialize each customer as a separate route (except depot)
        routes = [[0, c, 0] for c in customers]
        route_lengths = [distance_matrix[0][c] + distance_matrix[c][0] for c in customers]
        # Compute savings
        savings = []
        for i in customers:
            for j in customers:
                if i < j:
                    s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
                    savings.append((s, i, j))
        savings.sort(reverse=True, key=lambda x: x[0])
        # Merging process: use a union-find to track routes
        parent = {c: c for c in customers}
        # Actually, we need to track which route each customer belongs to (by index)
        route_of = {c: idx for idx, c in enumerate(customers)}
        # For each saving, try to merge
        for s, i, j in savings:
            if route_of[i] == route_of[j]:
                continue
            # Check if i is at the start (right after depot) or end (right before depot) of its route
            ri = routes[route_of[i]]
            rj = routes[route_of[j]]
            # Determine orientation: we need to connect end of one to start of another
            # For route ri, possible endpoints are after depot: ri[1] or before depot: ri[-2]
            # Use a simple rule: connect if i is last before depot in ri and j is first after depot in rj, or vice versa
            # Actually, for Clarke-Wright, we want to merge the two routes by connecting i's route end to j's route start
            # But we simplified: we store full routes. Let's just do a simple merge: combine routes
            # However, for correctness, we'll implement proper merge criteria
            # Check if i is at one end of its route and j at one end of its route
            def is_endpoint(route, node):
                return route[1] == node or route[-2] == node
            if is_endpoint(ri, i) and is_endpoint(rj, j):
                # Determine order: we want to avoid depot
                # If i is at end of ri and j at start of rj: ri + rj (removing duplicate depot)
                # Determine orientation
                if ri[1] == i and rj[1] == j:
                    # i is at start, j is at start -> reverse one
                    new_route = ri[:-1] + rj[1:]  # ri without last depot, then rj from depot to end? Actually careful: ri ends with 0, rj starts with 0
                    # Better: ri = [0, ..., i, 0], rj = [0, j, ..., 0]
                    # To connect end of ri to start of rj, we need i at end of ri and j at start of rj
                    # If i at end of ri (ri[-2]==i) and j at start of rj (rj[1]==j): new = ri[:-1] + rj[1:]
                    new_route = ri[:-1] + rj[1:]
                elif ri[-2] == i and rj[1] == j:
                    new_route = ri[:-1] + rj[1:]
                elif ri[1] == i and rj[-2] == j:
                    new_route = rj[:-1] + ri[1:]
                elif ri[-2] == i and rj[-2] == j:
                    new_route = ri[:-1] + list(reversed(rj[1:-1])) + [0]  # need to reverse one
                    # Actually careful: if both at end, we can connect by reversing one route
                    # But let's keep it simple: if both at end, reverse rj and attach
                    new_route = ri[:-1] + list(reversed(rj[1:-1])) + [0]
                else:
                    continue  # shouldn't happen given is_endpoint
                # Update route lengths
                new_len = route_lengths[route_of[i]] + route_lengths[route_of[j]] - distance_matrix[i][j]
                # Merge routes: assign all customers to one route index
                merge_idx = route_of[i]
                remove_idx = route_of[j]
                # Update route and length
                routes[merge_idx] = new_route
                route_lengths[merge_idx] = new_len
                # Update route_of for all customers in removed route
                for c in rj:
                    if c != 0:
                        route_of[c] = merge_idx
                # Invalidate removed route (set to empty but keep index)
                routes[remove_idx] = [0,0]
                route_lengths[remove_idx] = 0.0
        # After merging, collect non-empty routes
        final_routes = [r for r in routes if len(r) > 2]
        # If fewer than truck_count, add empty routes; if more, keep only truck_count best? Actually we need exactly truck_count
        # If we have more, we need to select truck_count routes with smallest max? But typically we produce truck_count routes by construction
        # Clarke-Wright tends to produce many routes. We'll cap by selecting the truck_count routes with highest total distance? Not ideal.
        # To simplify, we'll just use the first truck_count non-empty routes and merge remaining customers into them using best insertion.
        # Or we can use a different construction: nearest neighbor insertion with random seed.
        # Let's use nearest neighbor insertion instead, which inherently creates exactly truck_count routes.
        # Given complexity, we'll switch to nearest neighbor insertion.
        pass

    # We'll use nearest neighbor insertion with random order
    def nearest_neighbor_insertion(seed):
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

    def perturb(routes, route_lengths, strength=0.3):
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
                if len(route) == 2:
                    # empty route, treat specially: insert between depot
                    add = 2 * distance_matrix[0][cust]
                    new_len = add
                    new_max = max(route_lengths[:ri] + [new_len] + route_lengths[ri+1:])
                    if new_max < best_max or (new_max == best_max and (best_len is None or new_len < best_len)):
                        best_max = new_max
                        best_route = ri
                        best_pos = 1
                        best_len = new_len
                else:
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
            if len(route) == 2:
                route.insert(1, cust)
                route_lengths[best_route] = best_len
            else:
                prev = route[best_pos-1]
                nxt = route[best_pos]
                route.insert(best_pos, cust)
                route_lengths[best_route] += distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]

    def local_search(routes, route_lengths, best_max):
        max_passes = min(100, (n - 1) * truck_count)
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
                        if len(route_j) == 2:
                            # empty, insertion between depot
                            add = 2 * distance_matrix[0][cust]
                            new_len_j = add
                            new_max = max(new_len_i, new_len_j, max(route_lengths[k] for k in range(truck_count) if k not in (i, j)))
                            if new_max < best_max:
                                route_i.pop(pos_i)
                                route_j.insert(1, cust)
                                route_lengths[i] = new_len_i
                                route_lengths[j] = new_len_j
                                best_max = new_max
                                improved = True
                                break
                        else:
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

            # Intra-route 2-opt
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
            if improved:
                improved_global = True
                continue

            # Intra-route Or-opt (relocate 1,2,3 consecutive customers)
            for i in range(truck_count):
                route = routes[i]
                if len(route) <= 3:
                    continue
                for seg_len in [1, 2, 3]:
                    for start in range(1, len(route)-seg_len-1):
                        end = start + seg_len - 1
                        # extract segment
                        seg = route[start:end+1]
                        # remove segment and compute new route
                        new_route = route[:start] + route[end+1:]
                        # length after removal
                        removal_len = route_lengths[i]
                        # recalc removal delta (we'll compute full length after reinsertion)
                        # compute current route length minus segment's contribution (subtract cost of entering/exiting segment)
                        # For simplicity, we'll compute new route length by recalc
                        # We'll compute length of new_route
                        len_new = sum(distance_matrix[new_route[k]][new_route[k+1]] for k in range(len(new_route)-1))
                        # try inserting segment in all possible positions
                        for ins in range(1, len(new_route)):
                            # insert segment into new_route at position ins
                            candidate_route = new_route[:ins] + seg + new_route[ins:]
                            # compute length
                            cand_len = sum(distance_matrix[candidate_route[k]][candidate_route[k+1]] for k in range(len(candidate_route)-1))
                            new_max = max(cand_len, max(route_lengths[k] for k in range(truck_count) if k != i))
                            if new_max < best_max:
                                routes[i] = candidate_route
                                route_lengths[i] = cand_len
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

            if not improved:
                break
        return best_max, improved_global

    max_restarts = min(5, (n - 1) * truck_count)
    for restart in range(max_restarts):
        routes, route_lengths = nearest_neighbor_insertion(restart)
        best_max_local = max(route_lengths)
        best_routes_local = [list(r) for r in routes]
        report_best_vrp(best_routes_local)
        new_max, improved = local_search(routes, route_lengths, best_max_local)
        if improved:
            best_max_local = new_max
            best_routes_local = [list(r) for r in routes]
            report_best_vrp(best_routes_local)
        for _ in range(3):
            perturb(routes, route_lengths, 0.3)
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