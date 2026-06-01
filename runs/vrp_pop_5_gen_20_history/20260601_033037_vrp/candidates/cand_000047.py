import numpy as np
import heapq

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
    
    # ---------- Clarke-Wright savings construction ----------
    # each customer starts as a route
    route_list = [[0, c, 0] for c in customers]
    # compute savings and heap
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                savings.append((-s, i, j))  # negative for min-heap to get largest saving first
    heapq.heapify(savings)
    # maintain per route: first customer, last customer, and mapping from customer to route index
    # We'll use a simple structure: for each route, store first and last customer (excluding depot)
    first_cust = {c: c for c in customers}  # for route represented by its first customer
    last_cust = {c: c for c in customers}
    # mapping from customer to representative (first customer of its route)
    rep = {c: c for c in customers}
    def find(c):
        while rep[c] != c:
            rep[c] = rep[rep[c]]
            c = rep[c]
        return c
    def union(c1, c2):
        r1, r2 = find(c1), find(c2)
        if r1 == r2:
            return False
        # merge r2 into r1
        rep[r2] = r1
        return True
    # dictionary mapping representative to route
    routes_by_rep = {c: [0, c, 0] for c in customers}
    
    while len(set(find(c) for c in customers)) > truck_count and savings:
        neg_s, i, j = heapq.heappop(savings)
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        # check if i and j are endpoints of their routes
        # i must be first or last of its route, j must be first or last of its route
        # We'll only allow merging when last of one route connects to first of the other
        first_i, last_i = first_cust[ri], last_cust[ri]
        first_j, last_j = first_cust[rj], last_cust[rj]
        merge_possible = False
        new_first = None
        new_last = None
        if i == last_i and j == first_j:
            # route_i + route_j
            merge_possible = True
            new_first = first_i
            new_last = last_j
            route_i = routes_by_rep[ri]
            route_j = routes_by_rep[rj]
            new_route = route_i[:-1] + route_j[1:]
        elif i == first_i and j == last_j:
            # route_j + route_i
            merge_possible = True
            new_first = first_j
            new_last = last_i
            route_i = routes_by_rep[ri]
            route_j = routes_by_rep[rj]
            new_route = route_j[:-1] + route_i[1:]
        elif j == last_j and i == first_i:
            # route_j + route_i
            merge_possible = True
            new_first = first_j
            new_last = last_i
            route_i = routes_by_rep[ri]
            route_j = routes_by_rep[rj]
            new_route = route_j[:-1] + route_i[1:]
        elif j == first_j and i == last_i:
            # route_i + route_j
            merge_possible = True
            new_first = first_i
            new_last = last_j
            route_i = routes_by_rep[ri]
            route_j = routes_by_rep[rj]
            new_route = route_i[:-1] + route_j[1:]
        if not merge_possible:
            continue
        # perform merge
        # update representative
        if union(i, j):  # merges rj into ri
            new_rep = find(i)
        else:
            new_rep = find(i)  # already same? but we checked
        routes_by_rep[new_rep] = new_route
        first_cust[new_rep] = new_first
        last_cust[new_rep] = new_last
    
    # collect final routes after merging
    reps_seen = set()
    route_list = []
    for c in customers:
        r = find(c)
        if r not in reps_seen:
            reps_seen.add(r)
            route_list.append(routes_by_rep[r])
    # if still more routes than truck_count, merge smallest (by total distance) repeatedly
    while len(route_list) > truck_count:
        # find two routes with smallest total distance
        dists = [(route_distance(r), idx) for idx, r in enumerate(route_list)]
        dists.sort(key=lambda x: (x[0], x[1]))
        idx1 = dists[0][1]
        idx2 = dists[1][1]
        # merge: simple concatenation (order may not be optimal but improvement later)
        r1 = route_list[idx1]
        r2 = route_list[idx2]
        new_route = r1[:-1] + r2[1:]
        # build new list
        new_route_list = [r for i, r in enumerate(route_list) if i not in (idx1, idx2)]
        new_route_list.append(new_route)
        route_list = new_route_list
    
    report_best_vrp(route_list)
    
    # ---------- Improvement phase ----------
    max_iter = len(customers) * truck_count * 2
    for _ in range(max_iter):
        improved = False
        # find route with maximum distance (tie-break larger index? no, any)
        dists = [route_distance(r) for r in route_list]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        # 1. Try relocation from max route
        max_route = route_list[max_idx]
        interior = max_route[1:-1]
        if not interior:
            break
        # sort interior by customer index for deterministic order
        for cust in sorted(interior):
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = route_list[other_idx]
                # find best insertion position (minimizing delta)
                best_pos = None
                best_delta = float('inf')
                for pos in range(1, len(other_route)):
                    prev = other_route[pos-1]
                    nxt = other_route[pos]
                    delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_pos = pos
                # try move
                new_routes = [list(r) for r in route_list]
                new_routes[max_idx].remove(cust)
                new_routes[other_idx].insert(best_pos, cust)
                new_max = max(route_distance(r) for r in new_routes)
                if new_max < best_max - 1e-12:
                    route_list = new_routes
                    report_best_vrp(route_list)
                    improved = True
                    break
                elif new_max < best_max + 1e-12:
                    # tie: prefer smaller customer index, then smaller other_idx
                    # current best_max is the global best, but we want to update internal best?
                    # We'll still update if tie and this move is better in tie-breaking order
                    # But we need to compare with any previous tie move. For simplicity, only accept if strictly better.
                    pass
            if improved:
                break
        if improved:
            continue
        
        # 2. Try swapping between max route and another
        for other_idx in range(truck_count):
            if other_idx == max_idx:
                continue
            other_route = route_list[other_idx]
            interior_other = other_route[1:-1]
            if not interior_other:
                continue
            for cust_max in sorted(interior):
                for cust_other in sorted(interior_other):
                    new_routes = [list(r) for r in route_list]
                    # find indices
                    idx_max = new_routes[max_idx].index(cust_max)
                    idx_other = new_routes[other_idx].index(cust_other)
                    new_routes[max_idx][idx_max] = cust_other
                    new_routes[other_idx][idx_other] = cust_max
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_max - 1e-12:
                        route_list = new_routes
                        report_best_vrp(route_list)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        
        # 3. 2-opt on each route
        for idx in range(truck_count):
            route = route_list[idx]
            if len(route) <= 3:
                continue
            best_route = route[:]
            best_dist = route_distance(route)
            improved_2opt = False
            for a in range(1, len(route)-2):
                for b in range(a+1, len(route)-1):
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist - 1e-12:
                        best_dist = new_dist
                        best_route = new_route
                        improved_2opt = True
                        break
                if improved_2opt:
                    break
            if improved_2opt:
                route_list[idx] = best_route
                new_max = max(route_distance(r) for r in route_list)
                if new_max < best_max - 1e-12:
                    report_best_vrp(route_list)
                improved = True
                break
        if not improved:
            break
    
    final_routes = best_routes if best_routes is not None else route_list
    while len(final_routes) < truck_count:
        final_routes.append([0, 0])
    return final_routes