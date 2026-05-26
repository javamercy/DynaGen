import numpy as np
import random

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= len(customers):
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    def clarke_wright_init(shuffle=False):
        routes = [[0, c, 0] for c in customers]
        if shuffle:
            random.shuffle(customers)
            # recompute savings? Actually we use standard savings but shuffled order for merging ties? We'll just shuffle customers before initial routes? Better: Shuffle the list of customer indices to create different initial routes? Actually each customer already has its own route. To create different merging order, we can shuffle the list of routes? Let's instead shuffle the savings list? Simpler: after building initial routes, we'll shuffle the order of merging by randomly ordering the pair list. But we'll implement a simple shuffle: random.shuffle(customers) to change the order of creating single-customer routes? That changes which customer gets which route index but doesn't affect savings merging because savings are based on distances. So we'll keep standard. For alternative construction, we can order savings with random tie-breaking? But we want deterministic by default. We'll use a flag to optionally randomize tie-breaking.
        # Standard CW merging
        savings = []
        for i in range(len(customers)):
            for j in range(i+1, len(customers)):
                ci = customers[i]
                cj = customers[j]
                s = distance_matrix[0][ci] + distance_matrix[0][cj] - distance_matrix[ci][cj]
                if shuffle:
                    s += random.random() * 1e-9  # small random to break ties
                savings.append((s, i, j))
        savings.sort(reverse=True, key=lambda x: (x[0], -x[1], -x[2]) if not shuffle else x[0])
        # Use union-find to merge routes
        parent = list(range(len(customers)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
        # Build routes as lists; we need to track merged routes
        route_of_customer = {c: c for c in customers}  # mapping from customer to its route id (first customer id)
        for s, i, j in savings:
            ci = customers[i]
            cj = customers[j]
            ri = route_of_customer.get(ci)
            rj = route_of_customer.get(cj)
            if ri is None or rj is None or ri == rj:
                continue
            # Check if merging reduces route count
            # We'll store route list separately
            # Actually we need to maintain a list of routes; we'll have a list of routes each as a list
            # But we don't have routes yet; we start with each customer in own route.
            # We'll do as before: create list of routes and merge.
            # This is getting complicated in this lambda function. Let's instead use the same merging as parent code but with optional shuffle.
            # Simpler: we'll generate a list of routes = [[0,c,0] for c in customers] and then merge based on savings order, but with shuffle we randomize the order of considering savings pairs? Actually we can just sort savings by s descending, and for equal s, we can tie-break by i and j indexes. To add randomness, we'll add a small random to each saving when shuffle=True.
            # We'll implement it in the main loop.
            pass
        # For simplicity, we use the same merging as parent but we'll just return a placeholder; we'll write it inline.
        # Actually let's just use the same Clarke-Wright merging from parent code (the one in the candidate). We'll call that.
        # We'll duplicate logic here.
        # Start with each customer as a route.
        routes = [[0, c, 0] for c in customers]
        # Compute savings for all pairs
        saving_pairs = []
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                ri = routes[i]
                rj = routes[j]
                if len(ri) <= 2 or len(rj) <= 2:
                    continue
                last_i = ri[-2]
                first_i = ri[1]
                last_j = rj[-2]
                first_j = rj[1]
                s1 = distance_matrix[0][last_i] + distance_matrix[0][first_j] - distance_matrix[last_i][first_j]
                s2 = distance_matrix[0][last_j] + distance_matrix[0][first_i] - distance_matrix[last_j][first_i]
                max_s = max(s1, s2)
                if shuffle:
                    max_s += random.random() * 1e-9
                saving_pairs.append((max_s, i, j, 0 if s1 >= s2 else 1, s1, s2))
        saving_pairs.sort(reverse=True, key=lambda x: (x[0], -x[1], -x[2]))
        # Merge until we have truck_count routes
        merged = [False]*len(routes)
        route_list = routes
        # We'll maintain a list of active route indices
        active_indices = list(range(len(route_list)))
        # We'll need to efficiently find which index a customer belongs to; we can rebuild after each merge.
        # Simpler: implement iterative merging as in parent: while len(routes) > truck_count, find best saving pair, merge, and update.
        # But we need to incorporate shuffle into saving computation. We'll do it outside this function.
        # We'll just call the same merging routine but with optional shuffle of savings weights.
        # For now, we'll do standard merging (no shuffle) and later we may restart with shuffled savings.
        return None  # Will be replaced

    # We'll implement the main solver with restarts and adaptive perturbation
    best_routes = None
    best_max = float('inf')

    # Number of restarts: at most n (instance size)
    max_restarts = n
    ejection_fraction = 0.1
    max_ejection_fraction = 0.3
    plateau_count = 0

    for restart in range(max_restarts):
        # Construction: if restart==0, normal CW; else, small random perturbation to savings (shuffle) or use random order
        # We'll use a variant: for first restart, standard; for later, randomly shuffle the order of customers when building initial routes? But that doesn't change savings. Instead, we can inject randomness into the merging decision: when choosing which pair to merge, we add a small random noise to savings values (like parent cand_000020 used max-distance-aware merging; we could also consider distance of longest route? But we want simplicity). We'll just use standard CW with deterministic tie-breaking. If stuck, we'll rely on perturbation to escape.
        # Actually we can do: if plateau_count > 2, reinitialize with a shuffled sequence (randomly permute customer order before creating single-customer routes? That doesn't change savings. Better: create routes by sequentially inserting customers into existing routes using a farthest-insertion or cheapest insertion? But we want to keep CW.
        # Since we already have perturbation mechanism, we can keep same initialization but increase ejection fraction.
        
        # Build initial routes via Clarke-Wright (same as parent code)
        routes = [[0, c, 0] for c in customers]
        while len(routes) > truck_count:
            best_saving = -1e9
            best_pair = None
            best_order = 0
            for i in range(len(routes)):
                for j in range(i+1, len(routes)):
                    ri = routes[i]
                    rj = routes[j]
                    if len(ri) <= 2 or len(rj) <= 2:
                        continue
                    last_i = ri[-2]
                    first_i = ri[1]
                    last_j = rj[-2]
                    first_j = rj[1]
                    s1 = distance_matrix[0][last_i] + distance_matrix[0][first_j] - distance_matrix[last_i][first_j]
                    s2 = distance_matrix[0][last_j] + distance_matrix[0][first_i] - distance_matrix[last_j][first_i]
                    # We could add noise for diversification on later restarts
                    noise = 0.0
                    if restart > 0:
                        noise = random.uniform(0, 1e-9 * (1 + plateau_count))  # small noise
                    if s1 + noise > best_saving:
                        best_saving = s1 + noise
                        best_pair = (i, j)
                        best_order = 0
                    if s2 + noise > best_saving:
                        best_saving = s2 + noise
                        best_pair = (i, j)
                        best_order = 1
            if best_pair is None:
                break
            i, j = best_pair
            if best_order == 0:
                new_route = routes[i][:-1] + routes[j][1:]
            else:
                new_route = routes[j][:-1] + routes[i][1:]
            if i < j:
                del routes[j]
                del routes[i]
            else:
                del routes[i]
                del routes[j]
            routes.append(new_route)

        # Now we have truck_count routes
        # Local search and perturbation
        def local_search(routes):
            improved = True
            max_iter = n * truck_count
            for _ in range(max_iter):
                if not improved:
                    break
                improved = False
                dists = [route_distance(r, distance_matrix) for r in routes]
                max_dist = max(dists)
                max_idx = dists.index(max_dist)
                # Intra-route 2-opt on longest route
                if len(routes[max_idx]) > 3:
                    r = routes[max_idx]
                    best_imp = 0
                    best_new_route = None
                    for i in range(1, len(r)-2):
                        for j in range(i+1, len(r)-1):
                            if j - i == 1:
                                continue
                            new_route = r[:i] + r[i:j+1][::-1] + r[j+1:]
                            new_dist = route_distance(new_route, distance_matrix)
                            old_dist = route_distance(r, distance_matrix)
                            if new_dist < old_dist - 1e-9:
                                improvement = old_dist - new_dist
                                if improvement > best_imp:
                                    best_imp = improvement
                                    best_new_route = new_route
                    if best_new_route is not None:
                        routes[max_idx] = best_new_route
                        improved = True
                if improved:
                    continue
                # Inter-route relocate from longest route
                if len(routes[max_idx]) > 2:
                    r_max = routes[max_idx]
                    for pos in range(1, len(r_max)-1):
                        cust = r_max[pos]
                        new_max_route = r_max[:pos] + r_max[pos+1:]
                        new_max_dist = route_distance(new_max_route, distance_matrix)
                        for other_idx in range(truck_count):
                            if other_idx == max_idx:
                                continue
                            other_route = routes[other_idx]
                            for insert_pos in range(1, len(other_route)):
                                new_other_route = other_route[:insert_pos] + [cust] + other_route[insert_pos:]
                                new_other_dist = route_distance(new_other_route, distance_matrix)
                                new_dists = dists.copy()
                                new_dists[max_idx] = new_max_dist
                                new_dists[other_idx] = new_other_dist
                                new_max = max(new_dists)
                                if new_max < max_dist - 1e-9:
                                    routes[max_idx] = new_max_route
                                    routes[other_idx] = new_other_route
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                if improved:
                    continue
                # Inter-route swap
                if len(routes[max_idx]) > 2:
                    r_max = routes[max_idx]
                    for other_idx in range(truck_count):
                        if other_idx == max_idx or len(routes[other_idx]) <= 2:
                            continue
                        other_route = routes[other_idx]
                        for pos_max in range(1, len(r_max)-1):
                            cust_a = r_max[pos_max]
                            for pos_other in range(1, len(other_route)-1):
                                cust_b = other_route[pos_other]
                                new_max_route = r_max.copy()
                                new_max_route[pos_max] = cust_b
                                new_max_dist = route_distance(new_max_route, distance_matrix)
                                new_other_route = other_route.copy()
                                new_other_route[pos_other] = cust_a
                                new_other_dist = route_distance(new_other_route, distance_matrix)
                                new_dists = dists.copy()
                                new_dists[max_idx] = new_max_dist
                                new_dists[other_idx] = new_other_dist
                                new_max = max(new_dists)
                                if new_max < max_dist - 1e-9:
                                    routes[max_idx] = new_max_route
                                    routes[other_idx] = new_other_route
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                if improved:
                    continue
                # Inter-route 2-opt* (cross-exchange) on longest route
                # For each pair of routes (max_idx, other_idx), try to exchange endpoints
                for other_idx in range(truck_count):
                    if other_idx == max_idx or len(routes[other_idx]) <= 2:
                        continue
                    r1 = routes[max_idx]
                    r2 = routes[other_idx]
                    # Try all pairs of edges (i,i+1) in r1 and (j,j+1) in r2
                    for i in range(1, len(r1)-2):
                        for j in range(1, len(r2)-2):
                            # Option 1: (r1[i], r1[i+1]) swapped with (r2[j], r2[j+1]) such that new routes are:
                            # r1[:i+1] + r2[j+1:] reversed? Actually standard 2-opt*: exchange end parts
                            # We'll implement two possibilities:
                            # Case A: keep r1[0..i] and r2[0..j], then append the other's tail
                            # new1 = r1[:i+1] + r2[j+1:]
                            # new2 = r2[:j+1] + r1[i+1:]
                            # Ensure new routes start/end with depot if needed? r1[0]=0, r1[-1]=0, so if we cut at i (not at depot), new routes will start with 0 and end with 0 if we include full tails? Actually r1[:i+1] ends with r1[i], then we add r2[j+1:] which starts with r2[j+1]; the new route will end with r2[-1]=0, so it's fine except maybe missing depot at start? r1[0]=0 so it's fine. Similarly r2[:j+1] ends with r2[j], then r1[i+1:] starts with r1[i+1] and ends with 0. So valid.
                            # Evaluate
                            new_r1 = r1[:i+1] + r2[j+1:]
                            new_r2 = r2[:j+1] + r1[i+1:]
                            new_d1 = route_distance(new_r1, distance_matrix)
                            new_d2 = route_distance(new_r2, distance_matrix)
                            old_d1 = route_distance(r1, distance_matrix)
                            old_d2 = route_distance(r2, distance_matrix)
                            new_max = max(new_d1, new_d2)
                            old_max = max(old_d1, old_d2)
                            if new_max < old_max - 1e-9:
                                routes[max_idx] = new_r1
                                routes[other_idx] = new_r2
                                improved = True
                                break
                            # Case B: exchange opposite: r1[:i+1] + reversed(r2[j+1:])? Actually another common variant: connect r1[i] to r2[j+1] and r2[j] to r1[i+1] giving new routes:
                            new_r1b = r1[:i+1] + list(reversed(r2[j+1:]))
                            new_r2b = r2[:j+1] + list(reversed(r1[i+1:]))
                            new_d1b = route_distance(new_r1b, distance_matrix)
                            new_d2b = route_distance(new_r2b, distance_matrix)
                            new_maxb = max(new_d1b, new_d2b)
                            if new_maxb < old_max - 1e-9:
                                routes[max_idx] = new_r1b
                                routes[other_idx] = new_r2b
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                # (End of 2-opt*)
            return routes

        # Apply local search
        routes = local_search(routes)
        dists = [route_distance(r, distance_matrix) for r in routes]
        current_max = max(dists)
        if current_max < best_max - 1e-9:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
            plateau_count = 0
            ejection_fraction = 0.1  # reset
        else:
            plateau_count += 1
            # Increase ejection fraction if stuck
            if plateau_count > 3:
                ejection_fraction = min(ejection_fraction + 0.05, max_ejection_fraction)
            # Perturb: distance-aware ejection chain on longest route
            max_idx = dists.index(current_max)
            r_max = routes[max_idx]
            if len(r_max) > 3:
                # Compute contribution of each customer (distance saved if removed)
                contributions = []
                for k in range(1, len(r_max)-1):
                    prev = r_max[k-1]
                    curr = r_max[k]
                    nxt = r_max[k+1]
                    contrib = distance_matrix[prev][curr] + distance_matrix[curr][nxt] - distance_matrix[prev][nxt]
                    contributions.append((contrib, k, curr))
                contributions.sort(reverse=True)
                num_eject = max(1, int((len(r_max)-2) * ejection_fraction))
                ejected = [c[2] for c in contributions[:num_eject]]
                new_route = [x for x in r_max if x not in ejected]
                # Reinsert ejected customers into other routes at best positions
                for cust in ejected:
                    best_increase = float('inf')
                    best_route_idx = -1
                    best_pos = -1
                    for other_idx in range(truck_count):
                        if other_idx == max_idx:
                            continue
                        other_route = routes[other_idx]
                        for pos in range(1, len(other_route)):
                            new_other = other_route[:pos] + [cust] + other_route[pos:]
                            new_dist = route_distance(new_other, distance_matrix)
                            old_dist = route_distance(other_route, distance_matrix)
                            increase = new_dist - old_dist
                            if increase < best_increase:
                                best_increase = increase
                                best_route_idx = other_idx
                                best_pos = pos
                    # Perform insertion
                    routes[best_route_idx] = routes[best_route_idx][:best_pos] + [cust] + routes[best_route_idx][best_pos:]
                routes[max_idx] = new_route
            # Continue to next restart (the loop continues, but we don't reset local search; we just let the next iteration apply local search again)
        # End of for restart
    # Ensure best_routes is set
    if best_routes is None:
        best_routes = routes
    report_best_vrp(best_routes)
    return best_routes