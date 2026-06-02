import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    def route_distance(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    # Regret-2 construction
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    random.shuffle(customers)
    while customers:
        best_regret = -float('inf')
        best_cust = None
        best_insert = None  # (route_idx, pos)
        for cust in customers:
            # compute best and second best increase in max distance
            increases = []
            current_max = max_distance(routes)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_route_dist = route_distance(route) + added
                    other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                    new_max = max(new_route_dist, other_max)
                    increase = new_max - current_max
                    increases.append((increase, r_idx, pos))
            if not increases:
                continue
            increases.sort(key=lambda x: (x[0], x[1], x[2]))
            best_inc = increases[0]
            second_best = increases[1] if len(increases) > 1 else (float('inf'), -1, -1)
            regret = second_best[0] - best_inc[0]
            if regret > best_regret:
                best_regret = regret
                best_cust = cust
                best_insert = (best_inc[1], best_inc[2])
            elif regret == best_regret:
                if best_inc[0] < increases[0][0]:
                    best_regret = regret
                    best_cust = cust
                    best_insert = (best_inc[1], best_inc[2])
        if best_cust is None:
            break
        # insert best customer
        r_idx, pos = best_insert
        routes[r_idx].insert(pos, best_cust)
        customers.remove(best_cust)
    
    best_routes = [r[:] for r in routes]
    best_max = max_distance(routes)
    report_best_vrp(best_routes)
    
    # Local search cycles
    max_cycles = 3
    for cycle in range(max_cycles):
        max_iter = (n-1) * truck_count * 5
        no_improve_count = 0
        for iteration in range(max_iter):
            improved = False
            max_dist = max_distance(routes)
            longest_routes = [i for i, r in enumerate(routes) if route_distance(r) == max_dist]
            phases = ['2opt', 'relocate', 'swap', 'cross']
            random.shuffle(phases)
            for phase in phases:
                if improved:
                    break
                if phase == '2opt':
                    for r_idx in longest_routes:
                        route = routes[r_idx]
                        if len(route) <= 3:
                            continue
                        for i in range(1, len(route)-2):
                            for j in range(i+1, len(route)-1):
                                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                old_dist = route_distance(route)
                                new_dist = route_distance(new_route)
                                if new_dist >= old_dist:
                                    continue
                                other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != r_idx)
                                new_max = max(new_dist, other_max)
                                if new_max < best_max - 1e-12:
                                    routes[r_idx] = new_route
                                    best_routes = [r[:] for r in routes]
                                    best_max = new_max
                                    improved = True
                                    report_best_vrp(best_routes)
                                    break
                        if improved:
                            break
                elif phase == 'relocate':
                    for src in longest_routes:
                        route_src = routes[src]
                        if len(route_src) <= 2:
                            continue
                        for pos_src in range(1, len(route_src)-1):
                            cust = route_src[pos_src]
                            temp_src = route_src[:pos_src] + route_src[pos_src+1:]
                            dist_src = route_distance(temp_src)
                            for dst in range(truck_count):
                                if dst == src:
                                    continue
                                route_dst = routes[dst]
                                for pos_dst in range(1, len(route_dst)):
                                    new_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                                    dist_dst = route_distance(new_dst)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != src and x != dst)
                                    new_max = max(dist_src, dist_dst, other_max)
                                    if new_max < best_max - 1e-12:
                                        routes[src] = temp_src
                                        routes[dst] = new_dst
                                        best_routes = [r[:] for r in routes]
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                elif phase == 'swap':
                    for t1 in longest_routes:
                        route1 = routes[t1]
                        if len(route1) <= 2:
                            continue
                        for t2 in range(truck_count):
                            if t2 == t1:
                                continue
                            route2 = routes[t2]
                            if len(route2) <= 2:
                                continue
                            for i in range(1, len(route1)-1):
                                for j in range(1, len(route2)-1):
                                    cust1 = route1[i]
                                    cust2 = route2[j]
                                    new_route1 = route1[:i] + [cust2] + route1[i+1:]
                                    new_route2 = route2[:j] + [cust1] + route2[j+1:]
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < best_max - 1e-12:
                                        routes[t1] = new_route1
                                        routes[t2] = new_route2
                                        best_routes = [r[:] for r in routes]
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                elif phase == 'cross':
                    for t1 in longest_routes:
                        route1 = routes[t1]
                        if len(route1) <= 2:
                            continue
                        for t2 in range(truck_count):
                            if t2 == t1:
                                continue
                            route2 = routes[t2]
                            if len(route2) <= 2:
                                continue
                            for i in range(1, len(route1)-1):
                                for j in range(1, len(route2)-1):
                                    new_route1 = route1[:i] + route2[j:]
                                    new_route2 = route2[:j] + route1[i:]
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < best_max - 1e-12:
                                        routes[t1] = new_route1
                                        routes[t2] = new_route2
                                        best_routes = [r[:] for r in routes]
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                        if improved:
                            break
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= 3:
                    break
        # Shake: first longest, then second longest if needed
        # Compute route distances
        rdists = [route_distance(r) for r in routes]
        sorted_inds = sorted(range(truck_count), key=lambda i: rdists[i], reverse=True)
        for target_idx in sorted_inds[:2]:  # first two longest
            if len(routes[target_idx]) <= 3:
                continue
            route = routes[target_idx]
            # Compute savings for internal customers
            savings = []
            for i in range(1, len(route)-1):
                cust = route[i]
                prev = route[i-1]
                nxt = route[i+1]
                saving = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                savings.append((saving, i, cust))
            savings.sort(reverse=True, key=lambda x: x[0])
            num_remove = max(1, len(route)//3)
            remove_indices = [savings[k][1] for k in range(min(num_remove, len(savings)))]
            removed_customers = [route[i] for i in sorted(remove_indices, reverse=True)]
            for idx in sorted(remove_indices, reverse=True):
                route.pop(idx)
            # Reinsert greedily minimizing max distance increase
            for cust in removed_customers:
                best_increase = float('inf')
                best_route = -1
                best_pos = -1
                current_max = max_distance(routes)
                for r_idx in range(truck_count):
                    r = routes[r_idx]
                    for pos in range(1, len(r)):
                        prev = r[pos-1]
                        nxt = r[pos]
                        added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_route_dist = route_distance(r) + added
                        other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                        new_max = max(new_route_dist, other_max)
                        increase = new_max - current_max
                        if increase < best_increase - 1e-12 or (abs(increase - best_increase) < 1e-12 and r_idx < best_route):
                            best_increase = increase
                            best_route = r_idx
                            best_pos = pos
                routes[best_route].insert(best_pos, cust)
            cur_max = max_distance(routes)
            if cur_max < best_max - 1e-12:
                best_routes = [r[:] for r in routes]
                best_max = cur_max
                report_best_vrp(best_routes)
            else:
                # If no improvement, add block move perturbation
                longest_idx = sorted_inds[0]
                if len(routes[longest_idx]) > 3:
                    route = routes[longest_idx]
                    # pick a random block of length 1-3 from the longest route
                    block_len = random.randint(1, min(3, len(route)-2))
                    start = random.randint(1, len(route)-block_len-1)
                    block = route[start:start+block_len]
                    # remove block
                    for _ in range(block_len):
                        route.pop(start)
                    # find shortest route
                    short_idx = min(range(truck_count), key=lambda i: route_distance(routes[i]))
                    # insert block at best position in shortest route
                    for cust in block:
                        best_increase = float('inf')
                        best_pos = -1
                        current_max = max_distance(routes)
                        r = routes[short_idx]
                        for pos in range(1, len(r)):
                            prev = r[pos-1]
                            nxt = r[pos]
                            added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                            new_route_dist = route_distance(r) + added
                            other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != short_idx)
                            new_max = max(new_route_dist, other_max)
                            increase = new_max - current_max
                            if increase < best_increase - 1e-12:
                                best_increase = increase
                                best_pos = pos
                        r.insert(best_pos, cust)
                    cur_max = max_distance(routes)
                    if cur_max < best_max - 1e-12:
                        best_routes = [r[:] for r in routes]
                        best_max = cur_max
                        report_best_vrp(best_routes)
        if best_max < global_best_max - 1e-12:
            global_best_max = best_max
            global_best_routes = [r[:] for r in best_routes]
            report_best_vrp(global_best_routes)
    
    if global_best_routes is None:
        global_best_routes = [r[:] for r in routes]
    return global_best_routes