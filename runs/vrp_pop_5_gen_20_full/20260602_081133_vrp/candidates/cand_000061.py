import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customer_count = n - 1
    if truck_count >= customer_count:
        routes = [[0, 0] for _ in range(truck_count)]
        for i in range(1, n):
            routes[i-1] = [0, i, 0]
        try:
            report_best_vrp(routes)
        except:
            pass
        return routes

    def compute_distance(route):
        if len(route) <= 2:
            return 0.0
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total

    def regret_insertion(routes, unassigned, k=2):
        while unassigned:
            best_cust = None
            best_route_idx = None
            best_pos = None
            best_regret = -float('inf')
            best_max = float('inf')
            for cust in unassigned:
                insertion_costs = []
                for t in range(truck_count):
                    route = routes[t]
                    if len(route) == 2:
                        pos = 1
                        increase = 2 * distance_matrix[0][cust]
                    else:
                        best_pos_t = 1
                        best_inc = float('inf')
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            inc = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                            if inc < best_inc - 1e-9:
                                best_inc = inc
                                best_pos_t = pos
                        pos = best_pos_t
                        increase = best_inc
                    # compute resulting max distance after insertion
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = compute_distance(new_route)
                    other_dists = [compute_distance(routes[i]) for i in range(truck_count) if i != t]
                    new_max = max(new_dist, *other_dists)
                    insertion_costs.append((t, pos, increase, new_max))
                # sort by increase (cost to insert)
                insertion_costs.sort(key=lambda x: x[2])
                # regret: difference between second best and best cost
                if len(insertion_costs) >= 2:
                    regret = insertion_costs[1][2] - insertion_costs[0][2]
                else:
                    regret = insertion_costs[0][2]  # only one truck?
                # choose best max among those with highest regret
                if regret > best_regret - 1e-9:
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = insertion_costs[0][0]
                    best_pos = insertion_costs[0][1]
                    best_max = insertion_costs[0][3]
                elif abs(regret - best_regret) < 1e-9:
                    if insertion_costs[0][3] < best_max - 1e-9:
                        best_max = insertion_costs[0][3]
                        best_cust = cust
                        best_route_idx = insertion_costs[0][0]
                        best_pos = insertion_costs[0][1]
            # insert best_cust
            route = routes[best_route_idx]
            routes[best_route_idx] = route[:best_pos] + [best_cust] + route[best_pos:]
            unassigned.remove(best_cust)
        return routes

    def local_search(routes, max_iter=100):
        for _ in range(max_iter):
            improved = False
            distances = [compute_distance(r) for r in routes]
            max_dist = max(distances)
            # intra-route 2-opt
            for t in range(truck_count):
                route = routes[t]
                if len(route) <= 3:
                    continue
                best_route = route[:]
                best_dist = compute_distance(route)
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_distance(new_route)
                        if new_dist < best_dist - 1e-9:
                            best_dist = new_dist
                            best_route = new_route
                if best_dist < compute_distance(routes[t]) - 1e-9:
                    routes[t] = best_route
                    improved = True
            if improved:
                continue
            # inter-route relocate (best improvement on max distance)
            best_move = None
            best_reduction = 0.0
            for src in range(truck_count):
                src_route = routes[src]
                if len(src_route) <= 2:
                    continue
                for cust_idx in range(1, len(src_route)-1):
                    cust = src_route[cust_idx]
                    new_src = src_route[:cust_idx] + src_route[cust_idx+1:]
                    for dst in range(truck_count):
                        if dst == src:
                            continue
                        dst_route = routes[dst]
                        if len(dst_route) == 2:
                            pos = 1
                            new_dst = [0, cust, 0]
                        else:
                            best_pos = 1
                            best_inc = float('inf')
                            for pos in range(1, len(dst_route)):
                                prev = dst_route[pos-1]
                                nxt = dst_route[pos]
                                inc = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                                if inc < best_inc - 1e-9:
                                    best_inc = inc
                                    best_pos = pos
                            pos = best_pos
                            new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                        new_dist_src = compute_distance(new_src)
                        new_dist_dst = compute_distance(new_dst)
                        other_dists = [distances[i] for i in range(truck_count) if i not in (src, dst)]
                        new_max = max(new_dist_src, new_dist_dst, *other_dists)
                        reduction = max_dist - new_max
                        if reduction > best_reduction + 1e-9:
                            best_reduction = reduction
                            best_move = (src, dst, new_src, new_dst)
            if best_reduction > 1e-9:
                src, dst, new_src, new_dst = best_move
                routes[src] = new_src
                routes[dst] = new_dst
                improved = True
                continue
            # inter-route swap
            best_move = None
            best_reduction = 0.0
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for t2 in range(t1+1, truck_count):
                    route2 = routes[t2]
                    if len(route2) <= 2:
                        continue
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            cust1 = route1[i]
                            cust2 = route2[j]
                            new_route1 = route1[:i] + [cust2] + route1[i+1:]
                            new_route2 = route2[:j] + [cust1] + route2[j+1:]
                            new_dist1 = compute_distance(new_route1)
                            new_dist2 = compute_distance(new_route2)
                            other_dists = [distances[k] for k in range(truck_count) if k not in (t1, t2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            reduction = max_dist - new_max
                            if reduction > best_reduction + 1e-9:
                                best_reduction = reduction
                                best_move = (t1, t2, new_route1, new_route2)
            if best_reduction > 1e-9:
                t1, t2, new_route1, new_route2 = best_move
                routes[t1] = new_route1
                routes[t2] = new_route2
                improved = True
                continue
            # inter-route 2-opt*
            best_move = None
            best_reduction = 0.0
            for t1 in range(truck_count):
                for t2 in range(t1+1, truck_count):
                    route1 = routes[t1]
                    route2 = routes[t2]
                    if len(route1) <= 2 or len(route2) <= 2:
                        continue
                    for i in range(1, len(route1)-2):
                        for j in range(1, len(route2)-2):
                            new_route1 = route1[:i+1] + route2[j+1:]
                            new_route2 = route2[:j+1] + route1[i+1:]
                            new_dist1 = compute_distance(new_route1)
                            new_dist2 = compute_distance(new_route2)
                            other_dists = [distances[k] for k in range(truck_count) if k not in (t1, t2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            reduction = max_dist - new_max
                            if reduction > best_reduction + 1e-9:
                                best_reduction = reduction
                                best_move = (t1, t2, new_route1, new_route2)
            if best_reduction > 1e-9:
                t1, t2, new_route1, new_route2 = best_move
                routes[t1] = new_route1
                routes[t2] = new_route2
                improved = True
                continue
            if not improved:
                break
        return routes

    def perturbation(routes, unassigned, rnd_state):
        # remove 20% from each route (at least 1 customer)
        new_routes = [r[:] for r in routes]
        for t in range(truck_count):
            route = new_routes[t]
            if len(route) <= 2:
                continue
            removable = route[1:-1]
            k = max(1, int(len(removable) * 0.2))
            rnd_state.shuffle(removable)
            to_remove = removable[:k]
            for cust in to_remove:
                unassigned.append(cust)
                route.remove(cust)
            new_routes[t] = [0] + [c for c in route if c != 0] + [0]
        # if no unassigned after removal? but we removed some, so fine
        # reinsert using regret-2 minimax
        new_routes = regret_insertion(new_routes, unassigned)
        return new_routes

    best_routes = None
    best_max_dist = float('inf')
    num_restarts = 5
    cust_list = list(range(1, n))
    cust_depot_dist = [(distance_matrix[0][c], c) for c in cust_list]
    cust_depot_dist.sort(key=lambda x: (-x[0], x[1]))  # descending distance for farthest seed

    for restart in range(num_restarts):
        rnd_state = random.Random(42 + restart)
        # farthest seed construction
        seeds = []
        start_idx = restart % customer_count
        first_seed = cust_depot_dist[start_idx][1]
        seeds.append(first_seed)
        remaining = [c for c in cust_list if c != first_seed]
        while len(seeds) < truck_count and remaining:
            max_min_dist = -1
            new_seed = -1
            for cust in remaining:
                min_dist = min(distance_matrix[cust][s] for s in seeds)
                if min_dist > max_min_dist + 1e-9:
                    max_min_dist = min_dist
                    new_seed = cust
            if new_seed != -1:
                seeds.append(new_seed)
                remaining.remove(new_seed)
            else:
                break
        if len(seeds) < truck_count:
            for cust in cust_list:
                if cust not in seeds:
                    seeds.append(cust)
                    if len(seeds) == truck_count:
                        break

        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = cust_list[:]
        for t in range(truck_count):
            if t < len(seeds):
                cust = seeds[t]
                routes[t] = [0, cust, 0]
                unassigned.remove(cust)
        # initial insertion using regret-2 minimax
        routes = regret_insertion(routes, unassigned.copy())
        routes = local_search(routes, max_iter=100)
        distances = [compute_distance(r) for r in routes]
        max_dist = max(distances)
        if max_dist < best_max_dist - 1e-9:
            best_max_dist = max_dist
            best_routes = [r[:] for r in routes]
            try:
                report_best_vrp(best_routes)
            except:
                pass

        # perturbation and SA
        temperature = 1.0
        for pert_round in range(5):  # perturbation rounds
            unassigned_pert = []
            new_routes = perturbation(routes, unassigned_pert, rnd_state)
            new_routes = local_search(new_routes, max_iter=100)
            new_distances = [compute_distance(r) for r in new_routes]
            new_max = max(new_distances)
            delta = new_max - max_dist
            if delta < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / temperature)
            if accept:
                routes = new_routes
                distances = new_distances
                max_dist = new_max
                if max_dist < best_max_dist - 1e-9:
                    best_max_dist = max_dist
                    best_routes = [r[:] for r in routes]
                    try:
                        report_best_vrp(best_routes)
                    except:
                        pass
            temperature *= 0.95
    return best_routes