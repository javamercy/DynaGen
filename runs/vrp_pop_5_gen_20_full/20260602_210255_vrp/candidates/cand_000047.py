import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        routes += [[0, 0]] * (truck_count - (n - 1))
        return routes
    
    # Construction: farthest-point seeds with random tie-breaking
    seeds = []
    # first seed: farthest from depot
    far = max(customers, key=lambda c: distance_matrix[0][c])
    seeds.append(far)
    while len(seeds) < truck_count:
        best_custs = []
        best_min_dist = -1
        for c in customers:
            if c in seeds:
                continue
            min_dist = min(distance_matrix[c][s] for s in seeds)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_custs = [c]
            elif min_dist == best_min_dist:
                best_custs.append(c)
        seed = random.choice(best_custs)
        seeds.append(seed)
    
    # Initial routes from seeds
    routes = [[0, s, 0] for s in seeds]
    route_dists = [distance_matrix[0][s] + distance_matrix[s][0] for s in seeds]
    assigned = set(seeds)
    
    # Greedy insertion of remaining customers, farthest first
    remaining = [c for c in customers if c not in assigned]
    remaining.sort(key=lambda c: distance_matrix[0][c], reverse=True)
    
    for cust in remaining:
        best_new_max = float('inf')
        best_ops = []
        for idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                prev = route[pos-1]
                nxt = route[pos]
                new_dist = route_dists[idx] - distance_matrix[prev][nxt] + distance_matrix[prev][cust] + distance_matrix[cust][nxt]
                new_max = max(route_dists[:idx] + [new_dist] + route_dists[idx+1:])
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_ops = [(idx, pos, new_dist)]
                elif new_max == best_new_max:
                    best_ops.append((idx, pos, new_dist))
        idx, pos, new_dist = random.choice(best_ops)
        routes[idx].insert(pos, cust)
        route_dists[idx] = new_dist
        assigned.add(cust)
    
    best_routes = [r[:] for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)
    
    def route_dist(route):
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    
    max_passes = min(100, n * 2)
    for _ in range(max_passes):
        improved = False
        # Inter-route relocate with random order
        route_indices = list(range(len(routes)))
        random.shuffle(route_indices)
        for i in route_indices:
            route_i = routes[i]
            if len(route_i) <= 3:
                continue
            # iterate over positions in random order
            pos_indices = list(range(1, len(route_i)-1))
            random.shuffle(pos_indices)
            for pos_i in pos_indices:
                cust = route_i[pos_i]
                # try other routes in random order
                other_indices = [j for j in range(len(routes)) if j != i]
                random.shuffle(other_indices)
                for j in other_indices:
                    route_j = routes[j]
                    # insertion positions in random order
                    pos_j_list = list(range(1, len(route_j)+1))
                    random.shuffle(pos_j_list)
                    for pos_j in pos_j_list:
                        prev_i = route_i[pos_i-1]
                        next_i = route_i[pos_i+1]
                        new_dist_i = route_dists[i] - distance_matrix[prev_i][cust] - distance_matrix[cust][next_i] + distance_matrix[prev_i][next_i]
                        if len(route_i) == 3:
                            new_dist_i = 0.0
                        if pos_j == len(route_j):
                            prev_j = route_j[pos_j-1]
                            new_dist_j = route_dists[j] + distance_matrix[prev_j][cust] + distance_matrix[cust][0] - distance_matrix[prev_j][0]
                        else:
                            prev_j = route_j[pos_j-1]
                            next_j = route_j[pos_j]
                            new_dist_j = route_dists[j] + distance_matrix[prev_j][cust] + distance_matrix[cust][next_j] - distance_matrix[prev_j][next_j]
                        new_max = max(route_dists[:i] + [new_dist_i] + route_dists[i+1:j] + [new_dist_j] + route_dists[j+1:])
                        if new_max < best_max:
                            del routes[i][pos_i]
                            if len(routes[i]) == 2:
                                routes[i] = [0, 0]
                            routes[j].insert(pos_j, cust)
                            route_dists[i] = new_dist_i if len(routes[i]) > 2 else 0.0
                            route_dists[j] = new_dist_j
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
                            improved = True
                            report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Intra-route 2-opt with random order
        route_indices = list(range(len(routes)))
        random.shuffle(route_indices)
        for idx in route_indices:
            route = routes[idx]
            if len(route) <= 4:
                continue
            best_improvement = None
            i_list = list(range(1, len(route)-2))
            random.shuffle(i_list)
            for i in i_list:
                j_list = list(range(i+1, len(route)-1))
                random.shuffle(j_list)
                for j in j_list:
                    if j == i+1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < route_dists[idx]:
                        routes[idx] = new_route
                        route_dists[idx] = new_dist
                        new_max = max(route_dists)
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
                            improved = True
                            report_best_vrp(best_routes)
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best_routes