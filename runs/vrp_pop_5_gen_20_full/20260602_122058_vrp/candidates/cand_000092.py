import numpy as np
import math
import random
from collections import defaultdict

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    def route_dist(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def compute_max_dist(routes):
        maxd = 0.0
        for r in routes:
            d = route_dist(r)
            if d > maxd:
                maxd = d
        return maxd

    def copy_routes(routes):
        return [r[:] for r in routes]

    # Regret-2 construction with farthest-first seeds
    def construct_initial(seed_node=None):
        seeds = []
        if seed_node is None:
            # farthest from depot
            first_seed = max(range(1, n), key=lambda i: (distance_matrix[0, i], -i))
        else:
            first_seed = seed_node
        seeds.append(first_seed)
        for _ in range(1, truck_count):
            best_min_dist = -1
            best_node = None
            for node in range(1, n):
                if node in seeds:
                    continue
                min_dist = min(distance_matrix[node, s] for s in seeds)
                if min_dist > best_min_dist or (min_dist == best_min_dist and (best_node is None or node < best_node)):
                    best_min_dist = min_dist
                    best_node = node
            if best_node is None:
                break
            seeds.append(best_node)
        # Initialize routes with seeds
        routes = [[0, s, 0] for s in seeds]
        remaining = [c for c in customers if c not in seeds]
        # Use regret-2 to insert remaining customers
        return regret_insert(routes, remaining)

    def regret_insert(routes, remaining):
        remaining = list(remaining)
        while remaining:
            best_cust = None
            best_regret = -1
            best_route_idx = -1
            best_pos = -1
            best_delta = float('inf')
            for cust in remaining:
                deltas = []
                for idx, route in enumerate(routes):
                    min_delta = float('inf')
                    best_pos_local = -1
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nex = route[pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nex] - distance_matrix[prev, nex]
                        if delta < min_delta:
                            min_delta = delta
                            best_pos_local = pos
                    deltas.append((min_delta, idx, best_pos_local))
                deltas.sort(key=lambda x: x[0])
                if len(deltas) >= 2:
                    regret = deltas[1][0] - deltas[0][0]
                else:
                    regret = 0
                if regret > best_regret or (regret == best_regret and deltas[0][0] < best_delta):
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = deltas[0][1]
                    best_pos = deltas[0][2]
                    best_delta = deltas[0][0]
                elif regret == best_regret and deltas[0][0] == best_delta and cust < best_cust:
                    best_cust = cust
            routes[best_route_idx].insert(best_pos, best_cust)
            remaining.remove(best_cust)
        return routes

    def perturb_critical(routes, remove_frac=0.15):
        # Identify longest routes
        route_dists = [route_dist(r) for r in routes]
        max_dist = max(route_dists)
        critical_indices = [i for i, d in enumerate(route_dists) if d >= 0.9 * max_dist]
        if not critical_indices:
            critical_indices = [route_dists.index(max_dist)]
        # Collect customers from these routes
        all_custs = []
        for idx in critical_indices:
            all_custs.extend(routes[idx][1:-1])
        if len(all_custs) == 0:
            return routes
        random.shuffle(all_custs)
        remove_count = max(1, int(remove_frac * (n-1)))
        to_remove = all_custs[:remove_count]
        # Remove
        for cust in to_remove:
            for route in routes:
                if cust in route:
                    route.remove(cust)
                    break
        # Reinsert via regret-2
        routes = regret_insert(routes, to_remove)
        return routes

    def two_opt_longest(routes):
        # apply 2-opt on longest route
        longest_idx = max(range(truck_count), key=lambda i: route_dist(routes[i]))
        route = routes[longest_idx]
        improved = True
        max_iter = 20
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            for a in range(1, len(route)-2):
                for b in range(a+1, len(route)-1):
                    old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[(b+1)%len(route)]]
                    new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[(b+1)%len(route)]]
                    if new < old:
                        route[a:b+1] = route[a:b+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        routes[longest_idx] = route
        return routes

    def simulated_annealing(initial_routes, max_iter_factor=10, initial_temp_factor=1.0, cooling_rate=0.995):
        current = copy_routes(initial_routes)
        best = copy_routes(initial_routes)
        best_max = compute_max_dist(best)
        current_max = best_max
        # initial temperature
        temp = best_max * initial_temp_factor
        # compute total possible moves (for number of iterations per temp)
        n_moves = (n-1) * truck_count  # rough estimate
        iter_per_temp = max(10, int(n_moves * 0.1))
        max_iter = max_iter_factor * n
        iteration = 0
        no_improve = 0
        stuck_iter = 0
        # list of all possible move types for random selection
        while iteration < max_iter:
            iteration += 1
            # choose random move type
            move_type = random.choice(['relocate', 'swap', '2opt', 'cross'])
            # generate random move parameters
            success = False
            if move_type == 'relocate':
                # choose random route to remove from
                i = random.randrange(truck_count)
                if len(current[i]) <= 2:
                    continue
                pos = random.randrange(1, len(current[i])-1)
                cust = current[i][pos]
                # choose random route to insert into
                j = random.randrange(truck_count)
                if j == i:
                    continue
                best_delta = float('inf')
                best_pos = -1
                for ins_pos in range(1, len(current[j])):
                    prev = current[j][ins_pos-1]
                    nex = current[j][ins_pos]
                    delta = distance_matrix[prev, cust] + distance_matrix[cust, nex] - distance_matrix[prev, nex]
                    if delta < best_delta:
                        best_delta = delta
                        best_pos = ins_pos
                # apply move tentatively
                new_route_i = current[i][:pos] + current[i][pos+1:]
                new_route_j = current[j][:best_pos] + [cust] + current[j][best_pos:]
                new_max = compute_max_dist([new_route_i if k==i else new_route_j if k==j else current[k] for k in range(truck_count)])
                delta = new_max - current_max
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current[i] = new_route_i
                    current[j] = new_route_j
                    current_max = new_max
                    success = True
            elif move_type == 'swap':
                i = random.randrange(truck_count)
                j = random.randrange(truck_count)
                if i == j:
                    continue
                if len(current[i]) <= 2 or len(current[j]) <= 2:
                    continue
                pos_i = random.randrange(1, len(current[i])-1)
                pos_j = random.randrange(1, len(current[j])-1)
                cust_i = current[i][pos_i]
                cust_j = current[j][pos_j]
                new_route_i = current[i][:]
                new_route_i[pos_i] = cust_j
                new_route_j = current[j][:]
                new_route_j[pos_j] = cust_i
                new_max = compute_max_dist([new_route_i if k==i else new_route_j if k==j else current[k] for k in range(truck_count)])
                delta = new_max - current_max
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current[i] = new_route_i
                    current[j] = new_route_j
                    current_max = new_max
                    success = True
            elif move_type == '2opt':
                i = random.randrange(truck_count)
                if len(current[i]) <= 3:
                    continue
                a = random.randrange(1, len(current[i])-2)
                b = random.randrange(a+1, len(current[i])-1)
                new_route = current[i][:a] + current[i][a:b+1][::-1] + current[i][b+1:]
                new_max = max(compute_max_dist([new_route if k==i else current[k] for k in range(truck_count)]))
                delta = new_max - current_max
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current[i] = new_route
                    current_max = new_max
                    success = True
            elif move_type == 'cross':
                i = random.randrange(truck_count)
                j = random.randrange(truck_count)
                if i == j:
                    continue
                if len(current[i]) <= 2 or len(current[j]) <= 2:
                    continue
                cut_i = random.randrange(0, len(current[i])-1)
                cut_j = random.randrange(0, len(current[j])-1)
                new_i = current[i][:cut_i+1] + current[j][cut_j+1:]
                new_j = current[j][:cut_j+1] + current[i][cut_i+1:]
                if len(new_i) < 2 or len(new_j) < 2:
                    continue
                new_max = compute_max_dist([new_i if k==i else new_j if k==j else current[k] for k in range(truck_count)])
                delta = new_max - current_max
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current[i] = new_i
                    current[j] = new_j
                    current_max = new_max
                    success = True
            if success:
                if current_max < best_max:
                    best_max = current_max
                    best = copy_routes(current)
                    report_best_vrp(best)
                    no_improve = 0
                else:
                    no_improve += 1
            # cooling schedule
            if iteration % iter_per_temp == 0:
                temp *= cooling_rate
            # If stuck for many iterations, apply perturbation
            if no_improve >= 50 and iteration < max_iter - 10:
                current = perturb_critical(best)
                current = two_opt_longest(current)
                current_max = compute_max_dist(current)
                if current_max < best_max:
                    best_max = current_max
                    best = copy_routes(current)
                    report_best_vrp(best)
                no_improve = 0
                temp = best_max * initial_temp_factor  # reset temperature? maybe not
        return best

    # Main: run multiple restarts
    best_routes = None
    best_max = float('inf')
    for restart in range(3):
        if restart == 0:
            seed_node = None
        elif restart == 1:
            # different seed: second farthest from depot
            seeds_from_depot = sorted(range(1, n), key=lambda i: distance_matrix[0, i], reverse=True)
            seed_node = seeds_from_depot[min(restart, len(seeds_from_depot)-1)]
        else:
            seed_node = random.choice(customers)
        initial = construct_initial(seed_node)
        # initial local descent? not needed
        result = simulated_annealing(initial)
        result_max = compute_max_dist(result)
        if result_max < best_max:
            best_max = result_max
            best_routes = copy_routes(result)
    # Ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    report_best_vrp(best_routes)
    return best_routes