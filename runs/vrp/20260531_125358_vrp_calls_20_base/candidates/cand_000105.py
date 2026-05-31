import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def balance_routes(routes, lengths):
        improved = True
        max_balance_iter = n * truck_count
        it = 0
        while improved and it < max_balance_iter:
            improved = False
            it += 1
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
                break
            max_route = routes[max_idx]
            best_cust = None
            best_overall_reduction = 0
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max_route = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_distance(new_max_route)
                min_route = routes[min_idx]
                best_insertion_len = float('inf')
                best_pos = -1
                for p in range(1, len(min_route)):
                    new_min_route = min_route[:p] + [cust] + min_route[p:]
                    l = route_distance(new_min_route)
                    if l < best_insertion_len:
                        best_insertion_len = l
                        best_pos = p
                new_min_route = min_route[:best_pos] + [cust] + min_route[best_pos:]
                new_min_len = route_distance(new_min_route)
                other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                new_max_global = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                old_max_global = max(lengths)
                reduction = old_max_global - new_max_global
                if reduction > best_overall_reduction:
                    best_overall_reduction = reduction
                    best_cust = (cust, best_pos)
            if best_cust is not None and best_overall_reduction > 0:
                cust, best_insert_pos = best_cust
                new_max = [node for node in max_route if node != cust]
                min_route = routes[min_idx]
                new_min = min_route[:best_insert_pos] + [cust] + min_route[best_insert_pos:]
                routes[max_idx] = new_max
                routes[min_idx] = new_min
                lengths[max_idx] = route_distance(new_max)
                lengths[min_idx] = route_distance(new_min)
                improved = True
            else:
                break
        return routes, lengths

    def vnd(routes, lengths):
        max_iter = n * truck_count
        for _ in range(max_iter):
            best_move = None
            best_new_max = max(lengths)
            best_total = sum(lengths)
            # relocate
            for cust in range(1, n):
                src_idx = None
                src_pos = None
                for i, route in enumerate(routes):
                    if cust in route:
                        src_idx = i
                        src_pos = route.index(cust)
                        break
                if src_idx is None:
                    continue
                new_src = routes[src_idx][:src_pos] + routes[src_idx][src_pos+1:]
                src_len = route_distance(new_src)
                for dst_idx in range(truck_count):
                    if dst_idx == src_idx:
                        continue
                    if len(routes[dst_idx]) <= 2:
                        continue
                    for ins_pos in range(1, len(routes[dst_idx])):
                        new_dst = routes[dst_idx][:ins_pos] + [cust] + routes[dst_idx][ins_pos:]
                        new_lengths = lengths[:]
                        new_lengths[src_idx] = src_len
                        new_lengths[dst_idx] = route_distance(new_dst)
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_total):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('relocate', src_idx, src_pos, dst_idx, ins_pos, new_src, new_dst)
            # swap
            for i in range(truck_count):
                if len(routes[i]) <= 2:
                    continue
                for ip in range(1, len(routes[i])-1):
                    cust_i = routes[i][ip]
                    for j in range(i+1, truck_count):
                        if len(routes[j]) <= 2:
                            continue
                        for jp in range(1, len(routes[j])-1):
                            cust_j = routes[j][jp]
                            new_i = routes[i][:ip] + [cust_j] + routes[i][ip+1:]
                            new_j = routes[j][:jp] + [cust_i] + routes[j][jp+1:]
                            new_lengths = lengths[:]
                            new_lengths[i] = route_distance(new_i)
                            new_lengths[j] = route_distance(new_j)
                            new_max = max(new_lengths)
                            new_total = sum(new_lengths)
                            if new_max < best_new_max or (new_max == best_new_max and new_total < best_total):
                                best_new_max = new_max
                                best_total = new_total
                                best_move = ('swap', i, ip, j, jp, new_i, new_j)
            # 2-opt
            for i in range(truck_count):
                if len(routes[i]) <= 3:
                    continue
                for a in range(1, len(routes[i])-2):
                    for b in range(a+1, len(routes[i])-1):
                        new_route = routes[i][:a] + routes[i][a:b+1][::-1] + routes[i][b+1:]
                        new_len = route_distance(new_route)
                        if new_len >= lengths[i]:
                            continue
                        new_lengths = lengths[:]
                        new_lengths[i] = new_len
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_total):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('2opt', i, a, b, new_route)
            if best_move is not None and best_new_max < max(lengths):
                if best_move[0] == 'relocate':
                    routes[best_move[1]] = best_move[5]
                    routes[best_move[3]] = best_move[6]
                elif best_move[0] == 'swap':
                    routes[best_move[1]] = best_move[5]
                    routes[best_move[3]] = best_move[6]
                else:
                    routes[best_move[1]] = best_move[4]
                lengths = [route_distance(r) for r in routes]
            else:
                break
        return routes, lengths

    # ACO parameters
    alpha = 1.0
    beta = 2.0
    rho = 0.1
    q0 = 0.0  # deterministic selection not used; use probabilistic
    tau0 = 1.0
    num_ants = min(20, n)
    max_iter = min(30, n * 2)

    # Initialize pheromone matrix
    tau = np.full((n, n), tau0, dtype=float)
    # Avoid self loops
    for i in range(n):
        tau[i, i] = 0.0

    best_routes = None
    best_max = float('inf')

    for iteration in range(max_iter):
        ant_solutions = []
        for ant_idx in range(num_ants):
            routes = [[0, 0] for _ in range(truck_count)]
            unvisited = set(customers)
            while unvisited:
                candidates = []
                for cust in unvisited:
                    for t_idx, route in enumerate(routes):
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            succ = route[pos]
                            inc = distance_matrix[prev][cust] + distance_matrix[cust][succ] - distance_matrix[prev][succ]
                            heuristic = 1.0 / (1.0 + inc)
                            pheromone_factor = (tau[prev][cust] * tau[cust][succ]) ** alpha
                            des = pheromone_factor * (heuristic ** beta)
                            candidates.append((des, cust, t_idx, pos))
                if not candidates:
                    break
                # Probabilistic selection
                total_des = sum(c[0] for c in candidates)
                r = random.random() * total_des
                cum = 0.0
                for des, cust, t_idx, pos in candidates:
                    cum += des
                    if cum >= r:
                        selected_cust = cust
                        selected_truck = t_idx
                        selected_pos = pos
                        break
                # Insert
                routes[selected_truck].insert(selected_pos, selected_cust)
                unvisited.remove(selected_cust)
            # Compute lengths
            lengths = [route_distance(r) for r in routes]
            # Apply VND and balancing
            routes, lengths = vnd(routes, lengths)
            routes, lengths = balance_routes(routes, lengths)
            current_max = max(lengths)
            ant_solutions.append((routes, lengths, current_max))
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
        # Pheromone update: evaporate
        tau *= (1 - rho)
        # Deposit on global best
        if best_routes is not None:
            Q = 1.0
            delta = Q / best_max
            for route in best_routes:
                for i in range(len(route)-1):
                    u = route[i]
                    v = route[i+1]
                    tau[u][v] += delta
    return best_routes if best_routes is not None else [[0,0] for _ in range(truck_count)]