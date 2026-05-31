import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    # Construction: greedy insertion minimizing max route distance
    routes = [[0, 0] for _ in range(truck_count)]
    unvisited = list(range(1, n))
    random.shuffle(unvisited)
    for cust in unvisited:
        best_route = None
        best_pos = None
        best_max = float('inf')
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_len = route_distance(new_route)
                new_max = max(route_distance(r) for r in routes) if len(routes) > 1 else new_len
                new_max = max(new_max, new_len)
                if new_max < best_max:
                    best_max = new_max
                    best_route = r_idx
                    best_pos = pos
        routes[best_route].insert(best_pos, cust)

    lengths = [route_distance(r) for r in routes]
    best_max = max(lengths)
    best_routes = [r[:] for r in routes]

    # Tabu Search parameters
    tabu_tenure = 10
    stagnation_limit = 50
    max_iter = n * truck_count * 2
    stagnation = 0
    tabu_list = {}

    for iteration in range(max_iter):
        best_move = None
        best_new_max = best_max
        best_new_total = sum(lengths)

        # Inter-route relocate
        for cust in range(1, n):
            src_idx = None
            src_pos = None
            for r_idx, route in enumerate(routes):
                if cust in route:
                    src_idx = r_idx
                    src_pos = route.index(cust)
                    break
            if src_idx is None:
                continue
            src_route = routes[src_idx]
            new_src_route = src_route[:src_pos] + src_route[src_pos+1:]
            src_len = route_distance(new_src_route)
            for dst_idx in range(truck_count):
                if dst_idx == src_idx:
                    continue
                dst_route = routes[dst_idx]
                for ins_pos in range(1, len(dst_route)):
                    new_dst_route = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                    new_lengths = lengths[:]
                    new_lengths[src_idx] = src_len
                    new_lengths[dst_idx] = route_distance(new_dst_route)
                    new_max = max(new_lengths)
                    new_total = sum(new_lengths)
                    move_key = ('relocate', src_idx, dst_idx, cust)
                    tabu = tabu_list.get(move_key, 0) > 0
                    if (new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total)):
                        if new_max < best_max:
                            # aspiration
                            best_move = (new_src_route, new_dst_route, src_idx, dst_idx, move_key)
                            best_new_max = new_max
                            best_new_total = new_total
                        elif not tabu:
                            best_move = (new_src_route, new_dst_route, src_idx, dst_idx, move_key)
                            best_new_max = new_max
                            best_new_total = new_total

        # Inter-route swap
        for i_idx in range(truck_count):
            i_route = routes[i_idx]
            if len(i_route) <= 2:
                continue
            for i_pos in range(1, len(i_route)-1):
                cust_i = i_route[i_pos]
                for j_idx in range(i_idx+1, truck_count):
                    j_route = routes[j_idx]
                    if len(j_route) <= 2:
                        continue
                    for j_pos in range(1, len(j_route)-1):
                        cust_j = j_route[j_pos]
                        new_i_route = i_route[:i_pos] + [cust_j] + i_route[i_pos+1:]
                        new_j_route = j_route[:j_pos] + [cust_i] + j_route[j_pos+1:]
                        new_lengths = lengths[:]
                        new_lengths[i_idx] = route_distance(new_i_route)
                        new_lengths[j_idx] = route_distance(new_j_route)
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        move_key = ('swap', i_idx, j_idx, cust_i, cust_j)
                        tabu = tabu_list.get(move_key, 0) > 0
                        if (new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total)):
                            if new_max < best_max:
                                best_move = (new_i_route, new_j_route, i_idx, j_idx, move_key)
                                best_new_max = new_max
                                best_new_total = new_total
                            elif not tabu:
                                best_move = (new_i_route, new_j_route, i_idx, j_idx, move_key)
                                best_new_max = new_max
                                best_new_total = new_total

        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_len = route_distance(new_route)
                    if new_len >= lengths[r_idx]:
                        continue
                    new_lengths = lengths[:]
                    new_lengths[r_idx] = new_len
                    new_max = max(new_lengths)
                    new_total = sum(new_lengths)
                    move_key = ('2opt', r_idx, i, j)
                    tabu = tabu_list.get(move_key, 0) > 0
                    if (new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total)):
                        if new_max < best_max:
                            best_move = (new_route, r_idx, move_key)
                            best_new_max = new_max
                            best_new_total = new_total
                        elif not tabu:
                            best_move = (new_route, r_idx, move_key)
                            best_new_max = new_max
                            best_new_total = new_total

        if best_move is None:
            # No improving move, apply random perturbation
            for _ in range(5):
                r1 = random.randrange(truck_count)
                r2 = random.randrange(truck_count)
                if r1 == r2 or len(routes[r1]) <= 2 or len(routes[r2]) <= 2:
                    continue
                if random.random() < 0.5:
                    pos1 = random.randint(1, len(routes[r1])-2)
                    cust = routes[r1][pos1]
                    new_r1 = routes[r1][:pos1] + routes[r1][pos1+1:]
                    pos2 = random.randint(1, len(routes[r2])-1)
                    new_r2 = routes[r2][:pos2] + [cust] + routes[r2][pos2:]
                    routes[r1] = new_r1
                    routes[r2] = new_r2
                else:
                    pos1 = random.randint(1, len(routes[r1])-2)
                    pos2 = random.randint(1, len(routes[r2])-2)
                    cust1 = routes[r1][pos1]
                    cust2 = routes[r2][pos2]
                    new_r1 = routes[r1][:pos1] + [cust2] + routes[r1][pos1+1:]
                    new_r2 = routes[r2][:pos2] + [cust1] + routes[r2][pos2+1:]
                    routes[r1] = new_r1
                    routes[r2] = new_r2
            lengths = [route_distance(r) for r in routes]
            current_max = max(lengths)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
            stagnation = 0
            continue

        # Apply move
        if len(best_move) == 5:  # relocate
            new_src, new_dst, src_idx, dst_idx, move_key = best_move
            routes[src_idx] = new_src
            routes[dst_idx] = new_dst
        elif len(best_move) == 4:  # swap
            new_i, new_j, i_idx, j_idx, move_key = best_move
            routes[i_idx] = new_i
            routes[j_idx] = new_j
        else:  # 2opt
            new_route, r_idx, move_key = best_move
            routes[r_idx] = new_route

        lengths = [route_distance(r) for r in routes]
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Update tabu list
        for key in list(tabu_list.keys()):
            tabu_list[key] -= 1
            if tabu_list[key] <= 0:
                del tabu_list[key]
        tabu_list[move_key] = tabu_tenure

        if current_max >= best_max:
            stagnation += 1
        else:
            stagnation = 0

        if stagnation >= stagnation_limit:
            # Perturbation
            for _ in range(10):
                r1 = random.randrange(truck_count)
                r2 = random.randrange(truck_count)
                if r1 == r2 or len(routes[r1]) <= 2 or len(routes[r2]) <= 2:
                    continue
                if random.random() < 0.5:
                    pos1 = random.randint(1, len(routes[r1])-2)
                    cust = routes[r1][pos1]
                    new_r1 = routes[r1][:pos1] + routes[r1][pos1+1:]
                    pos2 = random.randint(1, len(routes[r2])-1)
                    new_r2 = routes[r2][:pos2] + [cust] + routes[r2][pos2:]
                    routes[r1] = new_r1
                    routes[r2] = new_r2
                else:
                    pos1 = random.randint(1, len(routes[r1])-2)
                    pos2 = random.randint(1, len(routes[r2])-2)
                    cust1 = routes[r1][pos1]
                    cust2 = routes[r2][pos2]
                    new_r1 = routes[r1][:pos1] + [cust2] + routes[r1][pos1+1:]
                    new_r2 = routes[r2][:pos2] + [cust1] + routes[r2][pos2+1:]
                    routes[r1] = new_r1
                    routes[r2] = new_r2
            lengths = [route_distance(r) for r in routes]
            current_max = max(lengths)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
            stagnation = 0

    # Ensure exactly truck_count routes, each starting and ending at 0
    for r in best_routes:
        if r[0] != 0:
            r.insert(0, 0)
        if r[-1] != 0:
            r.append(0)
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    return best_routes[:truck_count]