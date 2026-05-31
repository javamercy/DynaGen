import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    max_dist = np.max(distance_matrix)
    
    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def two_opt(route):
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
        return route
    
    def balance_routes(routes):
        lengths = [route_distance(r) for r in routes]
        for _ in range(10 * n):
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
                break
            max_route = routes[max_idx]
            best_saving = -float('inf')
            best_move = None
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_distance(new_max)
                min_route = routes[min_idx]
                for ins in range(1, len(min_route)):
                    new_min = min_route[:ins] + [cust] + min_route[ins:]
                    new_min_len = route_distance(new_min)
                    new_lengths = lengths[:]
                    new_lengths[max_idx] = new_max_len
                    new_lengths[min_idx] = new_min_len
                    new_max_val = max(new_lengths)
                    saving = lengths[max_idx] - new_max_val  # reduction in max
                    if saving > best_saving:
                        best_saving = saving
                        best_move = (max_idx, min_idx, pos, ins)
            if best_move and best_saving > 0:
                max_idx, min_idx, pos, ins = best_move
                cust = routes[max_idx][pos]
                new_max = routes[max_idx][:pos] + routes[max_idx][pos+1:]
                new_min = routes[min_idx][:ins] + [cust] + routes[min_idx][ins:]
                routes[max_idx] = new_max
                routes[min_idx] = new_min
                lengths[max_idx] = route_distance(new_max)
                lengths[min_idx] = route_distance(new_min)
            else:
                break
        return routes, lengths
    
    def regret_construction():
        routes = [[0,0] for _ in range(truck_count)]
        unvisited = set(customers)
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route = -1
            best_pos = -1
            for cust in unvisited:
                incs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        incs.append((inc, pos, r_idx))
                incs.sort(key=lambda x: x[0])
                if len(incs) >= 3:
                    regret = incs[1][0] - incs[0][0] + incs[2][0] - incs[0][0]
                else:
                    regret = 0.0
                inc = incs[0][0]
                pos = incs[0][1]
                r_idx = incs[0][2]
                if regret > best_regret or (regret == best_regret and inc < best_inc):
                    best_regret = regret
                    best_inc = inc
                    best_cust = cust
                    best_route = r_idx
                    best_pos = pos
            routes[best_route].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        return routes
    
    def local_search(routes, lengths):
        improved = True
        while improved:
            improved = False
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
                new_src = routes[src_idx][:src_pos] + routes[src_idx][src_pos+1:]
                src_len = route_distance(new_src)
                for dst_idx in range(truck_count):
                    if dst_idx == src_idx:
                        continue
                    dst_route = routes[dst_idx]
                    if len(dst_route) <= 2:
                        continue
                    for ins_pos in range(1, len(dst_route)):
                        new_dst = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                        new_lengths = lengths[:]
                        new_lengths[src_idx] = src_len
                        new_lengths[dst_idx] = route_distance(new_dst)
                        new_max = max(new_lengths)
                        if new_max < max(lengths):
                            routes[src_idx] = new_src
                            routes[dst_idx] = new_dst
                            lengths = new_lengths
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
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
                            new_i = i_route[:i_pos] + [cust_j] + i_route[i_pos+1:]
                            new_j = j_route[:j_pos] + [cust_i] + j_route[j_pos+1:]
                            new_lengths = lengths[:]
                            new_lengths[i_idx] = route_distance(new_i)
                            new_lengths[j_idx] = route_distance(new_j)
                            new_max = max(new_lengths)
                            if new_max < max(lengths):
                                routes[i_idx] = new_i
                                routes[j_idx] = new_j
                                lengths = new_lengths
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_len = route_distance(new_route)
                        if new_len < lengths[r_idx]:
                            routes[r_idx] = new_route
                            lengths[r_idx] = new_len
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes, lengths
    
    def ruin_recreate(routes, lengths, fraction=0.15):
        n_cust = n - 1
        num_remove = max(1, int(n_cust * fraction))
        savings = []
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)-1):
                cust = route[pos]
                prev = route[pos-1]
                next = route[pos+1]
                saving = distance_matrix[prev, cust] + distance_matrix[cust, next] - distance_matrix[prev, next]
                savings.append((saving, cust, r_idx, pos))
        savings.sort(reverse=True)
        to_remove = [item[1] for item in savings[:num_remove]]
        new_routes = [[0,0] for _ in range(truck_count)]
        for r_idx, route in enumerate(routes):
            new_routes[r_idx] = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
        unvisited = set(to_remove)
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route = -1
            best_pos = -1
            for cust in unvisited:
                incs = []
                for r_idx, route in enumerate(new_routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        incs.append((inc, pos, r_idx))
                incs.sort(key=lambda x: x[0])
                if len(incs) >= 3:
                    regret = incs[1][0] - incs[0][0] + incs[2][0] - incs[0][0]
                else:
                    regret = 0.0
                inc = incs[0][0]
                pos = incs[0][1]
                r_idx = incs[0][2]
                if regret > best_regret or (regret == best_regret and inc < best_inc):
                    best_regret = regret
                    best_inc = inc
                    best_cust = cust
                    best_route = r_idx
                    best_pos = pos
            new_routes[best_route].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        new_lengths = [route_distance(r) for r in new_routes]
        new_routes, new_lengths = balance_routes(new_routes)
        for r_idx in range(truck_count):
            if len(new_routes[r_idx]) > 2:
                new_routes[r_idx] = two_opt(new_routes[r_idx])
        new_lengths = [route_distance(r) for r in new_routes]
        return new_routes, new_lengths
    
    best_routes = None
    best_max = float('inf')
    num_restarts = 3  # fixed small number
    for restart in range(num_restarts):
        routes = regret_construction()
        lengths = [route_distance(r) for r in routes]
        routes, lengths = balance_routes(routes)
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        
        for phase in range(10):  # limited number of ruin phases
            routes, lengths = local_search(routes, lengths)
            current_max = max(lengths)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
            # Apply ruin and recreate
            routes, lengths = ruin_recreate(routes, lengths, fraction=0.15)
            current_max = max(lengths)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
    
    return best_routes