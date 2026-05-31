import numpy as np
import random
from math import exp

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]
    
    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    def max_route_len(routes):
        return max(route_length(r) for r in routes) if routes else float('inf')
    
    # Regret-2 construction
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))
    while unassigned:
        best_cust = None
        best_regret = -1.0
        best_data = None
        for cust in unassigned:
            insert_info = []
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_len = route_length(route) + cost
                    other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                    new_max = max(new_len, *other_lens)
                    insert_info.append((new_max, cost, r_idx, pos))
            if not insert_info:
                continue
            insert_info.sort(key=lambda x: (x[0], x[1]))
            best = insert_info[0]
            second = insert_info[1] if len(insert_info) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
            regret = second[0] - best[0]
            if best_cust is None or regret > best_regret or (regret == best_regret and best[1] < best_data[1]):
                best_cust = cust
                best_regret = regret
                best_data = (best[0], best[1], best[2], best[3])
        if best_cust is None:
            break
        _, _, r_idx, pos = best_data
        routes[r_idx].insert(pos, best_cust)
        unassigned.remove(best_cust)
    
    best_routes = [r[:] for r in routes]
    best_max = max_route_len(routes)
    report_best_vrp(routes)
    
    def compute_new_max(routes, route_lens, move_type, move_data):
        if move_type == 'relocate':
            cust, from_idx, to_idx, pos = move_data
            new_from = [x for x in routes[from_idx] if x != cust]
            new_to = routes[to_idx][:pos] + [cust] + routes[to_idx][pos:]
            new_lens = list(route_lens)
            new_lens[from_idx] = route_length(new_from)
            new_lens[to_idx] = route_length(new_to)
            return max(new_lens)
        elif move_type == 'swap':
            cust_i, from_idx, cust_j, to_idx = move_data
            new_from = [cust_j if x == cust_i else x for x in routes[from_idx]]
            new_to = [cust_i if x == cust_j else x for x in routes[to_idx]]
            new_lens = list(route_lens)
            new_lens[from_idx] = route_length(new_from)
            new_lens[to_idx] = route_length(new_to)
            return max(new_lens)
        return float('inf')
    
    # Local search
    max_iter = max(1, (n-1) * truck_count * 2)
    avg_len = sum(route_length(r) for r in routes) / truck_count
    T = avg_len * 0.05
    if T < 1e-12:
        T = 1.0
    cooling = 0.99
    stagnation = 0
    for iteration in range(max_iter):
        route_lens = [route_length(r) for r in routes]
        max_len = max(route_lens)
        longest_idx = route_lens.index(max_len)
        best_delta = 0.0
        best_move = None
        best_move_type = None
        # Inter-relocate from longest route
        for cust in routes[longest_idx][1:-1]:
            for t_idx in range(truck_count):
                if t_idx == longest_idx:
                    continue
                for pos in range(1, len(routes[t_idx])):
                    new_from = [x for x in routes[longest_idx] if x != cust]
                    new_to = routes[t_idx][:pos] + [cust] + routes[t_idx][pos:]
                    new_lens = list(route_lens)
                    new_lens[longest_idx] = route_length(new_from)
                    new_lens[t_idx] = route_length(new_to)
                    new_max = max(new_lens)
                    if new_max < max_len - 1e-12:
                        delta = max_len - new_max
                        if delta > best_delta:
                            best_delta = delta
                            best_move = (cust, longest_idx, t_idx, pos)
                            best_move_type = 'relocate'
        # Inter-swap between longest and other routes
        if truck_count > 1:
            for cust_i in routes[longest_idx][1:-1]:
                for t_idx in range(truck_count):
                    if t_idx == longest_idx:
                        continue
                    for cust_j in routes[t_idx][1:-1]:
                        new_from = [cust_j if x == cust_i else x for x in routes[longest_idx]]
                        new_to = [cust_i if x == cust_j else x for x in routes[t_idx]]
                        new_lens = list(route_lens)
                        new_lens[longest_idx] = route_length(new_from)
                        new_lens[t_idx] = route_length(new_to)
                        new_max = max(new_lens)
                        if new_max < max_len - 1e-12:
                            delta = max_len - new_max
                            if delta > best_delta:
                                best_delta = delta
                                best_move = (cust_i, longest_idx, cust_j, t_idx)
                                best_move_type = 'swap'
        if best_move is not None:
            if best_move_type == 'relocate':
                cust, from_idx, to_idx, pos = best_move
                routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                routes[to_idx].insert(pos, cust)
            else:
                cust_i, from_idx, cust_j, to_idx = best_move
                routes[from_idx] = [cust_j if x == cust_i else x for x in routes[from_idx]]
                routes[to_idx] = [cust_i if x == cust_j else x for x in routes[to_idx]]
            new_max = max_route_len(routes)
            if new_max < best_max - 1e-12:
                best_max = new_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(routes)
            stagnation = 0
        else:
            # Accept non-improving move with probability based on T
            # For exploitation, we only accept moves that don't worsen max by too much
            # Simple: accept with probability exp(-delta/T) where delta = new_max - max_len
            # But we need a candidate move. Pick a random reloc/swap that is not improving.
            # We'll generate a random feasible move
            move_found = False
            for _ in range(20):  # try 20 random moves
                if random.random() < 0.5:
                    # random relocate
                    src = random.randint(0, truck_count-1)
                    if len(routes[src]) <= 2:
                        continue
                    cust = random.choice(routes[src][1:-1])
                    dst = random.randint(0, truck_count-1)
                    if dst == src:
                        continue
                    pos = random.randint(1, len(routes[dst])-1)
                    new_from = [x for x in routes[src] if x != cust]
                    new_to = routes[dst][:pos] + [cust] + routes[dst][pos:]
                    new_lens = list(route_lens)
                    new_lens[src] = route_length(new_from)
                    new_lens[dst] = route_length(new_to)
                    new_max = max(new_lens)
                    delta = new_max - max_len
                    if delta <= 0 or random.random() < exp(-delta / T):
                        routes[src] = new_from
                        routes[dst] = new_to
                        move_found = True
                        break
                else:
                    # random swap
                    src = random.randint(0, truck_count-1)
                    if len(routes[src]) <= 2:
                        continue
                    dst = random.randint(0, truck_count-1)
                    if dst == src:
                        continue
                    if len(routes[dst]) <= 2:
                        continue
                    cust_i = random.choice(routes[src][1:-1])
                    cust_j = random.choice(routes[dst][1:-1])
                    new_src = [cust_j if x == cust_i else x for x in routes[src]]
                    new_dst = [cust_i if x == cust_j else x for x in routes[dst]]
                    new_lens = list(route_lens)
                    new_lens[src] = route_length(new_src)
                    new_lens[dst] = route_length(new_dst)
                    new_max = max(new_lens)
                    delta = new_max - max_len
                    if delta <= 0 or random.random() < exp(-delta / T):
                        routes[src] = new_src
                        routes[dst] = new_dst
                        move_found = True
                        break
            if move_found:
                new_max = max_route_len(routes)
                if new_max < best_max - 1e-12:
                    best_max = new_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(routes)
                stagnation = 0
            else:
                stagnation += 1
        # Intra-route 2-opt on all routes every 5 iterations
        if iteration % 5 == 0:
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                improved = True
                while improved:
                    improved = False
                    for i in range(1, len(route)-2):
                        for k in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                            if route_length(new_route) < route_length(route) - 1e-12:
                                route = new_route
                                improved = True
                                break
                        if improved:
                            break
                routes[r_idx] = route
        # Temperature update
        T *= cooling
        if T < 1e-12:
            T = 1e-12
        # Perturbation if stagnation
        if stagnation >= 10:
            # Remove up to 20% of customers from routes with largest length (focus on longest)
            sorted_routes = sorted(range(truck_count), key=lambda i: route_lens[i], reverse=True)
            num_remove = max(1, int((n-1) * 0.10))
            removed = []
            for r_idx in sorted_routes:
                if len(routes[r_idx]) <= 2:
                    continue
                can_remove = min(num_remove - len(removed), len(routes[r_idx])-2)
                if can_remove <= 0:
                    break
                candidates = routes[r_idx][1:-1]
                remove_set = set(random.sample(candidates, can_remove))
                for cust in remove_set:
                    removed.append((r_idx, cust))
                routes[r_idx] = [x for x in routes[r_idx] if x not in remove_set]
                if len(removed) >= num_remove:
                    break
            # Reinsert all removed customers using regret-2
            unassigned = [cust for _, cust in removed]
            random.shuffle(unassigned)
            while unassigned:
                best_cust = None
                best_regret = -1.0
                best_data = None
                for cust in unassigned:
                    insert_info = []
                    for r_idx, route in enumerate(routes):
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                            new_len = route_length(route) + cost
                            other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                            new_max = max(new_len, *other_lens)
                            insert_info.append((new_max, cost, r_idx, pos))
                    if not insert_info:
                        continue
                    insert_info.sort(key=lambda x: (x[0], x[1]))
                    best = insert_info[0]
                    second = insert_info[1] if len(insert_info) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                    regret = second[0] - best[0]
                    if best_cust is None or regret > best_regret or (regret == best_regret and best[1] < best_data[1]):
                        best_cust = cust
                        best_regret = regret
                        best_data = (best[0], best[1], best[2], best[3])
                if best_cust is None:
                    break
                _, _, r_idx, pos = best_data
                routes[r_idx].insert(pos, best_cust)
                unassigned.remove(best_cust)
            new_max = max_route_len(routes)
            if new_max < best_max - 1e-12:
                best_max = new_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(routes)
            stagnation = 0
    return best_routes