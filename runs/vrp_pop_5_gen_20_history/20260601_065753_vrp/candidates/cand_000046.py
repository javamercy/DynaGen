import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def compute_max(routes):
        return max(route_length(r) for r in routes)

    # Initial construction: greedy min-max insertion (random order to reduce bias)
    routes = [[0, 0] for _ in range(truck_count)]
    lengths = [0.0] * truck_count
    unassigned = customers[:]
    random.shuffle(unassigned)

    for cust in unassigned:
        best_max = float('inf')
        best_r = -1
        best_p = -1
        for r in range(truck_count):
            route = routes[r]
            for p in range(1, len(route)):
                prev = route[p-1]
                nxt = route[p]
                new_len = lengths[r] - distance_matrix[prev, nxt] + distance_matrix[prev, cust] + distance_matrix[cust, nxt]
                new_max = new_len
                for rr in range(truck_count):
                    if rr != r and lengths[rr] > new_max:
                        new_max = lengths[rr]
                if new_max < best_max or (new_max == best_max and (r < best_r or (r == best_r and p < best_p))):
                    best_max = new_max
                    best_r = r
                    best_p = p
        routes[best_r].insert(best_p, cust)
        lengths[best_r] += distance_matrix[routes[best_r][best_p-1], cust] + distance_matrix[cust, routes[best_r][best_p+1]] - distance_matrix[routes[best_r][best_p-1], routes[best_r][best_p+1]]
    best_routes = [list(r) for r in routes]
    best_max = compute_max(routes)

    # Simulated Annealing
    def copy_routes(rs):
        return [list(r) for r in rs]

    def random_customer_from_route(route):
        # return index in route list (excluding depots)
        if len(route) <= 2:
            return None
        return random.randint(1, len(route)-2)

    def relocate(routes, lengths):
        # pick a customer from the longest route and move to another
        max_len = max(lengths)
        candidates = [i for i, l in enumerate(lengths) if l == max_len]
        src_r = random.choice(candidates)
        src_route = routes[src_r]
        cust_idx = random_customer_from_route(src_route)
        if cust_idx is None:
            return False, None, None
        cust = src_route[cust_idx]
        # remove cust from src
        new_src = src_route[:cust_idx] + src_route[cust_idx+1:]
        new_src_len = route_length(new_src)
        # choose a different target route
        tgt_r = random.randrange(truck_count)
        while tgt_r == src_r:
            tgt_r = random.randrange(truck_count)
        tgt_route = routes[tgt_r]
        # choose random insertion position
        pos = random.randint(1, len(tgt_route)-1)
        new_tgt = tgt_route[:pos] + [cust] + tgt_route[pos:]
        new_tgt_len = route_length(new_tgt)
        # compute new max
        new_max = max(new_src_len, new_tgt_len)
        for i in range(truck_count):
            if i != src_r and i != tgt_r:
                new_max = max(new_max, lengths[i])
        if new_max < best_max or (_accept(new_max - best_max)):
            # apply
            routes[src_r] = new_src
            routes[tgt_r] = new_tgt
            lengths[src_r] = new_src_len
            lengths[tgt_r] = new_tgt_len
            return True, new_max, routes
        return False, None, None

    def swap(routes, lengths):
        # pick two routes and swap random customers
        if truck_count < 2:
            return False, None, None
        r1 = random.randrange(truck_count)
        r2 = random.randrange(truck_count)
        while r2 == r1:
            r2 = random.randrange(truck_count)
        route1 = routes[r1]
        route2 = routes[r2]
        if len(route1) <= 2 or len(route2) <= 2:
            return False, None, None
        idx1 = random.randint(1, len(route1)-2)
        idx2 = random.randint(1, len(route2)-2)
        cust1 = route1[idx1]
        cust2 = route2[idx2]
        new1 = route1[:idx1] + [cust2] + route1[idx1+1:]
        new2 = route2[:idx2] + [cust1] + route2[idx2+1:]
        new_len1 = route_length(new1)
        new_len2 = route_length(new2)
        new_max = max(new_len1, new_len2)
        for i in range(truck_count):
            if i != r1 and i != r2:
                new_max = max(new_max, lengths[i])
        if new_max < best_max or (_accept(new_max - best_max)):
            routes[r1] = new1
            routes[r2] = new2
            lengths[r1] = new_len1
            lengths[r2] = new_len2
            return True, new_max, routes
        return False, None, None

    def two_opt(routes, lengths):
        # pick a route (weighted towards longest) and apply a 2-opt move
        max_len = max(lengths)
        if random.random() < 0.5:
            candidates = [i for i, l in enumerate(lengths) if l == max_len]
            r = random.choice(candidates)
        else:
            r = random.randrange(truck_count)
        route = routes[r]
        if len(route) <= 3:  # need at least 2 customers to reverse
            return False, None, None
        i = random.randint(1, len(route)-3)
        j = random.randint(i+1, len(route)-2)
        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
        new_len = route_length(new_route)
        new_max = max(new_len, max_len) if r != list(l for l in lengths).index(max_len) else max(new_len, max((l for idx,l in enumerate(lengths) if idx!=r), default=0))
        # compute new_max carefully
        new_max = new_len
        for idx, l in enumerate(lengths):
            if idx != r:
                new_max = max(new_max, l)
        if new_max < best_max or (_accept(new_max - best_max)):
            routes[r] = new_route
            lengths[r] = new_len
            return True, new_max, routes
        return False, None, None

    def _accept(delta):
        if delta < 0:
            return True
        else:
            return random.random() < math.exp(-delta / temperature)

    # SA parameters
    max_iter = 50 * n
    init_temp = best_max * 0.1  # initial temperature relative to initial max
    if init_temp == 0:
        init_temp = 1.0
    cooling = 0.995
    temperature = init_temp

    best_max = compute_max(routes)
    best_routes = copy_routes(routes)
    lengths = [route_length(r) for r in routes]

    for it in range(max_iter):
        # choose move type
        move_type = random.random()
        if move_type < 0.4:
            ok, new_max, new_routes = relocate(routes, lengths)
        elif move_type < 0.7:
            ok, new_max, new_routes = swap(routes, lengths)
        else:
            ok, new_max, new_routes = two_opt(routes, lengths)

        if ok:
            routes = new_routes
            current_max = compute_max(routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = copy_routes(routes)
        else:
            # keep current solution
            pass

        # cooling
        temperature *= cooling

    # Ensure exactly truck_count routes, each starting/ending at 0, all customers visited once
    return best_routes