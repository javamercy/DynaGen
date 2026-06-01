import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    def route_length(route):
        if len(route) <= 1:
            return 0.0
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    best_max = float('inf')
    best_routes = None
    
    def update_best(routes):
        nonlocal best_max, best_routes
        m = max(route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
    
    # Regret-based construction
    def construct():
        routes = [[0, 0] for _ in range(truck_count)]
        lens = [0.0] * truck_count
        unassigned = set(customers)
        while unassigned:
            best_cust = None
            best_regret = -float('inf')
            best_first = float('inf')
            best_info = None
            for cust in unassigned:
                first = float('inf')
                second = float('inf')
                best_t = -1
                best_p = -1
                for t in range(truck_count):
                    route = routes[t]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_len = route_length(new_route)
                        other_max = max(lens[:t] + lens[t+1:], default=0.0)
                        new_max = max(new_len, other_max)
                        if new_max < first:
                            second = first
                            first = new_max
                            best_t = t
                            best_p = pos
                        elif new_max < second:
                            second = new_max
                if first < float('inf'):
                    regret = second - first
                    if regret > best_regret or (regret == best_regret and (first < best_first or (first == best_first and cust < best_cust))):
                        best_regret = regret
                        best_first = first
                        best_cust = cust
                        best_info = (best_t, best_p)
            if best_cust is not None:
                t, pos = best_info
                routes[t].insert(pos, best_cust)
                lens[t] = route_length(routes[t])
                unassigned.remove(best_cust)
        return routes, lens
    
    routes, lens = construct()
    update_best(routes)
    
    # Simulated annealing
    initial_temp = best_max * 0.5
    if initial_temp == 0:
        initial_temp = 1.0
    temp = initial_temp
    cooling_rate = 0.999
    max_iter = 2000 * n
    
    for _ in range(max_iter):
        r = random.random()
        old_max = max(lens)
        best_delta_curr = float('inf')
        best_move = None
        
        if r < 0.4:  # relocate
            t1 = random.randrange(truck_count)
            route1 = routes[t1]
            if len(route1) <= 3:
                continue
            i = random.randrange(1, len(route1)-1)
            cust = route1[i]
            t2 = random.randrange(truck_count)
            if t2 == t1:
                continue
            route2 = routes[t2]
            if len(route2) < 2:
                continue
            j = random.randrange(1, len(route2))
            new_route1 = route1[:i] + route1[i+1:]
            new_len1 = route_length(new_route1)
            new_route2 = route2[:j] + [cust] + route2[j:]
            new_len2 = route_length(new_route2)
            new_max = max(new_len1, new_len2, max(lens[:t1] + lens[t1+1:t2] + lens[t2+1:]))
            if new_max < old_max or random.random() < math.exp((old_max - new_max) / temp):
                routes[t1] = new_route1
                routes[t2] = new_route2
                lens[t1] = new_len1
                lens[t2] = new_len2
                if max(lens) < best_max:
                    update_best(routes)
        elif r < 0.7:  # swap
            t1 = random.randrange(truck_count)
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            i = random.randrange(1, len(route1)-1)
            cust1 = route1[i]
            t2 = random.randrange(truck_count)
            if t2 == t1:
                continue
            route2 = routes[t2]
            if len(route2) <= 2:
                continue
            j = random.randrange(1, len(route2)-1)
            cust2 = route2[j]
            new_route1 = route1[:i] + [cust2] + route1[i+1:]
            new_len1 = route_length(new_route1)
            new_route2 = route2[:j] + [cust1] + route2[j+1:]
            new_len2 = route_length(new_route2)
            new_max = max(new_len1, new_len2, max(lens[:t1] + lens[t1+1:t2] + lens[t2+1:]))
            if new_max < old_max or random.random() < math.exp((old_max - new_max) / temp):
                routes[t1] = new_route1
                routes[t2] = new_route2
                lens[t1] = new_len1
                lens[t2] = new_len2
                if max(lens) < best_max:
                    update_best(routes)
        else:  # 2-opt
            t = random.randrange(truck_count)
            route = routes[t]
            if len(route) <= 3:
                continue
            i = random.randrange(1, len(route)-2)
            j = random.randrange(i+1, len(route)-1)
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            new_len = route_length(new_route)
            new_max = max(new_len, max(lens[:t] + lens[t+1:]))
            if new_max < old_max or random.random() < math.exp((old_max - new_max) / temp):
                routes[t] = new_route
                lens[t] = new_len
                if max(lens) < best_max:
                    update_best(routes)
        temp *= cooling_rate
    
    return best_routes