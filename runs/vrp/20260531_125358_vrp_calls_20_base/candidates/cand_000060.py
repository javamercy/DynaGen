import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def decode_greedy(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in perm:
            best_max = float('inf')
            best_truck = -1
            best_pos = -1
            for t in range(truck_count):
                route = routes[t]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_len = route_length(new_route)
                    new_lengths = [route_length(r) for r in routes]
                    new_lengths[t] = new_len
                    new_max = max(new_lengths)
                    if new_max < best_max or (new_max == best_max and t < best_truck):
                        best_max = new_max
                        best_truck = t
                        best_pos = pos
            routes[best_truck] = routes[best_truck][:best_pos] + [cust] + routes[best_truck][best_pos:]
        return routes

    def regret_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        unvisited = set(customers)
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route_idx = -1
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
                    best_route_idx = r_idx
                    best_pos = pos
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        perm = []
        for r in routes:
            perm.extend(r[1:-1])
        return perm, routes

    # Initial population: random keys
    pop_size = 20
    pop = []
    for _ in range(pop_size - 1):
        keys = [random.random() for _ in range(n-1)]
        perm = [customers[i] for i in np.argsort(keys)]
        routes = decode_greedy(perm)
        fit = max(route_length(r) for r in routes)
        pop.append((fit, keys, routes))

    # Add regret-3 solution
    perm_reg, routes_reg = regret_construction()
    fit_reg = max(route_length(r) for r in routes_reg)
    order_map = {cust: i for i, cust in enumerate(perm_reg)}
    keys_reg = [order_map[c] / (n-1) for c in customers]
    pop.append((fit_reg, keys_reg, routes_reg))

    pop.sort(key=lambda x: x[0])
    best_routes = pop[0][2][:]
    best_max = pop[0][0]
    report_best_vrp(best_routes)

    def tournament_select(pop, k):
        selected = random.sample(pop, min(k, len(pop)))
        selected.sort(key=lambda x: x[0])
        return selected[0]

    max_gen = max(100, n * truck_count * 2)
    for gen in range(max_gen):
        parent1 = tournament_select(pop, 3)
        parent2 = tournament_select(pop, 3)
        # BLX-alpha crossover
        alpha = 0.5
        child_keys = []
        for i in range(len(keys_reg)):
            min_k = min(parent1[1][i], parent2[1][i])
            max_k = max(parent1[1][i], parent2[1][i])
            delta = max_k - min_k
            new_k = random.uniform(min_k - alpha*delta, max_k + alpha*delta)
            new_k = max(0.0, min(1.0, new_k))
            child_keys.append(new_k)
        # Mutation
        mut_std = 0.1
        for i in range(len(child_keys)):
            child_keys[i] += random.gauss(0, mut_std)
            child_keys[i] = max(0.0, min(1.0, child_keys[i]))
        # Decode
        perm_child = [customers[i] for i in np.argsort(child_keys)]
        routes_child = decode_greedy(perm_child)
        fit_child = max(route_length(r) for r in routes_child)
        # Replace worst if better
        if fit_child < pop[-1][0]:
            pop.pop()
            pop.append((fit_child, child_keys, routes_child))
            pop.sort(key=lambda x: x[0])
            if fit_child < best_max:
                best_max = fit_child
                best_routes = routes_child[:]
                report_best_vrp(best_routes)
    return best_routes