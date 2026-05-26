import numpy as np
import random

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def decode(keys, truck_count, customers, dm):
    n_cust = len(customers)
    order = np.argsort(keys)
    routes = [[0, 0] for _ in range(truck_count)]
    for idx in order:
        cust = customers[idx]
        best_truck = None
        best_new_max = float('inf')
        for t in range(truck_count):
            route = routes[t]
            current_dists = [route_distance(routes[tt], dm) for tt in range(truck_count) if tt != t]
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_dist = route_distance(new_route, dm)
                new_max = max(max(current_dists) if current_dists else 0, new_dist)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_truck = t
        route = routes[best_truck]
        best_pos = None
        best_new_dist = float('inf')
        for pos in range(1, len(route)):
            new_route = route[:pos] + [cust] + route[pos:]
            new_dist = route_distance(new_route, dm)
            if new_dist < best_new_dist:
                best_new_dist = new_dist
                best_pos = pos
        if best_pos is None:
            best_pos = len(route)
        routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
    return routes

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    np.random.seed(0)
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    num_cust = n - 1
    pop_size = max(15, num_cust)
    generations = max(10, int(1.5 * num_cust))
    patience = 10
    best_max = float('inf')
    best_routes = None
    pop = [np.random.uniform(size=num_cust) for _ in range(pop_size)]
    last_improvement_gen = -1
    for gen in range(generations):
        fitness = []
        all_routes = []
        for keys in pop:
            routes = decode(keys, truck_count, customers, distance_matrix)
            max_dist = max(route_distance(r, distance_matrix) for r in routes)
            fitness.append(max_dist)
            all_routes.append(routes)
            if max_dist < best_max - 1e-12:
                best_max = max_dist
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
                last_improvement_gen = gen
        if gen - last_improvement_gen >= patience:
            break
        best_idx = int(np.argmin(fitness))
        next_pop = [pop[best_idx]]
        for _ in range(pop_size - 1):
            i1, i2 = np.random.choice(pop_size, 2, replace=False)
            winner = i1 if fitness[i1] < fitness[i2] else i2
            parent1 = pop[winner]
            parent2 = pop[np.random.randint(pop_size)]
            child = np.where(np.random.uniform(size=num_cust) < 0.5, parent1, parent2)
            mut_rate = 1.0 / num_cust
            mut_mask = np.random.uniform(size=num_cust) < mut_rate
            child[mut_mask] = np.random.uniform(size=int(mut_mask.sum()))
            next_pop.append(child)
        pop = next_pop
    if best_routes is None:
        best_routes = decode(pop[0], truck_count, customers, distance_matrix)
        report_best_vrp(best_routes)
    return best_routes