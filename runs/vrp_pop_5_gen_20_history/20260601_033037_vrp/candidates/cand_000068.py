import numpy as np
import heapq
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    best_routes = None
    best_max = float('inf')

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    # Initial savings list
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                savings.append((-s, i, j))
    heapq.heapify(savings)

    route_list = [[0, c, 0] for c in customers]
    cust_to_route = {c: idx for idx, c in enumerate(customers)}
    endpoints = [(c, c) for c in customers]

    while len(route_list) > truck_count and savings:
        neg_s, i, j = heapq.heappop(savings)
        if i not in cust_to_route or j not in cust_to_route:
            continue
        ri = cust_to_route[i]
        rj = cust_to_route[j]
        if ri == rj:
            continue
        first_i, last_i = endpoints[ri]
        first_j, last_j = endpoints[rj]
        merged = None
        if i == last_i and j == first_j:
            merged = route_list[ri][:-1] + route_list[rj][1:]
            new_first = first_i
            new_last = last_j
        elif j == last_j and i == first_i:
            merged = route_list[rj][:-1] + route_list[ri][1:]
            new_first = first_j
            new_last = last_i
        elif i == first_i and j == last_j:
            merged = route_list[rj][:-1] + route_list[ri][1:]
            new_first = first_j
            new_last = last_i
        elif j == first_j and i == last_i:
            merged = route_list[ri][:-1] + route_list[rj][1:]
            new_first = first_i
            new_last = last_j
        else:
            continue
        new_route_list = [r for idx, r in enumerate(route_list) if idx not in (ri, rj)]
        new_route_list.append(merged)
        route_list = new_route_list
        cust_to_route.clear()
        endpoints.clear()
        for idx, r in enumerate(route_list):
            interior = r[1:-1]
            for c in interior:
                cust_to_route[c] = idx
            first_c = interior[0] if interior else None
            last_c = interior[-1] if interior else None
            endpoints.append((first_c, last_c))

    while len(route_list) > truck_count:
        dists = [(route_distance(r), idx) for idx, r in enumerate(route_list)]
        dists.sort(key=lambda x: (x[0], x[1]))
        idx1 = dists[0][1]
        idx2 = dists[1][1]
        r1 = route_list[idx1]
        r2 = route_list[idx2]
        merged = r1[:-1] + r2[1:]
        new_route_list = [r for i, r in enumerate(route_list) if i not in (idx1, idx2)]
        new_route_list.append(merged)
        route_list = new_route_list

    report_best_vrp(route_list)

    # Simulated annealing improvement
    max_iter = min(500, n * truck_count * 2)
    current_routes = [list(r) for r in route_list]
    current_max = max(route_distance(r) for r in current_routes)
    temperature = current_max * 0.5
    cooling_rate = 0.99
    for _ in range(max_iter):
        if temperature < 1e-12:
            break
        # Random move type selection
        move_type = random.choice(['relocate', 'swap', '2opt'])
        new_routes = [list(r) for r in current_routes]
        feasible = False
        if move_type == 'relocate':
            # Select random customer from a route with more than 1 customer
            non_empty = [i for i, r in enumerate(new_routes) if len(r) > 3]
            if len(non_empty) < 2:
                continue
            src = random.choice(non_empty)
            src_interior = new_routes[src][1:-1]
            if not src_interior:
                continue
            cust = random.choice(src_interior)
            # Remove cust from source
            new_routes[src].remove(cust)
            # If source becomes [0,0], keep it
            # Insert into another route at random position
            dst = random.choice([i for i in range(truck_count) if i != src])
            if len(new_routes[dst]) == 2:
                pos = 1
            else:
                pos = random.randint(1, len(new_routes[dst]) - 2)
            new_routes[dst].insert(pos, cust)
            feasible = True
        elif move_type == 'swap':
            # Swap two customers from different routes with interior customers
            routes_with_interior = [i for i, r in enumerate(new_routes) if len(r) > 3]
            if len(routes_with_interior) < 2:
                continue
            r1 = random.choice(routes_with_interior)
            r2 = random.choice([i for i in routes_with_interior if i != r1])
            c1 = random.choice(new_routes[r1][1:-1])
            c2 = random.choice(new_routes[r2][1:-1])
            idx1 = new_routes[r1].index(c1)
            idx2 = new_routes[r2].index(c2)
            new_routes[r1][idx1] = c2
            new_routes[r2][idx2] = c1
            feasible = True
        elif move_type == '2opt':
            # 2-opt on a random route
            eligible = [i for i, r in enumerate(new_routes) if len(r) > 4]
            if not eligible:
                continue
            ri = random.choice(eligible)
            route = new_routes[ri]
            if len(route) <= 4:
                continue
            a = random.randint(1, len(route) - 3)
            b = random.randint(a + 1, len(route) - 2)
            new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
            new_routes[ri] = new_route
            feasible = True
        if not feasible:
            continue
        new_max = max(route_distance(r) for r in new_routes)
        delta = new_max - current_max
        if delta < 0 or random.random() < np.exp(-delta / temperature):
            current_routes = new_routes
            current_max = new_max
            if new_max < best_max - 1e-12:
                report_best_vrp(current_routes)
        temperature *= cooling_rate

    return best_routes if best_routes is not None else current_routes