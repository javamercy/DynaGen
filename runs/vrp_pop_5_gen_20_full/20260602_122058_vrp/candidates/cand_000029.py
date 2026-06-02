import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    num_customers = n - 1
    if truck_count < 1:
        return []
    customers = list(range(1, n))
    if truck_count >= num_customers:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes[:truck_count]

    # Farthest-first seeding
    seeds = []
    first_seed = max(range(1, n), key=lambda c: distance_matrix[0][c])
    seeds.append(first_seed)
    while len(seeds) < truck_count:
        max_min_dist = -1
        best_cust = None
        for c in range(1, n):
            if c in seeds:
                continue
            min_dist = min(distance_matrix[c][s] for s in seeds)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_cust = c
            elif min_dist == max_min_dist and best_cust is not None and c < best_cust:
                best_cust = c
        seeds.append(best_cust)
    seeds.sort()

    # Assign customers to nearest seed
    clusters = {i: [] for i in range(truck_count)}
    for c in range(1, n):
        min_dist = float('inf')
        best_idx = None
        for idx, s in enumerate(seeds):
            d = distance_matrix[c][s]
            if d < min_dist:
                min_dist = d
                best_idx = idx
            elif d == min_dist and best_idx is not None and idx < best_idx:
                best_idx = idx
        clusters[best_idx].append(c)

    # Cheapest insertion to build routes
    def route_dist(r):
        d = 0
        for a, b in zip(r, r[1:]):
            d += distance_matrix[a][b]
        return d

    routes = []
    for idx in range(truck_count):
        cluster = clusters[idx]
        if not cluster:
            routes.append([0, 0])
            continue
        route = [0]
        remaining = cluster[:]
        while remaining:
            best_increase = float('inf')
            best_pos = None
            best_cust = None
            for cust in remaining:
                for pos in range(1, len(route)+1):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = 0
                    for a, b in zip(new_route, new_route[1:]):
                        new_dist += distance_matrix[a][b]
                    new_dist += distance_matrix[new_route[-1]][0]
                    original_dist = 0
                    for a, b in zip(route, route[1:]):
                        original_dist += distance_matrix[a][b]
                    original_dist += distance_matrix[route[-1]][0]
                    increase = new_dist - original_dist
                    if increase < best_increase:
                        best_increase = increase
                        best_pos = pos
                        best_cust = cust
                    elif increase == best_increase and best_cust is not None and cust < best_cust:
                        best_pos = pos
                        best_cust = cust
            route = route[:best_pos] + [best_cust] + route[best_pos:]
            remaining.remove(best_cust)
        route.append(0)
        routes.append(route)

    while len(routes) < truck_count:
        routes.append([0, 0])
    routes = routes[:truck_count]

    # Helper to compute max distance
    def max_route_dist(routes):
        return max(route_dist(r) for r in routes)

    report_best_vrp(routes)

    # Simulated annealing parameters
    initial_temp = max_route_dist(routes) * 0.1
    temp = initial_temp
    cooling_rate = 0.95
    max_iter = max(50, num_customers * 3)

    for iteration in range(max_iter):
        current_max = max_route_dist(routes)
        # Identify longest route
        longest_idx = max(range(len(routes)), key=lambda i: route_dist(routes[i]))
        # Choose a random move type: 0=2-opt on longest, 1=relocate from longest, 2=swap from longest
        move_type = np.random.randint(0, 3)
        new_routes = None
        if move_type == 0:
            # 2-opt on longest route
            route = routes[longest_idx]
            if len(route) >= 4:
                i = np.random.randint(1, len(route)-2)
                j = np.random.randint(i+1, len(route)-1)
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_routes = routes[:]
                new_routes[longest_idx] = new_route
        elif move_type == 1:
            # Relocate a customer from longest to another route
            if len(routes[longest_idx]) >= 3:
                pos = np.random.randint(1, len(routes[longest_idx])-1)
                cust = routes[longest_idx][pos]
                new_longest = routes[longest_idx][:pos] + routes[longest_idx][pos+1:]
                other_idx = np.random.choice([i for i in range(len(routes)) if i != longest_idx and len(routes[i]) >= 1])
                other_route = routes[other_idx]
                ins_pos = np.random.randint(1, len(other_route))
                new_other = other_route[:ins_pos] + [cust] + other_route[ins_pos:]
                new_routes = routes[:]
                new_routes[longest_idx] = new_longest
                new_routes[other_idx] = new_other
        else:
            # Swap two customers between longest and another route
            if len(routes[longest_idx]) >= 3 and any(len(r)>=3 for r in routes if r != routes[longest_idx]):
                pos1 = np.random.randint(1, len(routes[longest_idx])-1)
                cust1 = routes[longest_idx][pos1]
                other_idx = np.random.choice([i for i in range(len(routes)) if i != longest_idx and len(routes[i]) >= 3])
                other_route = routes[other_idx]
                pos2 = np.random.randint(1, len(other_route)-1)
                cust2 = other_route[pos2]
                new_longest = routes[longest_idx][:]
                new_longest[pos1] = cust2
                new_other = other_route[:]
                new_other[pos2] = cust1
                new_routes = routes[:]
                new_routes[longest_idx] = new_longest
                new_routes[other_idx] = new_other

        if new_routes is not None:
            new_max = max_route_dist(new_routes)
            delta = new_max - current_max
            if delta < 0:  # improvement
                routes = new_routes
                report_best_vrp(routes)
            else:
                # Accept with probability
                prob = math.exp(-delta / temp)
                if np.random.random() < prob:
                    routes = new_routes
                    report_best_vrp(routes)
        # Cool down
        temp *= cooling_rate
        if temp < 1e-6:
            break

    # Post-processing: simple balance move (deterministic)
    balance_iters = num_customers // truck_count
    for _ in range(balance_iters):
        longest_idx = max(range(len(routes)), key=lambda i: route_dist(routes[i]))
        shortest_idx = min(range(len(routes)), key=lambda i: route_dist(routes[i]))
        if shortest_idx == longest_idx:
            break
        best_max = max_route_dist(routes)
        best_move = None
        longest_route = routes[longest_idx]
        shortest_route = routes[shortest_idx]
        for pos, cust in enumerate(longest_route[1:-1]):
            new_longest = longest_route[:pos+1] + longest_route[pos+2:]
            for ins_pos in range(1, len(shortest_route)):
                new_shortest = shortest_route[:ins_pos] + [cust] + shortest_route[ins_pos:]
                new_max = max(route_dist(new_longest), route_dist(new_shortest))
                if new_max < best_max:
                    best_max = new_max
                    best_move = (pos, ins_pos, cust)
        if best_move is not None:
            pos, ins_pos, cust = best_move
            routes[longest_idx] = longest_route[:pos+1] + longest_route[pos+2:]
            routes[shortest_idx] = shortest_route[:ins_pos] + [cust] + shortest_route[ins_pos:]
            report_best_vrp(routes)

    # Final 2-opt on all routes (greedy improvement)
    for idx in range(len(routes)):
        improved = True
        max_iter_inner = num_customers
        iter_count = 0
        while improved and iter_count < max_iter_inner:
            improved = False
            iter_count += 1
            route = routes[idx]
            best_delta = 0
            best_i = best_j = None
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old1 = distance_matrix[route[i-1]][route[i]]
                    old2 = distance_matrix[route[j]][route[j+1]]
                    new1 = distance_matrix[route[i-1]][route[j]]
                    new2 = distance_matrix[route[i]][route[j+1]]
                    delta = (new1 + new2) - (old1 + old2)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            if best_delta < 0:
                route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                routes[idx] = route
                improved = True
                report_best_vrp(routes)

    return routes