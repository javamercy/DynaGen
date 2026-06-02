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

    # ---- Clarke-Wright Savings Initialization ----
    # Start with each customer in its own route
    routes = [[0, c, 0] for c in customers]
    # Compute savings matrix
    savings = {}
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            savings[(i, j)] = s
    # Sort savings descending
    sorted_savings = sorted(savings.items(), key=lambda x: -x[1])
    # Merge routes until we have exactly truck_count routes
    while len(routes) > truck_count:
        # Find the best feasible merge
        best_merge = None
        best_saving = -float('inf')
        for (i, j), s in sorted_savings:
            # Find routes containing i and j
            route_i_idx = next((idx for idx, r in enumerate(routes) if i in r[1:-1]), None)
            route_j_idx = next((idx for idx, r in enumerate(routes) if j in r[1:-1]), None)
            if route_i_idx is None or route_j_idx is None:
                continue
            if route_i_idx == route_j_idx:
                continue
            route_i = routes[route_i_idx]
            route_j = routes[route_j_idx]
            # Check if i and j are endpoints (adjacent to depot) in their routes
            # For savings merge, we need i at end of route_i (last node before depot) and j at start of route_j (first node after depot) or vice versa.
            # Since routes are depot-c...-depot, we check if i is the last customer and j is the first customer
            # Actually, we can merge if i is at the end of route_i (position -2) and j is at the start of route_j (position 1)
            # Or i at start of route_i and j at end of route_j.
            # For simplicity, allow any order: we can reverse routes if needed.
            # But we want to keep deterministic tie-breaking.
            # We'll merge by adding the customers of route_j into route_i at appropriate position.
            if route_i[-2] == i and route_j[1] == j:
                # Merge: route_i + route_j (skip depot of route_j)
                new_route = route_i[:-1] + route_j[1:]
            elif route_i[1] == i and route_j[-2] == j:
                # Merge: route_j + route_i (skip depot of route_i)
                new_route = route_j[:-1] + route_i[1:]
            else:
                # Try if reversal helps
                if route_i[-2] == i and route_j[-2] == j:
                    # Reverse route_j so that j becomes first customer
                    rev_j = [0] + route_j[1:-1][::-1] + [0]
                    new_route = route_i[:-1] + rev_j[1:]
                elif route_i[1] == i and route_j[1] == j:
                    # Reverse route_i so that i becomes last customer
                    rev_i = [0] + route_i[1:-1][::-1] + [0]
                    new_route = rev_i[:-1] + route_j[1:]
                else:
                    continue
            # Feasible merge
            new_routes = [routes[idx] for idx in range(len(routes)) if idx != route_i_idx and idx != route_j_idx]
            new_routes.append(new_route)
            # Ensure we have correct number of routes after merge
            if len(new_routes) == truck_count:
                # Apply this merge and break
                routes = new_routes
                break
        else:
            # No feasible merge found, fallback: just truncate to truck_count by concatenating extra routes
            break
    # If still too many routes, concatenate last ones arbitrarily
    while len(routes) > truck_count:
        # Merge the last two routes (simple concatenation)
        route1 = routes.pop()
        route2 = routes.pop()
        new_route = route1[:-1] + route2[1:]
        routes.append(new_route)
    # If less than truck_count, add empty routes
    while len(routes) < truck_count:
        routes.append([0, 0])
    routes = routes[:truck_count]

    # ---- Helper functions (same as parent) ----
    def route_dist(r):
        d = 0
        for a, b in zip(r, r[1:]):
            d += distance_matrix[a][b]
        return d

    def max_route_dist(routes):
        return max(route_dist(r) for r in routes)

    report_best_vrp(routes)

    # ---- Simulated Annealing (same as parent) ----
    initial_temp = max_route_dist(routes) * 0.1
    temp = initial_temp
    cooling_rate = 0.95
    max_iter = max(50, num_customers * 3)

    for iteration in range(max_iter):
        current_max = max_route_dist(routes)
        longest_idx = max(range(len(routes)), key=lambda i: route_dist(routes[i]))
        move_type = np.random.randint(0, 3)
        new_routes = None
        if move_type == 0:
            route = routes[longest_idx]
            if len(route) >= 4:
                i = np.random.randint(1, len(route)-2)
                j = np.random.randint(i+1, len(route)-1)
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_routes = routes[:]
                new_routes[longest_idx] = new_route
        elif move_type == 1:
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
            if len(routes[longest_idx]) >= 3 and any(len(r) >= 3 for r in routes if r != routes[longest_idx]):
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
            if delta < 0:
                routes = new_routes
                report_best_vrp(routes)
            else:
                prob = math.exp(-delta / temp)
                if np.random.random() < prob:
                    routes = new_routes
                    report_best_vrp(routes)
        temp *= cooling_rate
        if temp < 1e-6:
            break

    # ---- Post-processing (same as parent) ----
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

    # ---- Final 2-opt (same as parent) ----
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