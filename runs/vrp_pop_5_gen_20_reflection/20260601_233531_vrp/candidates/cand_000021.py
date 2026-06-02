import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    
    best_routes = None
    best_max_dist = float('inf')
    
    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    for restart in range(5):  # multi-start restarts
        # Clarke-Wright savings
        savings = []
        for i in range(1, n):
            for j in range(i+1, n):
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                savings.append((s, i, j))
        savings.sort(key=lambda x: (-x[0], x[1], x[2]))
        
        # Shuffle savings to randomize merging order (preserve grouping by equal savings?)
        # Group by savings value for deterministic? Shuffle within groups? For simplicity, shuffle all.
        random.shuffle(savings)
        
        # Initialize routes as single-customer tours
        routes = [[0, c, 0] for c in range(1, n)]
        cust_to_route = {c: idx for idx, c in enumerate(range(1, n))}
        
        # Merge routes using randomized savings
        for s, i, j in savings:
            if cust_to_route[i] == cust_to_route[j]:
                continue
            ri = cust_to_route[i]
            rj = cust_to_route[j]
            route_i = routes[ri]
            route_j = routes[rj]
            if len(route_i) < 3 or len(route_j) < 3:
                continue
            endpoints_i = [route_i[1], route_i[-2]]
            endpoints_j = [route_j[1], route_j[-2]]
            if i not in endpoints_i or j not in endpoints_j:
                continue
            # Orient so that i is at start and j at end
            if route_i[-2] == i:
                route_i[1:-1] = route_i[-2:0:-1]
            if route_j[1] == j:
                route_j[1:-1] = route_j[-2:0:-1]
            if route_i[1] == i and route_j[-2] == j:
                new_route = route_i[:-1] + route_j[1:]
                routes[ri] = new_route
                routes[rj] = [0, 0]
                for c in route_j[1:-1]:
                    cust_to_route[c] = ri
        
        # Clean up routes: remove empty, pad to truck_count
        non_empty = [r for r in routes if len(r) > 2]
        if len(non_empty) > truck_count:
            # Merge extra routes into largest
            non_empty.sort(key=lambda r: -len(r))
            while len(non_empty) > truck_count:
                extra = non_empty.pop()
                target = min(range(len(non_empty)), key=lambda i: len(non_empty[i]))
                for c in extra[1:-1]:
                    non_empty[target].insert(-1, c)
        elif len(non_empty) < truck_count:
            non_empty += [[0, 0] for _ in range(truck_count - len(non_empty))]
        routes = non_empty[:truck_count]
        
        # Intra-route 2-opt
        for idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            improved = True
            max_iters = len(route) * len(route)
            iters = 0
            while improved and iters < max_iters:
                improved = False
                iters += 1
                best_gain = 0
                best_ij = None
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        gain = old - new
                        if gain > best_gain:
                            best_gain = gain
                            best_ij = (i, j)
                if best_gain > 0:
                    i, j = best_ij
                    route[i:j+1] = route[i:j+1][::-1]
                    improved = True
        
        # Inter-route improvement focusing on max route distance
        total_cust = sum(len(r)-2 for r in routes)
        max_iter = 2 * total_cust
        for iteration in range(max_iter):
            dists = [route_distance(r) for r in routes]
            max_dist = max(dists)
            max_idx = dists.index(max_dist)
            best_new_max = max_dist
            best_imp = None
            # Move customer from longest route to another
            max_route = routes[max_idx]
            cust_list = max_route[1:-1]
            for c in cust_list:
                for other_idx in range(len(routes)):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for pos in range(1, len(other_route)):
                        new_max = [x for x in max_route if x != c]
                        new_other = other_route[:pos] + [c] + other_route[pos:]
                        if len(new_other) < 3:
                            new_other = [0, c, 0]
                        new_max_dist = route_distance(new_max)
                        new_other_dist = route_distance(new_other)
                        cand_max = max(new_max_dist, new_other_dist)
                        if cand_max < best_new_max:
                            best_new_max = cand_max
                            best_imp = ('move', c, other_idx, pos, new_max, new_other)
            # Swap customers between max route and others
            for c in cust_list:
                for other_idx in range(len(routes)):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    other_custs = other_route[1:-1]
                    for d in other_custs:
                        new_max = [d if x==c else x for x in max_route]
                        new_other = [c if x==d else x for x in other_route]
                        new_max_dist = route_distance(new_max)
                        new_other_dist = route_distance(new_other)
                        cand_max = max(new_max_dist, new_other_dist)
                        if cand_max < best_new_max:
                            best_new_max = cand_max
                            best_imp = ('swap', c, d, other_idx, new_max, new_other)
            if best_imp is None:
                break
            op = best_imp[0]
            if op == 'move':
                _, c, other_idx, pos, new_max, new_other = best_imp
                routes[max_idx] = new_max
                routes[other_idx] = new_other
            else:
                _, c, d, other_idx, new_max, new_other = best_imp
                routes[max_idx] = new_max
                routes[other_idx] = new_other
            # Optionally call report_best_vrp if improved globally
            dists = [route_distance(r) for r in routes]
            new_max_dist = max(dists)
            if new_max_dist < best_max_dist:
                best_max_dist = new_max_dist
                report_best_vrp(routes)
        
        # Perturbation: move a random customer to another random route if stuck
        if len(routes) > 1 and total_cust > 0:
            for _ in range(2):  # limited perturbation steps
                dists = [route_distance(r) for r in routes]
                current_max = max(dists)
                # Pick a random customer from a random route
                ri = random.randrange(len(routes))
                while len(routes[ri]) <= 2:
                    ri = random.randrange(len(routes))
                route_ri = routes[ri]
                c = random.choice(route_ri[1:-1])
                # Pick another route (different)
                rj = random.randrange(len(routes))
                while rj == ri or len(routes[rj]) < 3:
                    rj = random.randrange(len(routes))
                # Remove c from route_ri
                route_ri.remove(c)
                # Insert into route_rj at random position
                pos = random.randrange(1, len(routes[rj]))
                routes[rj].insert(pos, c)
                # Re-optimize with local search
                for idx, route in enumerate(routes):
                    if len(route) <= 3:
                        continue
                    improved = True
                    max_iters = len(route) * len(route)
                    iters = 0
                    while improved and iters < max_iters:
                        improved = False
                        iters += 1
                        best_gain = 0
                        best_ij = None
                        for i in range(1, len(route)-2):
                            for j in range(i+1, len(route)-1):
                                old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                                new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                                gain = old - new
                                if gain > best_gain:
                                    best_gain = gain
                                    best_ij = (i, j)
                        if best_gain > 0:
                            i, j = best_ij
                            route[i:j+1] = route[i:j+1][::-1]
                            improved = True
                # Update best if improved
                dists = [route_distance(r) for r in routes]
                new_max = max(dists)
                if new_max < best_max_dist:
                    best_max_dist = new_max
                    report_best_vrp(routes)
        
        # Update best overall
        dists = [route_distance(r) for r in routes]
        current_max = max(dists)
        if current_max < best_max_dist:
            best_max_dist = current_max
            best_routes = [r[:] for r in routes]
    
    # Fallback if best_routes not set
    if best_routes is None:
        best_routes = routes
    # Ensure exactly truck_count routes and proper format
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    best_routes = best_routes[:truck_count]
    for r in best_routes:
        if r[0] != 0:
            r.insert(0, 0)
        if r[-1] != 0:
            r.append(0)
    return best_routes