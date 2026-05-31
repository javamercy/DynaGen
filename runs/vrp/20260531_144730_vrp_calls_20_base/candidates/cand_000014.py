import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    best_routes = None
    best_max = float('inf')
    
    # Outer loop for restarts (small number for bounded runtime)
    for restart in range(max(1, n // 10 + 1)):
        # Initialize routes
        routes = [[0, 0] for _ in range(truck_count)]
        unrouted = list(range(1, n))
        
        # Stochastic cheapest insertion with temperature
        temp = 100.0
        decay = 0.99
        while unrouted and temp > 1e-3:
            best_candidates = []
            for cust in unrouted:
                for ridx, route in enumerate(routes):
                    # compute insertion cost at each position
                    for pos in range(1, len(route)):
                        delta = (distance_matrix[route[pos-1], cust] +
                                 distance_matrix[cust, route[pos]] -
                                 distance_matrix[route[pos-1], route[pos]])
                        if not best_candidates or delta < best_candidates[-1][0]:
                            best_candidates.append((delta, cust, ridx, pos))
            if not best_candidates:
                break
            # Sort by delta
            best_candidates.sort(key=lambda x: x[0])
            # Keep only the best few (e.g., 10) for probability selection
            k = min(10, len(best_candidates))
            top = best_candidates[:k]
            # Compute softmax probabilities with temperature
            min_delta = top[0][0]
            weights = [math.exp(-(d - min_delta) / temp) for d, _, _, _ in top]
            total = sum(weights)
            probs = [w/total for w in weights]
            # Roulette wheel selection
            r = random.random()
            cum = 0.0
            for idx, prob in enumerate(probs):
                cum += prob
                if r <= cum:
                    _, cust, ridx, pos = top[idx]
                    break
            # Insert chosen customer
            routes[ridx].insert(pos, cust)
            unrouted.remove(cust)
            temp *= decay
        # Force insert remaining customers (if any) with cheapest deterministic
        while unrouted:
            best_cust = None
            best_ridx = None
            best_pos = None
            best_delta = float('inf')
            for cust in unrouted:
                for ridx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        delta = (distance_matrix[route[pos-1], cust] +
                                 distance_matrix[cust, route[pos]] -
                                 distance_matrix[route[pos-1], route[pos]])
                        if delta < best_delta:
                            best_delta = delta
                            best_cust = cust
                            best_ridx = ridx
                            best_pos = pos
            routes[best_ridx].insert(best_pos, best_cust)
            unrouted.remove(best_cust)
        
        # Compute initial max
        def route_dist(route):
            d = 0.0
            for i in range(len(route)-1):
                d += distance_matrix[route[i], route[i+1]]
            return d
        max_dist = max(route_dist(r) for r in routes)
        if max_dist < best_max:
            best_max = max_dist
            best_routes = [r[:] for r in routes]
        # report initial
        try:
            import builtins
            if hasattr(builtins, 'report_best_vrp'):
                report_best_vrp(best_routes)
        except:
            pass
        
        # Local search: try to reduce max by moving a customer from longest route to another
        max_iter = n * truck_count
        for _ in range(max_iter):
            # find longest route
            max_dist = 0
            max_ridx = -1
            for idx, r in enumerate(routes):
                d = route_dist(r)
                if d > max_dist:
                    max_dist = d
                    max_ridx = idx
            improved = False
            longest = routes[max_ridx]
            # consider each customer in longest (skip depots)
            for cust_idx in range(1, len(longest)-1):
                cust = longest[cust_idx]
                # removal delta
                prev = longest[cust_idx-1]
                next_ = longest[cust_idx+1]
                removal_delta = distance_matrix[prev, next_] - distance_matrix[prev, cust] - distance_matrix[cust, next_]
                new_longest = longest[:cust_idx] + longest[cust_idx+1:]
                new_longest_dist = max_dist + removal_delta
                # try inserting in other routes
                for other_ridx in range(truck_count):
                    if other_ridx == max_ridx:
                        continue
                    other = routes[other_ridx]
                    # find best insertion position in other
                    best_delta = float('inf')
                    best_pos = None
                    for pos in range(1, len(other)):
                        delta = (distance_matrix[other[pos-1], cust] +
                                 distance_matrix[cust, other[pos]] -
                                 distance_matrix[other[pos-1], other[pos]])
                        if delta < best_delta:
                            best_delta = delta
                            best_pos = pos
                    if best_pos is None:
                        continue
                    new_other = other[:best_pos] + [cust] + other[best_pos:]
                    new_other_dist = route_dist(other) + best_delta
                    potential_max = max(new_longest_dist, new_other_dist)
                    for r_idx, r in enumerate(routes):
                        if r_idx != max_ridx and r_idx != other_ridx:
                            potential_max = max(potential_max, route_dist(r))
                    if potential_max < best_max - 1e-12:
                        # accept move
                        routes[max_ridx] = new_longest
                        routes[other_ridx] = new_other
                        best_max = potential_max
                        improved = True
                        try:
                            if hasattr(builtins, 'report_best_vrp'):
                                report_best_vrp(routes)
                        except:
                            pass
                        break
                if improved:
                    break
            if not improved:
                break
        
        # Update best after local search
        current_max = max(route_dist(r) for r in routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            try:
                if hasattr(builtins, 'report_best_vrp'):
                    report_best_vrp(best_routes)
            except:
                pass
        
        # If restart is not the last, apply shake (perturbation) to escape local optima
        if restart < max(1, n // 10 + 1) - 1:
            # remove up to 5 customers (or 10% of n) at random
            num_remove = min(5, max(1, n // 10))
            # collect all customers currently in routes
            all_custs = []
            for r in routes:
                for c in r:
                    if c != 0:
                        all_custs.append(c)
            # shuffle and pick first num_remove
            random.shuffle(all_custs)
            remove_set = set(all_custs[:num_remove])
            # remove them from routes
            new_unrouted = []
            for i, r in enumerate(routes):
                new_route = [0]
                for c in r[1:-1]:
                    if c in remove_set:
                        new_unrouted.append(c)
                    else:
                        new_route.append(c)
                new_route.append(0)
                routes[i] = new_route
            # Reinsert removed customers using stochastic insertion (resetting temperature)
            temp = 100.0
            while new_unrouted and temp > 1e-3:
                best_candidates = []
                for cust in new_unrouted:
                    for ridx, route in enumerate(routes):
                        for pos in range(1, len(route)):
                            delta = (distance_matrix[route[pos-1], cust] +
                                     distance_matrix[cust, route[pos]] -
                                     distance_matrix[route[pos-1], route[pos]])
                            best_candidates.append((delta, cust, ridx, pos))
                if not best_candidates:
                    break
                best_candidates.sort(key=lambda x: x[0])
                k = min(10, len(best_candidates))
                top = best_candidates[:k]
                min_delta = top[0][0]
                weights = [math.exp(-(d - min_delta) / temp) for d, _, _, _ in top]
                total = sum(weights)
                probs = [w/total for w in weights]
                r = random.random()
                cum = 0.0
                for idx, prob in enumerate(probs):
                    cum += prob
                    if r <= cum:
                        _, cust, ridx, pos = top[idx]
                        break
                routes[ridx].insert(pos, cust)
                new_unrouted.remove(cust)
                temp *= decay
            # Force insert remaining
            while new_unrouted:
                best_cust = None
                best_ridx = None
                best_pos = None
                best_delta = float('inf')
                for cust in new_unrouted:
                    for ridx, route in enumerate(routes):
                        for pos in range(1, len(route)):
                            delta = (distance_matrix[route[pos-1], cust] +
                                     distance_matrix[cust, route[pos]] -
                                     distance_matrix[route[pos-1], route[pos]])
                            if delta < best_delta:
                                best_delta = delta
                                best_cust = cust
                                best_ridx = ridx
                                best_pos = pos
                routes[best_ridx].insert(best_pos, best_cust)
                new_unrouted.remove(best_cust)
            # Update best after shake
            current_max = max(route_dist(r) for r in routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                try:
                    if hasattr(builtins, 'report_best_vrp'):
                        report_best_vrp(best_routes)
                except:
                    pass
    
    return best_routes