import numpy as np
import random

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
    
    def copy_routes(routes):
        return [r[:] for r in routes]
    
    best_routes = None
    best_max = float('inf')
    max_restarts = max(1, n // 5)
    max_iter_local = n * truck_count
    
    for restart in range(max_restarts):
        # Construction: greedy min-max insertion (no regret)
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        random.shuffle(unassigned)
        for cust in unassigned:
            best_insert = None
            best_new_max = float('inf')
            best_cost = float('inf')
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_len = route_length(route) + cost
                    other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                    new_max = max(new_len, *other_lens)
                    if (new_max < best_new_max) or (new_max == best_new_max and cost < best_cost):
                        best_new_max = new_max
                        best_cost = cost
                        best_insert = (r_idx, pos)
            if best_insert is None:
                # fallback: insert into first route at end
                best_insert = (0, len(routes[0])-1)
            r_idx, pos = best_insert
            routes[r_idx].insert(pos, cust)
        
        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = copy_routes(routes)
            report_best_vrp(routes)
        
        # Local search (improve until no improvement or max iterations)
        for iter_count in range(max_iter_local):
            improved = False
            # Inter-relocate from longest route
            lengths = [route_length(r) for r in routes]
            max_idx = int(np.argmax(lengths))
            max_route = routes[max_idx]
            if len(max_route) > 2:
                best_delta = 0.0
                best_move = None
                for cust in max_route[1:-1]:
                    new_max_route = [x for x in max_route if x != cust]
                    new_max_len = route_length(new_max_route)
                    for r_idx in range(truck_count):
                        if r_idx == max_idx:
                            continue
                        other_route = routes[r_idx]
                        for pos in range(1, len(other_route)):
                            new_other = other_route[:pos] + [cust] + other_route[pos:]
                            new_other_len = route_length(new_other)
                            other_lens = [lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)]
                            new_max_candidate = max(new_max_len, new_other_len, *other_lens)
                            if new_max_candidate < current_max - 1e-12:
                                delta = current_max - new_max_candidate
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = (cust, max_idx, r_idx, pos, new_max_candidate)
                if best_move:
                    cust, from_idx, to_idx, pos, new_max_val = best_move
                    routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                    routes[to_idx].insert(pos, cust)
                    current_max = new_max_val
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = copy_routes(routes)
                        report_best_vrp(routes)
                    improved = True
            # Intra-2opt on each route (only longest routes can improve max)
            if not improved:
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 3:
                        continue
                    best_delta = 0.0
                    best_ij = None
                    best_new_len = None
                    for i in range(1, len(route)-2):
                        for k in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                            new_len = route_length(new_route)
                            if new_len < route_length(route) - 1e-12:
                                delta = route_length(route) - new_len
                                if delta > best_delta:
                                    best_delta = delta
                                    best_ij = (i, k, r_idx)
                                    best_new_len = new_len
                    if best_ij:
                        i, k, r_idx = best_ij
                        routes[r_idx] = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        new_max = max_route_len(routes)
                        if new_max < current_max:
                            current_max = new_max
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = copy_routes(routes)
                                report_best_vrp(routes)
                        improved = True
                        break  # only one improvement per iteration? Actually we want multiple, but let's break to avoid bias; we can continue but simpler: break to re-evaluate longest route
                if improved:
                    continue
            if not improved:
                break  # local optimum
        
        # Perturbation: remove some customers from longest routes and reinsert greedily
        routes = copy_routes(best_routes)  # restart from best
        remove_count = max(1, (n-1) // 10)
        lengths = [(route_length(r), idx) for idx, r in enumerate(routes)]
        lengths.sort(reverse=True)
        removed = []
        for _, r_idx in lengths:
            route = routes[r_idx]
            if len(route) <= 2:
                continue
            can_remove = min(remove_count - len(removed), len(route)-2)
            if can_remove <= 0:
                break
            remove_set = random.sample(route[1:-1], can_remove)
            for cust in remove_set:
                removed.append(cust)
            routes[r_idx] = [x for x in route if x not in remove_set]
        random.shuffle(removed)
        # Reinsert greedily (same as construction)
        for cust in removed:
            best_insert = None
            best_new_max = float('inf')
            best_cost = float('inf')
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_len = route_length(route) + cost
                    other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                    new_max = max(new_len, *other_lens)
                    if (new_max < best_new_max) or (new_max == best_new_max and cost < best_cost):
                        best_new_max = new_max
                        best_cost = cost
                        best_insert = (r_idx, pos)
            if best_insert is None:
                best_insert = (0, len(routes[0])-1)
            r_idx, pos = best_insert
            routes[r_idx].insert(pos, cust)
        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = copy_routes(routes)
            report_best_vrp(routes)
        # After perturbation, local search again? But we already have a loop over restarts, so we'll just continue to next restart
    
    if best_routes is None:
        best_routes = routes
    return best_routes