import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    def compute_route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def nearest_neighbor_tsp(nodes):
        if not nodes:
            return [0, 0]
        unvisited = set(nodes)
        current = 0
        route = [0]
        while unvisited:
            next_node = min(unvisited, key=lambda x: distance_matrix[current, x])
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        route.append(0)
        return route
    
    def two_opt_improve(route, max_iter=10):
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    old_len = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new_len = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    if new_len < old_len:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
            if improved:
                break
        return route
    
    def build_route_from_assignment(assignment):
        routes_with_depot = []
        lengths = []
        for truck in assignment:
            if truck:
                route = nearest_neighbor_tsp(truck)
                route = two_opt_improve(route, 10)
            else:
                route = [0, 0]
            routes_with_depot.append(route)
            lengths.append(compute_route_length(route))
        return routes_with_depot, lengths
    
    # Initialize random assignment
    assignment = [[] for _ in range(truck_count)]
    for cust in customers:
        truck = random.randrange(truck_count)
        assignment[truck].append(cust)
    
    current_routes, current_lengths = build_route_from_assignment(assignment)
    current_max = max(current_lengths)
    best_max = current_max
    best_routes = [list(r) for r in current_routes]
    best_assignment = [list(a) for a in assignment]
    
    def report_best_vrp(routes):
        nonlocal best_max, best_routes, best_assignment
        m = max(compute_route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]
            # update best_assignment from routes (extract customers except depot)
            best_assignment = [[c for c in r if c != 0] for r in routes]
    
    report_best_vrp(current_routes)
    
    # Precompute all pairs distances for fast evaluation
    dist = distance_matrix
    
    def evaluate_relocate(t1, idx1, t2):
        # move customer from t1 at idx1 to t2, insert at best position
        cust = current_assign[t1][idx1]
        # remove from t1
        new_assign_t1 = current_assign[t1][:idx1] + current_assign[t1][idx1+1:]
        # insert into t2 at best position (based on existing route? Instead we evaluate using current route? For speed, we'll use a temporary route approach: compute insertion cost using current route of t2)
        # Actually we don't have current routes easily; we only have assignment. So we'll rebuild routes for evaluation? Too heavy.
        # Better: we maintain current_routes and current_lengths for the current solution. For evaluation, we can compute the new route lengths by simulating insertion/removal on current routes.
        # But we need the customer's position in the current route of t1. We have that via idx1 (position in assignment order? No, route order may differ from assignment order because nearest-neighbor reorders.
        # To simplify, we'll work directly with routes. But that means we need to know customer position in route.
        # Alternative: use assignment directly and rebuild routes each time? That would be too slow.
        # For the shake of simplicity and to ensure correctness, we'll implement local search by evaluating all moves using the current routes (tours) directly, not the assignment.
        # So we need a representation that stores routes (list of nodes including depot). The assignment is implicit in the routes.
        # Let's redesign: Represent solution as a list of routes (with depot). This is more convenient for local search moves.
        # We'll convert back to assignment for shaking? Actually we can keep routes directly.
        # So we'll store current_routes (list of lists) and current_lengths.
        # Moves: relocate (remove customer from one route, insert into another), swap (swap two customers).
        # This is more straightforward and similar to local search in parents, but we'll use VNS.
        # To avoid duplication with SA, we'll still do VNS with deterministic local search.
        pass
    
    # We'll rewrite with routes representation.
    # Let's start over with route-based VNS.
    # But the requirement says 'different algorithmic family'. Route-based VNS is still local search on routes, similar to SA but with different acceptance. However, VNS uses deterministic descent and systematic neighborhood change, which is distinct.
    # I'll implement route-based VNS.
    
    # To keep the code concise, I'll implement directly the route-based VNS without precomputing assignment.
    
    # Initialize routes via greedy min-max insertion? That would be similar to parents. Instead, random initialization.
    # Random initialization: assign each customer to a random truck, then build routes with nearest neighbor + 2opt.
    
    import itertools
    
    def compute_route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def nearest_neighbor_tsp(nodes):
        if not nodes:
            return [0, 0]
        unvisited = set(nodes)
        current = 0
        route = [0]
        while unvisited:
            next_node = min(unvisited, key=lambda x: distance_matrix[current, x])
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        route.append(0)
        return route
    
    def two_opt_improve(route, max_iter=10):
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    if distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]] > distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        break
                if improved:
                    break
        return route
    
    # Initial random assignment and build routes
    assign = [[] for _ in range(truck_count)]
    for cust in customers:
        assign[random.randrange(truck_count)].append(cust)
    
    routes = []
    lengths = []
    for truck in assign:
        route = nearest_neighbor_tsp(truck)
        route = two_opt_improve(route, 10)
        routes.append(route)
        lengths.append(compute_route_length(route))
    
    best_max = max(lengths)
    best_routes = [list(r) for r in routes]
    
    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(compute_route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]
    
    report_best_vrp(routes)
    
    # Helper functions for move evaluation
    def eval_relocate(t1, i, t2):
        # t1: source route index, i: position of customer in route t1 (must be in 1..len(routes[t1])-2)
        # t2: destination route index
        if t1 == t2:
            return None
        if len(routes[t1]) <= 2:
            return None
        cust = routes[t1][i]
        # Remove customer from t1
        new_route1 = routes[t1][:i] + routes[t1][i+1:]
        # Find best insertion position in t2
        best_inc = float('inf')
        best_pos = -1
        for j in range(1, len(routes[t2])):
            prev = routes[t2][j-1]
            nxt = routes[t2][j]
            inc = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
            if inc < best_inc:
                best_inc = inc
                best_pos = j
        new_route2 = routes[t2][:best_pos] + [cust] + routes[t2][best_pos:]
        new_len1 = compute_route_length(new_route1)
        new_len2 = compute_route_length(new_route2)
        new_max = new_len1
        for k in range(truck_count):
            if k == t1:
                new_max = max(new_max, new_len1)
            elif k == t2:
                new_max = max(new_max, new_len2)
            else:
                new_max = max(new_max, lengths[k])
        total = sum(lengths) - lengths[t1] - lengths[t2] + new_len1 + new_len2
        return (new_max, total, t1, i, t2, best_pos)
    
    def eval_swap(t1, i, t2, j):
        if t1 == t2:
            return None
        if len(routes[t1]) <= 2 or len(routes[t2]) <= 2:
            return None
        cust1 = routes[t1][i]
        cust2 = routes[t2][j]
        # Swap customers
        new_route1 = routes[t1][:i] + [cust2] + routes[t1][i+1:]
        new_route2 = routes[t2][:j] + [cust1] + routes[t2][j+1:]
        new_len1 = compute_route_length(new_route1)
        new_len2 = compute_route_length(new_route2)
        new_max = new_len1
        for k in range(truck_count):
            if k == t1:
                new_max = max(new_max, new_len1)
            elif k == t2:
                new_max = max(new_max, new_len2)
            else:
                new_max = max(new_max, lengths[k])
        total = sum(lengths) - lengths[t1] - lengths[t2] + new_len1 + new_len2
        return (new_max, total, t1, i, t2, j)
    
    # Local search (steepest descent) considering relocate and swap
    def local_search():
        nonlocal routes, lengths
        improved = True
        while improved:
            improved = False
            best_move = None
            best_delta = (float('inf'), float('inf'), -1)
            # Relocate moves
            for t1 in range(truck_count):
                if len(routes[t1]) <= 2:
                    continue
                for i in range(1, len(routes[t1])-1):
                    for t2 in range(truck_count):
                        if t1 == t2:
                            continue
                        move = eval_relocate(t1, i, t2)
                        if move is None:
                            continue
                        new_max, total, _, _, _, _ = move
                        if new_max < best_delta[0] or (new_max == best_delta[0] and total < best_delta[1]) or (new_max == best_delta[0] and total == best_delta[1] and (t1 < best_delta[2][0] or (t1 == best_delta[2][0] and i < best_delta[2][1]))):
                            best_delta = (new_max, total, (t1, i, t2, move[5]))
                            best_move = ('relocate', t1, i, t2, move[5])
            # Swap moves
            for t1 in range(truck_count):
                if len(routes[t1]) <= 2:
                    continue
                for i in range(1, len(routes[t1])-1):
                    for t2 in range(t1+1, truck_count):
                        if len(routes[t2]) <= 2:
                            continue
                        for j in range(1, len(routes[t2])-1):
                            move = eval_swap(t1, i, t2, j)
                            if move is None:
                                continue
                            new_max, total, _, _, _, _ = move
                            if new_max < best_delta[0] or (new_max == best_delta[0] and total < best_delta[1]) or (new_max == best_delta[0] and total == best_delta[1] and (t1 < best_delta[2][0] or (t1 == best_delta[2][0] and i < best_delta[2][1]))):
                                best_delta = (new_max, total, (t1, i, t2, j))
                                best_move = ('swap', t1, i, t2, j)
            if best_move is not None and best_delta[0] < max(lengths):
                # Consider only strictly improving? For descent we accept only improvement.
                # Actually we should accept if better; ties not improving.
                if best_delta[0] < max(lengths):
                    if best_move[0] == 'relocate':
                        _, t1, i, t2, pos = best_move
                        cust = routes[t1][i]
                        del routes[t1][i]
                        routes[t2].insert(pos, cust)
                    else:
                        _, t1, i, t2, j = best_move
                        cust1 = routes[t1][i]
                        cust2 = routes[t2][j]
                        routes[t1][i] = cust2
                        routes[t2][j] = cust1
                    # Recompute lengths for affected routes
                    for t in {t1, t2}:
                        lengths[t] = compute_route_length(routes[t])
                    improved = True
                    report_best_vrp(routes)
        return
    
    # VNS parameters
    max_iterations = 50 * n
    iteration = 0
    kmax = 5
    k = 1
    while iteration < max_iterations:
        # Shake: apply k random moves (relocate or swap) to current solution
        shaken_routes = [list(r) for r in routes]
        shaken_lengths = list(lengths)
        for _ in range(k):
            move_type = random.choice(['relocate', 'swap'])
            if move_type == 'relocate':
                # pick two distinct routes with at least one customer
                routes_with_cust = [r for r in range(truck_count) if len(shaken_routes[r]) > 2]
                if len(routes_with_cust) < 2:
                    continue
                t1 = random.choice(routes_with_cust)
                t2 = random.choice([r for r in range(truck_count) if r != t1])
                i = random.randint(1, len(shaken_routes[t1])-2)
                # relocate to best position in t2? Or random position? For shake we can insert at random position.
                pos = random.randint(1, len(shaken_routes[t2])-1)
                cust = shaken_routes[t1][i]
                del shaken_routes[t1][i]
                shaken_routes[t2].insert(pos, cust)
                shaken_lengths[t1] = compute_route_length(shaken_routes[t1])
                shaken_lengths[t2] = compute_route_length(shaken_routes[t2])
            else:  # swap
                routes_with_cust = [r for r in range(truck_count) if len(shaken_routes[r]) > 2]
                if len(routes_with_cust) < 2:
                    continue
                t1 = random.choice(routes_with_cust)
                t2 = random.choice([r for r in range(truck_count) if r != t1 and len(shaken_routes[r]) > 2])
                i = random.randint(1, len(shaken_routes[t1])-2)
                j = random.randint(1, len(shaken_routes[t2])-2)
                shaken_routes[t1][i], shaken_routes[t2][j] = shaken_routes[t2][j], shaken_routes[t1][i]
                shaken_lengths[t1] = compute_route_length(shaken_routes[t1])
                shaken_lengths[t2] = compute_route_length(shaken_routes[t2])
        # Save current for potential restoration
        old_routes = [list(r) for r in routes]
        old_lengths = list(lengths)
        # Set as current
        routes = shaken_routes
        lengths = shaken_lengths
        # Apply local search
        local_search()
        new_max = max(lengths)
        best_max_cur = max(lengths)
        # Acceptance: if new solution is better than current best, keep it and reset k; else if new_max <= old_max, keep; else restore with probability? For VNS we accept if new cost is better or sometimes if equal? Standard VNS accepts if new is better, else increase k. So we compare to old_max (before shake). To simplify, if new_max < best_max_cur (i.e., improved over best), keep and k=1; else if new_max <= old_max, keep and k=1; else restore and increase k.
        # Actually we should compare to the solution before shake (old_max). We'll compute old_max = max(old_lengths).
        old_max = max(old_lengths)
        if new_max < old_max:
            # improvement, keep new solution, reset k
            k = 1
        else:
            # no improvement, restore old solution
            routes = old_routes
            lengths = old_lengths
            k = min(k+1, kmax)
        iteration += 1
    
    # Ensure best_routes has exactly truck_count routes, each with depot start/end, and every customer appears once
    # Already maintained
    # If any route is empty, it should be [0,0]
    for r in range(truck_count):
        if len(best_routes[r]) == 0 or (len(best_routes[r]) == 1 and best_routes[r][0] == 0):
            best_routes[r] = [0, 0]
    # Check all customers present
    all_customers = []
    for r in best_routes:
        for c in r:
            if c != 0:
                all_customers.append(c)
    missing = set(customers) - set(all_customers)
    if missing:
        # Fallback: assign missing customers arbitrarily (very unlikely)
        for c in missing:
            best_routes[0].insert(-1, c)
    return best_routes