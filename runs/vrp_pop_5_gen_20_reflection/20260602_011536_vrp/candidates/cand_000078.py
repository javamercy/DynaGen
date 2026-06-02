import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()

    def route_distance(route):
        if len(route) < 2:
            return 0
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))

    def objective(routes):
        return max(route_distance(r) for r in routes)

    # Minimax construction
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    while unassigned:
        best_max = float('inf')
        best_total = float('inf')
        best_node = None
        best_route = None
        best_pos = None
        for node in unassigned:
            for r in range(truck_count):
                route = routes[r]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [node] + route[pos:]
                    new_route_dist = route_distance(new_route)
                    new_max = max(route_distance(routes[rr]) if rr != r else new_route_dist for rr in range(truck_count))
                    if (new_max < best_max) or (new_max == best_max and new_route_dist < best_total):
                        best_max = new_max
                        best_total = new_route_dist
                        best_node = node
                        best_route = r
                        best_pos = pos
        routes[best_route].insert(best_pos, best_node)
        unassigned.remove(best_node)

    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)
    report_best_vrp(best_routes)

    max_iter = min(50, 2 * n)
    T_start = 5.0
    T_end = 0.1
    routes = [list(r) for r in best_routes]
    routes_obj = best_obj

    for it in range(max_iter):
        current_routes = [list(r) for r in routes]
        # Ruin: biased removal from longest route
        route_lengths = [len(r)-2 for r in current_routes]  # number of customers per route
        max_len = max(route_lengths)
        # probability proportional to route length
        total_len = sum(route_lengths)
        if total_len == 0:
            removal_probs = [1/truck_count] * truck_count
        else:
            removal_probs = [l/total_len for l in route_lengths]
        remove_frac = random.uniform(0.2, 0.4)
        remove_count = max(1, int(remove_frac * (n-1)))
        # choose routes to remove customers from based on probabilities
        # we'll sample customers uniformly from chosen routes (biased route selection)
        # Alternatively: for each customer, assign to route, then sample with weights from routes
        # Simpler: first decide how many to remove per route using multinomial
        # but we'll just sample customers by first selecting route, then customer from that route
        removed_list = []
        customers_by_route = [[] for _ in range(truck_count)]
        for r in range(truck_count):
            for node in current_routes[r][1:-1]:
                customers_by_route[r].append(node)
        all_customers = [(r, c) for r in range(truck_count) for c in customers_by_route[r]]
        if len(all_customers) == 0:
            continue
        # sample without replacement weighted by route length (route's customer count)
        selected = random.choices(range(truck_count), weights=route_lengths, k=remove_count)
        # but we need distinct customers, so we'll just sample customers uniformly from all but with bias
        # Actually, we want more from long routes; let's do this: for each removal, pick a route with probability proportional to its customer count, then pick a random customer from that route.
        # To avoid duplicates, we can shuffle and remove sequentially.
        # We'll create a list of (route_idx, customer) for all customers, then sample without replacement with weights equal to 1/len_of_route? No, we want route-level weight.
        # Implement by repeatedly sampling route, then customer, and removing from list.
        all_customer_list = [(r, c) for r in range(truck_count) for c in customers_by_route[r]]
        random.shuffle(all_customer_list)
        # We'll just pick first remove_count from shuffled list (but that's uniform over customers, not biased)
        # To bias, we can sort routes by length, then take more from long ones? But random.
        # Let's do: for each removal with probability 0.7 pick a customer from the longest route (or one of the longest if tie), else uniformly random.
        # Code below implements that.
        removed_candidates = []
        # Identify longest route(s)
        longest_routes = [i for i, l in enumerate(route_lengths) if l == max_len]
        for _ in range(remove_count):
            if random.random() < 0.7 and longest_routes:
                r = random.choice(longest_routes)
            else:
                r = random.randrange(truck_count)
            # pick a random customer from that route that hasn't been removed yet
            candidates = [c for c in customers_by_route[r] if c not in removed_candidates]
            if not candidates:
                # fallback to any other route
                for rr in range(truck_count):
                    if rr == r:
                        continue
                    candidates = [c for c in customers_by_route[rr] if c not in removed_candidates]
                    if candidates:
                        r = rr
                        break
                if not candidates:
                    break
            c = random.choice(candidates)
            removed_candidates.append(c)
        removed_list = removed_candidates
        # Remove customers from routes
        to_remove_set = set(removed_list)
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            new_route = [route[0]]
            for node in route[1:-1]:
                if node in to_remove_set:
                    continue
                else:
                    new_route.append(node)
            new_route.append(0)
            current_routes[r_idx] = new_route
            if len(current_routes[r_idx]) < 2:
                current_routes[r_idx] = [0, 0]
        random.shuffle(removed_list)

        # Reconstruct with minimax and tie-breaking
        unassigned = removed_list
        while unassigned:
            best_max = float('inf')
            best_route_total = float('inf')
            best_candidates = []
            for node in unassigned:
                for r in range(truck_count):
                    route = current_routes[r]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [node] + route[pos:]
                        new_route_dist = route_distance(new_route)
                        new_max = max(route_distance(current_routes[rr]) if rr != r else new_route_dist for rr in range(truck_count))
                        current_route_dist = route_distance(route)
                        if new_max < best_max or (new_max == best_max and new_route_dist < best_route_total) or (new_max == best_max and new_route_dist == best_route_total and current_route_dist < best_candidates[0][3] if best_candidates else True):
                            best_max = new_max
                            best_route_total = new_route_dist
                            best_candidates = [(node, r, pos, current_route_dist)]
                        elif new_max == best_max and new_route_dist == best_route_total and current_route_dist == best_candidates[0][3]:
                            best_candidates.append((node, r, pos, current_route_dist))
            if not best_candidates:
                break
            chosen = min(best_candidates, key=lambda x: (x[3], x[1], x[2]))
            node, r, pos, _ = chosen
            current_routes[r].insert(pos, node)
            unassigned.remove(node)

        # Inter-route improvement: relocate and swap (first improvement reducing max)
        improved = True
        attempts = 0
        max_attempts = 20 * truck_count
        while improved and attempts < max_attempts:
            improved = False
            attempts += 1
            current_max = objective(current_routes)
            best_delta = 0
            best_move = None
            # Relocate
            for i in range(truck_count):
                if len(current_routes[i]) <= 2:
                    continue
                for ci_idx in range(1, len(current_routes[i])-1):
                    ci = current_routes[i][ci_idx]
                    for j in range(truck_count):
                        if i == j:
                            continue
                        for cj_idx in range(1, len(current_routes[j])):
                            new_route_i = current_routes[i][:ci_idx] + current_routes[i][ci_idx+1:]
                            new_route_j = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx:]
                            new_max = max(route_distance(new_route_i), route_distance(new_route_j))
                            for k in range(truck_count):
                                if k != i and k != j:
                                    new_max = max(new_max, route_distance(current_routes[k]))
                            if new_max < current_max:
                                delta = current_max - new_max
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = ('relocate', i, ci_idx, j, cj_idx)
            # Swap
            for i in range(truck_count):
                if len(current_routes[i]) <= 2:
                    continue
                for ci_idx in range(1, len(current_routes[i])-1):
                    ci = current_routes[i][ci_idx]
                    for j in range(i+1, truck_count):
                        if len(current_routes[j]) <= 2:
                            continue
                        for cj_idx in range(1, len(current_routes[j])-1):
                            cj = current_routes[j][cj_idx]
                            new_route_i = current_routes[i][:ci_idx] + [cj] + current_routes[i][ci_idx+1:]
                            new_route_j = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx+1:]
                            new_max = max(route_distance(new_route_i), route_distance(new_route_j))
                            for k in range(truck_count):
                                if k != i and k != j:
                                    new_max = max(new_max, route_distance(current_routes[k]))
                            if new_max < current_max:
                                delta = current_max - new_max
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = ('swap', i, ci_idx, j, cj_idx)
            if best_move is not None:
                if best_move[0] == 'relocate':
                    _, i, ci_idx, j, cj_idx = best_move
                    ci = current_routes[i][ci_idx]
                    del current_routes[i][ci_idx]
                    if len(current_routes[i]) == 1:
                        current_routes[i] = [0, 0]
                    current_routes[j].insert(cj_idx, ci)
                else:
                    _, i, ci_idx, j, cj_idx = best_move
                    ci = current_routes[i][ci_idx]
                    cj = current_routes[j][cj_idx]
                    current_routes[i][ci_idx] = cj
                    current_routes[j][cj_idx] = ci
                improved = True

        # Intra-route 2-opt: extra iterations on longest route(s)
        # First apply standard 2-opt on all routes
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            improved_opt = True
            for _ in range(10):
                improved_opt = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if route_distance(new_route) < route_distance(route):
                            route = new_route
                            improved_opt = True
                            break
                    if improved_opt:
                        break
                if not improved_opt:
                    break
            current_routes[r_idx] = route
        # Extra 2-opt on the longest route(s) - more iterations
        route_lengths = [len(r)-2 for r in current_routes]
        max_len = max(route_lengths) if route_lengths else 0
        longest_routes = [i for i, l in enumerate(route_lengths) if l == max_len]
        for r_idx in longest_routes:
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            for _ in range(20):  # extra iterations
                improved_opt = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if route_distance(new_route) < route_distance(route):
                            route = new_route
                            improved_opt = True
                            break
                    if improved_opt:
                        break
                if not improved_opt:
                    break
            current_routes[r_idx] = route

        new_obj = objective(current_routes)
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)
        # Simulated annealing acceptance
        T = T_start * (T_end / T_start) ** (it / max_iter)
        delta = new_obj - routes_obj
        if delta < 0 or random.random() < math.exp(-delta / T):
            routes = [list(r) for r in current_routes]
            routes_obj = new_obj

    return best_routes