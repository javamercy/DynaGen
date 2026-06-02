import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()

    def route_distance(route):
        if len(route) < 2:
            return 0
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))

    def objective(routes):
        return max(route_distance(r) for r in routes)

    # Initial solution via minimax construction
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
                    # compute new distance for this route
                    new_d = 0
                    prev = route[0]
                    for k in range(1, len(route)):
                        if k == pos:
                            new_d += dist[prev][node]
                            prev = node
                        new_d += dist[prev][route[k]]
                        prev = route[k]
                    # current max among all routes
                    current_max = 0
                    for rr in range(truck_count):
                        if rr == r:
                            d = new_d
                        else:
                            d = route_distance(routes[rr])
                        if d > current_max:
                            current_max = d
                    if (current_max < best_max) or (current_max == best_max and new_d < best_total):
                        best_max = current_max
                        best_total = new_d
                        best_node = node
                        best_route = r
                        best_pos = pos
        routes[best_route].insert(best_pos, best_node)
        unassigned.remove(best_node)

    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)
    report_best_vrp(best_routes)  # internal call

    # Ruin-and-recreate with targeted removal from worst route
    max_iter = min(50, 2 * n)
    T_start = 5.0
    T_end = 0.1
    routes_current = [list(r) for r in routes]
    obj_current = objective(routes_current)

    for iteration in range(max_iter):
        # Identify the route(s) with maximum distance
        route_dists = [route_distance(route) for route in routes_current]
        max_dist = max(route_dists)
        # Select all routes with max distance, pick one randomly
        worst_routes = [i for i, d in enumerate(route_dists) if d == max_dist]
        target_route_idx = random.choice(worst_routes)
        target_route = routes_current[target_route_idx]

        # Only proceed if route has customers
        if len(target_route) <= 2:
            continue

        # Compute combined weight for each customer in target_route
        customers = target_route[1:-1]  # exclude depot
        weights = []
        for node in customers:
            # detour contribution: distance saved if node removed
            i = target_route.index(node)
            prev_node = target_route[i-1]
            next_node = target_route[i+1]
            contrib = dist[prev_node][node] + dist[node][next_node] - dist[prev_node][next_node]
            # distance from depot
            depot_dist = dist[0][node]
            combined = contrib + 0.5 * depot_dist  # factor can be tuned
            weights.append(combined)

        # remove a random fraction from target route
        remove_frac = random.uniform(0.2, 0.4)
        remove_count = max(1, int(remove_frac * len(customers)))
        # weighted random selection without replacement
        if sum(weights) == 0:
            selected_nodes = random.sample(customers, min(remove_count, len(customers)))
        else:
            cum_weights = []
            s = 0
            for w in weights:
                s += w
                cum_weights.append(s)
            total = cum_weights[-1]
            selected = set()
            while len(selected) < remove_count and len(selected) < len(customers):
                r = random.random() * total
                lo, hi = 0, len(cum_weights)-1
                while lo < hi:
                    mid = (lo+hi)//2
                    if cum_weights[mid] < r:
                        lo = mid+1
                    else:
                        hi = mid
                idx = lo
                node = customers[idx]
                if node not in selected:
                    selected.add(node)
            selected_nodes = list(selected)

        # Remove selected nodes from target route
        new_route = [0]
        for node in target_route[1:-1]:
            if node not in selected_nodes:
                new_route.append(node)
        new_route.append(0)
        routes_current[target_route_idx] = new_route
        removed = selected_nodes

        # Shuffle removed list for randomness
        random.shuffle(removed)

        # Reconstruct via minimax insertion with random tie-breaking
        unassigned = removed
        while unassigned:
            best_candidates = []
            best_max = float('inf')
            best_total = float('inf')
            for node in unassigned:
                for r in range(truck_count):
                    route = routes_current[r]
                    for pos in range(1, len(route)):
                        # compute new distance for this route
                        new_d = 0
                        prev = route[0]
                        for k in range(1, len(route)):
                            if k == pos:
                                new_d += dist[prev][node]
                                prev = node
                            new_d += dist[prev][route[k]]
                            prev = route[k]
                        # current max among all routes
                        current_max = 0
                        for rr in range(truck_count):
                            if rr == r:
                                d = new_d
                            else:
                                d = route_distance(routes_current[rr])
                            if d > current_max:
                                current_max = d
                        if current_max < best_max:
                            best_max = current_max
                            best_total = new_d
                            best_candidates = [(node, r, pos)]
                        elif current_max == best_max:
                            if new_d < best_total:
                                best_total = new_d
                                best_candidates = [(node, r, pos)]
                            elif new_d == best_total:
                                best_candidates.append((node, r, pos))
            if not best_candidates:
                break
            chosen = random.choice(best_candidates)
            node, r, pos = chosen
            routes_current[r].insert(pos, node)
            unassigned.remove(node)

        # Intra-route 2-opt limited steps
        for r_idx in range(truck_count):
            route = routes_current[r_idx]
            if len(route) <= 3:
                continue
            for _ in range(10):
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if route_distance(new_route) < route_distance(route):
                            route = new_route
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            routes_current[r_idx] = route

        new_obj = objective(routes_current)
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [list(r) for r in routes_current]
            report_best_vrp(best_routes)
            # accept new as current
            obj_current = new_obj
        else:
            # Simulated annealing acceptance
            T = T_start - (T_start - T_end) * (iteration / max_iter)
            delta = new_obj - obj_current
            if delta > 0:
                if random.random() < np.exp(-delta / T):
                    obj_current = new_obj
                else:
                    # revert to previous current? Not needed because we don't store previous.
                    # We'll skip updating routes_current, but we need to revert. 
                    # Actually we already modified routes_current; we need to keep a copy before modification.
                    pass
        # Actually we need to handle rejection properly: revert routes_current to previous state.
        # For simplicity, we'll just not modify routes_current on rejection? But we already did.
        # To fix, we should make a copy of routes_current before ruin phase.
        # We'll revise the loop: save state before ruin, and revert if not accepted.
        # Below is corrected version with backup.
        # (The code above is conceptual; final code below includes proper backup.)

    return best_routes

# Note: The above code has a flaw in SA acceptance; the actual implementation below corrects it.
# Due to instruction length limits, the final code is provided as a single block after this message.

# Actual implementation with proper backup and SA acceptance:

import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()

    def route_distance(route):
        if len(route) < 2:
            return 0
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))

    def objective(routes):
        return max(route_distance(r) for r in routes)

    # Initial solution via minimax construction
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
                    new_d = 0
                    prev = route[0]
                    for k in range(1, len(route)):
                        if k == pos:
                            new_d += dist[prev][node]
                            prev = node
                        new_d += dist[prev][route[k]]
                        prev = route[k]
                    current_max = 0
                    for rr in range(truck_count):
                        if rr == r:
                            d = new_d
                        else:
                            d = route_distance(routes[rr])
                        if d > current_max:
                            current_max = d
                    if (current_max < best_max) or (current_max == best_max and new_d < best_total):
                        best_max = current_max
                        best_total = new_d
                        best_node = node
                        best_route = r
                        best_pos = pos
        routes[best_route].insert(best_pos, best_node)
        unassigned.remove(best_node)

    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)
    report_best_vrp(best_routes)

    # Ruin-and-recreate with targeted removal from worst route
    max_iter = min(50, 2 * n)
    T_start = 5.0
    T_end = 0.1
    current_routes = [list(r) for r in routes]
    current_obj = objective(current_routes)

    for iteration in range(max_iter):
        # Backup state
        backup_routes = [list(r) for r in current_routes]
        backup_obj = current_obj

        # Identify worst route
        route_dists = [route_distance(r) for r in current_routes]
        max_dist = max(route_dists)
        worst_indices = [i for i, d in enumerate(route_dists) if d == max_dist]
        target_idx = random.choice(worst_indices)
        target_route = current_routes[target_idx]

        if len(target_route) <= 2:
            continue

        # Compute weights for customers in target route
        customers = target_route[1:-1]
        weights = []
        for node in customers:
            i = target_route.index(node)
            prev_node = target_route[i-1]
            next_node = target_route[i+1]
            contrib = dist[prev_node][node] + dist[node][next_node] - dist[prev_node][next_node]
            depot_dist = dist[0][node]
            combined = contrib + 0.5 * depot_dist
            weights.append(combined)

        remove_frac = random.uniform(0.2, 0.4)
        remove_count = max(1, int(remove_frac * len(customers)))
        if sum(weights) == 0:
            to_remove = set(random.sample(customers, min(remove_count, len(customers))))
        else:
            cum_weights = []
            s = 0
            for w in weights:
                s += w
                cum_weights.append(s)
            total = cum_weights[-1]
            to_remove = set()
            while len(to_remove) < remove_count and len(to_remove) < len(customers):
                rnd = random.random() * total
                lo, hi = 0, len(cum_weights)-1
                while lo < hi:
                    mid = (lo+hi)//2
                    if cum_weights[mid] < rnd:
                        lo = mid+1
                    else:
                        hi = mid
                node = customers[lo]
                if node not in to_remove:
                    to_remove.add(node)

        # Remove customers from target route
        new_route = [0]
        for node in target_route[1:-1]:
            if node not in to_remove:
                new_route.append(node)
        new_route.append(0)
        current_routes[target_idx] = new_route

        # Reconstruct
        removed_list = list(to_remove)
        random.shuffle(removed_list)
        unassigned = removed_list
        while unassigned:
            best_candidates = []
            best_max = float('inf')
            best_total = float('inf')
            for node in unassigned:
                for r in range(truck_count):
                    route = current_routes[r]
                    for pos in range(1, len(route)):
                        # compute new distance for this route
                        new_d = 0
                        prev = route[0]
                        for k in range(1, len(route)):
                            if k == pos:
                                new_d += dist[prev][node]
                                prev = node
                            new_d += dist[prev][route[k]]
                            prev = route[k]
                        # current max among all routes
                        current_max = 0
                        for rr in range(truck_count):
                            if rr == r:
                                d = new_d
                            else:
                                d = route_distance(current_routes[rr])
                            if d > current_max:
                                current_max = d
                        if current_max < best_max:
                            best_max = current_max
                            best_total = new_d
                            best_candidates = [(node, r, pos)]
                        elif current_max == best_max:
                            if new_d < best_total:
                                best_total = new_d
                                best_candidates = [(node, r, pos)]
                            elif new_d == best_total:
                                best_candidates.append((node, r, pos))
            if not best_candidates:
                break
            chosen = random.choice(best_candidates)
            node, r, pos = chosen
            current_routes[r].insert(pos, node)
            unassigned.remove(node)

        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            for _ in range(10):
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if route_distance(new_route) < route_distance(route):
                            route = new_route
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            current_routes[r_idx] = route

        new_obj = objective(current_routes)
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)
            current_obj = new_obj
        else:
            # Simulated annealing acceptance
            T = T_start - (T_start - T_end) * (iteration / max_iter)
            delta = new_obj - current_obj
            if delta > 0 and random.random() < np.exp(-delta / T):
                current_obj = new_obj
            else:
                # Revert to backup
                current_routes = backup_routes
                current_obj = backup_obj

    return best_routes