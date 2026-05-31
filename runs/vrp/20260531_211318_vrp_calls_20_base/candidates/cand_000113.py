import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    dist = distance_matrix

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    def insertion_delta(route, pos, cust):
        prev = route[pos-1]
        nxt = route[pos]
        return dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt]

    def removal_delta(route, pos):
        prev = route[pos-1]
        nxt = route[pos+1]
        return dist[prev, route[pos]] + dist[route[pos], nxt] - dist[prev, nxt]

    # Initial construction: farthest-first insertion
    customers = sorted(range(1, n), key=lambda c: -dist[0][c])
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    for cust in customers:
        best_truck = None
        best_pos = None
        best_max = float('inf')
        best_total = float('inf')
        for t, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_dist = route_dists[t] + insertion_delta(route, pos, cust)
                new_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                new_total = sum(route_dists) + insertion_delta(route, pos, cust)
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = pos
        route = routes[best_truck]
        routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
        route_dists[best_truck] += insertion_delta(route, best_pos, cust)

    current_routes = [list(r) for r in routes]
    current_dists = list(route_dists)
    current_max = max(current_dists)
    current_total = sum(current_dists)
    best_routes = [list(r) for r in routes]
    best_dists = list(route_dists)
    best_max = current_max
    best_total = current_total
    report_best_vrp(best_routes)

    # Operator weights: 4 destroy, 3 repair
    destroy_weights = [1.0, 1.0, 1.0, 1.0]
    repair_weights = [1.0, 1.0, 1.0]
    destroy_scores = [0.0, 0.0, 0.0, 0.0]
    repair_scores = [0.0, 0.0, 0.0]
    destroy_usage = [0.0, 0.0, 0.0, 0.0]
    repair_usage = [0.0, 0.0, 0.0]

    max_iter = min(5000, 30 * n)
    removal_fraction = 0.2
    T0 = best_max * 2.0
    T = T0
    alpha = 0.995
    no_improve_iter = 0
    restart_threshold = int(0.2 * max_iter)
    decay = 0.8
    segment_length = max(1, int(0.1 * max_iter))
    iter_since_segment = 0

    for it in range(max_iter):
        num_removals = max(1, int(removal_fraction * (n - 1)))

        # Select destroy operator via roulette wheel
        total_d = sum(destroy_weights)
        destroy_probs = [w / total_d for w in destroy_weights]
        destroy_op = random.choices([0, 1, 2, 3], weights=destroy_probs)[0]

        # Destroy
        if destroy_op == 0:  # worst removal
            all_contribs = []
            for t, route in enumerate(current_routes):
                if len(route) <= 2:
                    continue
                for pos in range(1, len(route)-1):
                    contrib = removal_delta(route, pos)
                    all_contribs.append((contrib, t, pos, route[pos]))
            all_contribs.sort(key=lambda x: (-x[0], x[3]))
            to_remove = set()
            for _, t, pos, cust in all_contribs[:num_removals]:
                to_remove.add(cust)
            new_routes = []
            new_dists = []
            for t, route in enumerate(current_routes):
                new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                new_routes.append(new_route)
                new_dists.append(route_distance(new_route))
        elif destroy_op == 1:  # random removal
            all_customers = [c for r in current_routes for c in r[1:-1]]
            random.shuffle(all_customers)
            to_remove = set(all_customers[:num_removals])
            new_routes = []
            new_dists = []
            for route in current_routes:
                new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                new_routes.append(new_route)
                new_dists.append(route_distance(new_route))
        elif destroy_op == 2:  # route removal
            # Remove a random route and add its customers to removal list
            t_rem = random.randrange(truck_count)
            to_remove = set(current_routes[t_rem][1:-1])
            new_routes = []
            new_dists = []
            for t, route in enumerate(current_routes):
                if t == t_rem:
                    new_route = [0, 0]
                else:
                    new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                new_routes.append(new_route)
                new_dists.append(route_distance(new_route))
            # If more removals needed, add random ones
            if len(to_remove) < num_removals:
                remaining = [c for r in new_routes for c in r[1:-1]]
                random.shuffle(remaining)
                extra = remaining[:num_removals - len(to_remove)]
                to_remove.update(extra)
                new_routes = []
                new_dists = []
                for route in current_routes:
                    new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                    new_routes.append(new_route)
                    new_dists.append(route_distance(new_route))
        else:  # Shaw removal (similarity based on distance)
            # Choose a seed customer randomly, then remove customers most similar (smallest distance)
            all_customers = [c for r in current_routes for c in r[1:-1]]
            if not all_customers:
                to_remove = set()
            else:
                seed = random.choice(all_customers)
                # sort by distance to seed, related
                candidates = [(dist[seed][c], c) for c in all_customers if c != seed]
                candidates.sort(key=lambda x: x[0])
                to_remove = set([seed] + [c for _, c in candidates[:num_removals-1]])
            new_routes = []
            new_dists = []
            for route in current_routes:
                new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                new_routes.append(new_route)
                new_dists.append(route_distance(new_route))
        removed = list(to_remove)

        # Select repair operator
        total_r = sum(repair_weights)
        repair_probs = [w / total_r for w in repair_weights]
        repair_op = random.choices([0, 1, 2], weights=repair_probs)[0]

        # Repair
        if repair_op == 0:  # greedy
            routes_repair = [list(r) for r in new_routes]
            dists_repair = list(new_dists)
            unassigned = list(removed)
            current_max_repair = max(dists_repair)
            for cust in unassigned:
                best_truck = None
                best_pos = None
                best_new_max = float('inf')
                best_new_total = float('inf')
                best_delta = None
                for t, route in enumerate(routes_repair):
                    old_dist = dists_repair[t]
                    for pos in range(1, len(route)):
                        delta = insertion_delta(route, pos, cust)
                        new_dist = old_dist + delta
                        new_max = max(current_max_repair, new_dist)
                        new_total = sum(dists_repair) + delta
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                            best_new_max = new_max
                            best_new_total = new_total
                            best_truck = t
                            best_pos = pos
                            best_delta = delta
                route = routes_repair[best_truck]
                routes_repair[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
                dists_repair[best_truck] += best_delta
                if dists_repair[best_truck] > current_max_repair:
                    current_max_repair = dists_repair[best_truck]
            new_routes_final = routes_repair
            new_dists_final = dists_repair
        elif repair_op == 1:  # regret-2
            routes_repair = [list(r) for r in new_routes]
            dists_repair = list(new_dists)
            unassigned = list(removed)
            current_max_repair = max(dists_repair)
            while unassigned:
                best_info = None
                for cust in unassigned:
                    best_max_val = float('inf')
                    best_total_val = float('inf')
                    best_truck = None
                    best_pos = None
                    best_delta = None
                    second_best_max = float('inf')
                    second_best_total = float('inf')
                    for t, route in enumerate(routes_repair):
                        old_dist = dists_repair[t]
                        for pos in range(1, len(route)):
                            delta = insertion_delta(route, pos, cust)
                            new_dist = old_dist + delta
                            new_max = max(current_max_repair, new_dist)
                            new_total = sum(dists_repair) + delta
                            if new_max < best_max_val or (new_max == best_max_val and new_total < best_total_val):
                                second_best_max = best_max_val
                                second_best_total = best_total_val
                                best_max_val = new_max
                                best_total_val = new_total
                                best_truck = t
                                best_pos = pos
                                best_delta = delta
                            elif new_max < second_best_max or (new_max == second_best_max and new_total < second_best_total):
                                second_best_max = new_max
                                second_best_total = new_total
                    regret = (second_best_max - best_max_val) if second_best_max != float('inf') else float('inf')
                    if best_info is None or regret > best_info[0] or (regret == best_info[0] and (best_max_val < best_info[1] or (best_max_val == best_info[1] and cust < best_info[4]))):
                        best_info = (regret, best_max_val, best_total_val, cust, best_truck, best_pos, best_delta)
                regret, best_max_val, best_total_val, cust, best_truck, best_pos, best_delta = best_info
                route = routes_repair[best_truck]
                routes_repair[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
                dists_repair[best_truck] += best_delta
                if dists_repair[best_truck] > current_max_repair:
                    current_max_repair = dists_repair[best_truck]
                unassigned.remove(cust)
            new_routes_final = routes_repair
            new_dists_final = dists_repair
        else:  # greedy with noise
            routes_repair = [list(r) for r in new_routes]
            dists_repair = list(new_dists)
            unassigned = list(removed)
            current_max_repair = max(dists_repair)
            noise_range = 0.1 * np.max(dist) if n > 1 else 1.0
            for cust in unassigned:
                best_truck = None
                best_pos = None
                best_new_max = float('inf')
                best_new_total = float('inf')
                best_delta = None
                for t, route in enumerate(routes_repair):
                    old_dist = dists_repair[t]
                    for pos in range(1, len(route)):
                        delta = insertion_delta(route, pos, cust) + random.uniform(-noise_range, noise_range)
                        new_dist = old_dist + delta
                        new_max = max(current_max_repair, new_dist)
                        new_total = sum(dists_repair) + delta
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                            best_new_max = new_max
                            best_new_total = new_total
                            best_truck = t
                            best_pos = pos
                            best_delta = delta
                route = routes_repair[best_truck]
                routes_repair[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
                dists_repair[best_truck] += best_delta
                if dists_repair[best_truck] > current_max_repair:
                    current_max_repair = dists_repair[best_truck]
            new_routes_final = routes_repair
            new_dists_final = dists_repair

        new_max = max(new_dists_final)
        new_total = sum(new_dists_final)
        delta = new_max - current_max
        accepted = False
        if delta < 0 or (delta == 0 and new_total < current_total) or random.random() < math.exp(-delta / max(T, 1e-9)):
            accepted = True
            current_routes = [list(r) for r in new_routes_final]
            current_dists = list(new_dists_final)
            current_max = new_max
            current_total = new_total
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                best_max = new_max
                best_total = new_total
                best_routes = [list(r) for r in new_routes_final]
                best_dists = list(new_dists_final)
                report_best_vrp(best_routes)
                no_improve_iter = 0
                # Update scores: +1 for improving
                destroy_scores[destroy_op] += 1.0
                repair_scores[repair_op] += 1.0
            else:
                no_improve_iter += 1
                # Update scores: +0.5 for accepted but not improved
                destroy_scores[destroy_op] += 0.5
                repair_scores[repair_op] += 0.5
        else:
            no_improve_iter += 1
            # Update scores: +0 for not accepted
            # (no change)
        destroy_usage[destroy_op] += 1.0
        repair_usage[repair_op] += 1.0

        # Cooling
        T *= alpha

        # Restart if stuck (large perturbation)
        if no_improve_iter >= restart_threshold:
            large_removal_count = max(1, int(0.5 * (n - 1)))
            all_customers = [c for r in current_routes for c in r[1:-1]]
            random.shuffle(all_customers)
            to_remove = set(all_customers[:large_removal_count])
            new_routes = []
            new_dists = []
            for route in current_routes:
                new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                new_routes.append(new_route)
                new_dists.append(route_distance(new_route))
            # greedy with noise repair
            routes_repair = [list(r) for r in new_routes]
            dists_repair = list(new_dists)
            unassigned = list(to_remove)
            current_max_repair = max(dists_repair)
            noise_range = 0.05 * np.max(dist) if n > 1 else 1.0
            for cust in unassigned:
                best_truck = None
                best_pos = None
                best_new_max = float('inf')
                best_new_total = float('inf')
                best_delta = None
                for t, route in enumerate(routes_repair):
                    old_dist = dists_repair[t]
                    for pos in range(1, len(route)):
                        delta = insertion_delta(route, pos, cust) + random.uniform(-noise_range, noise_range)
                        new_dist = old_dist + delta
                        new_max = max(current_max_repair, new_dist)
                        new_total = sum(dists_repair) + delta
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                            best_new_max = new_max
                            best_new_total = new_total
                            best_truck = t
                            best_pos = pos
                            best_delta = delta
                route = routes_repair[best_truck]
                routes_repair[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
                dists_repair[best_truck] += best_delta
                if dists_repair[best_truck] > current_max_repair:
                    current_max_repair = dists_repair[best_truck]
            current_routes = routes_repair
            current_dists = dists_repair
            current_max = max(current_dists)
            current_total = sum(current_dists)
            T = T0
            no_improve_iter = 0

        # Decay and reset weights every segment
        iter_since_segment += 1
        if iter_since_segment == segment_length:
            # Update weights based on scores
            for i in range(len(destroy_weights)):
                if destroy_usage[i] > 0:
                    destroy_weights[i] = decay * destroy_weights[i] + (1 - decay) * (destroy_scores[i] / destroy_usage[i] + 0.01)
                else:
                    destroy_weights[i] = decay * destroy_weights[i] + (1 - decay) * 0.01
            for i in range(len(repair_weights)):
                if repair_usage[i] > 0:
                    repair_weights[i] = decay * repair_weights[i] + (1 - decay) * (repair_scores[i] / repair_usage[i] + 0.01)
                else:
                    repair_weights[i] = decay * repair_weights[i] + (1 - decay) * 0.01
            destroy_scores = [0.0] * 4
            repair_scores = [0.0] * 3
            destroy_usage = [0.0] * 4
            repair_usage = [0.0] * 3
            iter_since_segment = 0

    # Post-optimization: 2-opt on best solution
    improved = True
    while improved:
        improved = False
        for t, route in enumerate(best_routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dists[t] - 1e-9:
                        new_max = max(best_dists[:t] + [new_dist] + best_dists[t+1:])
                        new_total = sum(best_dists[:t]) + new_dist + sum(best_dists[t+1:])
                        if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                            best_routes[t] = new_route
                            best_dists[t] = new_dist
                            best_max = new_max
                            best_total = new_total
                            report_best_vrp(best_routes)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

    return best_routes