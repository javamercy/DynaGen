import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    def report_best_vrp(routes):
        pass
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

    # Random initial construction: shuffle customers, greedy insertion
    customers = list(range(1, n))
    random.shuffle(customers)
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

    # Adaptive weights for 3 destroy operators: worst, random, shaw
    destroy_weights = [1.0, 1.0, 1.0]
    # Repair: greedy, best (with random tie-break)
    repair_weights = [1.0, 1.0]
    alpha = 0.9
    score_best = 3.0
    score_accepted = 1.0
    score_rejected = 0.0

    max_iter = min(5000, 30 * n)
    removal_start = 0.3
    removal_end = 0.1
    temp_start = 100.0
    temp_end = 0.01
    no_improve_iter = 0
    restart_threshold = int(0.3 * max_iter)
    large_removal_fraction = 0.5
    shaking_interval = int(0.1 * max_iter)

    def route_similarity(routes1, routes2):
        set1 = set()
        for r in routes1:
            for i in range(len(r)-1):
                set1.add((min(r[i], r[i+1]), max(r[i], r[i+1])))
        set2 = set()
        for r in routes2:
            for i in range(len(r)-1):
                set2.add((min(r[i], r[i+1]), max(r[i], r[i+1])))
        intersect = set1 & set2
        union = set1 | set2
        return len(intersect) / len(union) if union else 0.0

    # Shaw removal: remove customers that are similar to a seed customer
    def shaw_removal(removed_count):
        # Choose a seed customer randomly from current routes
        all_custs = [c for r in current_routes for c in r[1:-1]]
        if not all_custs:
            return []
        seed = random.choice(all_custs)
        # Compute relatedness: distance + (1 if same route else 0)
        related = []
        for c in all_custs:
            if c == seed:
                continue
            same_route = 0
            for r in current_routes:
                if seed in r and c in r:
                    same_route = 1
                    break
            rel = dist[seed][c] + same_route
            related.append((rel, c))
        related.sort(key=lambda x: x[0])
        to_remove = [seed]
        # Randomly select from among the most related
        for _ in range(removed_count - 1):
            if not related:
                break
            idx = random.randint(0, min(len(related)-1, int(removed_count*0.5)))
            to_remove.append(related[idx][1])
            del related[idx]
        return to_remove

    for it in range(max_iter):
        removal_fraction = removal_start + (removal_end - removal_start) * (it / max_iter)
        num_removals = max(1, int(removal_fraction * (n - 1)))
        temperature = temp_start + (temp_end - temp_start) * (it / max_iter)

        # Select destroy operator
        total_d = sum(destroy_weights)
        destroy_probs = [w / total_d for w in destroy_weights]
        destroy_op = random.choices([0, 1, 2], weights=destroy_probs)[0]

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
        elif destroy_op == 1:  # random removal
            all_customers = [c for r in current_routes for c in r[1:-1]]
            random.shuffle(all_customers)
            to_remove = set(all_customers[:num_removals])
        else:  # shaw removal
            to_remove = set(shaw_removal(num_removals))

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
        repair_op = random.choices([0, 1], weights=repair_probs)[0]

        # Repair
        if repair_op == 0:  # greedy (deterministic)
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
        else:  # best insertion (random tie-break)
            routes_repair = [list(r) for r in new_routes]
            dists_repair = list(new_dists)
            unassigned = list(removed)
            current_max_repair = max(dists_repair)
            for cust in unassigned:
                best_options = []
                best_new_max = float('inf')
                best_new_total = float('inf')
                for t, route in enumerate(routes_repair):
                    old_dist = dists_repair[t]
                    for pos in range(1, len(route)):
                        delta = insertion_delta(route, pos, cust)
                        new_dist = old_dist + delta
                        new_max = max(current_max_repair, new_dist)
                        new_total = sum(dists_repair) + delta
                        if new_max < best_new_max - 1e-9 or (abs(new_max - best_new_max) < 1e-9 and new_total < best_new_total - 1e-9):
                            best_new_max = new_max
                            best_new_total = new_total
                            best_options = [(t, pos, delta)]
                        elif abs(new_max - best_new_max) < 1e-9 and abs(new_total - best_new_total) < 1e-9:
                            best_options.append((t, pos, delta))
                t, pos, delta = random.choice(best_options)
                route = routes_repair[t]
                routes_repair[t] = route[:pos] + [cust] + route[pos:]
                dists_repair[t] += delta
                if dists_repair[t] > current_max_repair:
                    current_max_repair = dists_repair[t]
            new_routes_final = routes_repair
            new_dists_final = dists_repair

        new_max = max(new_dists_final)
        new_total = sum(new_dists_final)
        accepted = False
        # Simulated Annealing acceptance (minimize)
        if new_max <= current_max:
            accepted = True
        else:
            delta_max = new_max - current_max
            prob = math.exp(-delta_max / temperature) if temperature > 0 else 0.0
            if random.random() < prob:
                accepted = True
        if accepted:
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
                destroy_weights[destroy_op] = alpha * destroy_weights[destroy_op] + (1 - alpha) * score_best
                repair_weights[repair_op] = alpha * repair_weights[repair_op] + (1 - alpha) * score_best
            else:
                no_improve_iter += 1
                destroy_weights[destroy_op] = alpha * destroy_weights[destroy_op] + (1 - alpha) * score_accepted
                repair_weights[repair_op] = alpha * repair_weights[repair_op] + (1 - alpha) * score_accepted
        else:
            no_improve_iter += 1
            destroy_weights[destroy_op] = alpha * destroy_weights[destroy_op] + (1 - alpha) * score_rejected
            repair_weights[repair_op] = alpha * repair_weights[repair_op] + (1 - alpha) * score_rejected

        # Shaking: if no improvement for shaking_interval, apply large random perturbation
        if no_improve_iter > 0 and no_improve_iter % shaking_interval == 0:
            # Shaking: swap random customers between routes or relocate
            for _ in range(max(1, int(0.1 * (n-1)))):
                all_cust = [c for r in current_routes for c in r[1:-1]]
                if len(all_cust) < 2:
                    break
                a, b = random.sample(all_cust, 2)
                # Find routes containing a and b
                route_a_idx = None
                route_b_idx = None
                for idx, r in enumerate(current_routes):
                    if a in r[1:-1]:
                        route_a_idx = idx
                    if b in r[1:-1]:
                        route_b_idx = idx
                if route_a_idx is None or route_b_idx is None:
                    continue
                # Swap positions
                route_a = current_routes[route_a_idx]
                route_b = current_routes[route_b_idx]
                pos_a = route_a.index(a)
                pos_b = route_b.index(b)
                # Perform swap only if it changes the solution
                if route_a_idx == route_b_idx:
                    # Same route: swap positions
                    route_a[pos_a], route_a[pos_b] = route_a[pos_b], route_a[pos_a]
                    current_dists[route_a_idx] = route_distance(route_a)
                else:
                    # Different routes: exchange customers
                    route_a[pos_a] = b
                    route_b[pos_b] = a
                    current_dists[route_a_idx] = route_distance(route_a)
                    current_dists[route_b_idx] = route_distance(route_b)
                current_max = max(current_dists)
                current_total = sum(current_dists)
                if current_max < best_max - 1e-9 or (abs(current_max - best_max) < 1e-9 and current_total < best_total):
                    best_max = current_max
                    best_total = current_total
                    best_routes = [list(r) for r in current_routes]
                    best_dists = list(current_dists)
                    report_best_vrp(best_routes)

        # Reactive restart based on similarity
        similarity = route_similarity(current_routes, best_routes)
        if no_improve_iter >= restart_threshold or (no_improve_iter > 10 and similarity > 0.9):
            large_removal_count = max(1, int(large_removal_fraction * (n - 1)))
            all_customers = [c for r in current_routes for c in r[1:-1]]
            random.shuffle(all_customers)
            to_remove = set(all_customers[:large_removal_count])
            new_routes = []
            new_dists = []
            for route in current_routes:
                new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                new_routes.append(new_route)
                new_dists.append(route_distance(new_route))
            # Greedy repair
            routes_repair = [list(r) for r in new_routes]
            dists_repair = list(new_dists)
            unassigned = list(to_remove)
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
            current_routes = routes_repair
            current_dists = dists_repair
            current_max = max(current_dists)
            current_total = sum(current_dists)
            no_improve_iter = 0

    # 2-opt post-optimization on best solution
    max_opt_iter = 200
    for _ in range(max_opt_iter):
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
        if not improved:
            break

    return best_routes