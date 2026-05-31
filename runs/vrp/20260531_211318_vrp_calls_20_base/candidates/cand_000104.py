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

    # Initial random order, greedy insertion minimizing max distance
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
                delta = insertion_delta(route, pos, cust)
                new_dist = route_dists[t] + delta
                new_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                new_total = sum(route_dists) + delta
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

    max_iter = min(2000, 15 * n)
    removal_fraction = 0.25
    num_removals = max(1, int(removal_fraction * (n - 1)))
    T0 = best_max / 2.0
    T = T0
    no_improve = 0
    restart_threshold = 100

    for it in range(max_iter):
        # Destroy operator selection
        destroy_op = random.randint(0, 2)

        # Destroy
        if destroy_op == 0:  # worst removal
            all_contribs = []
            for t, route in enumerate(current_routes):
                if len(route) <= 2:
                    continue
                for pos in range(1, len(route)-1):
                    contrib = removal_delta(route, pos)
                    all_contribs.append((contrib, t, pos, route[pos]))
            all_contribs.sort(key=lambda x: x[0], reverse=True)
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
        else:  # route removal: remove all customers from a random non-empty route
            non_empty = [t for t, r in enumerate(current_routes) if len(r) > 2]
            if non_empty:
                t_remove = random.choice(non_empty)
                route = current_routes[t_remove]
                to_remove = set(route[1:-1])
                new_routes = []
                new_dists = []
                for t, route in enumerate(current_routes):
                    if t == t_remove:
                        new_route = [0, 0]
                    else:
                        new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                    new_routes.append(new_route)
                    new_dists.append(route_distance(new_route))
            else:
                # fallback to random removal
                all_customers = [c for r in current_routes for c in r[1:-1]]
                random.shuffle(all_customers)
                to_remove = set(all_customers[:num_removals])
                new_routes = []
                new_dists = []
                for route in current_routes:
                    new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                    new_routes.append(new_route)
                    new_dists.append(route_distance(new_route))
        removed = list(to_remove)

        # Greedy repair with occasional random insertion
        routes_repair = [list(r) for r in new_routes]
        dists_repair = list(new_dists)
        unassigned = list(removed)
        current_max_repair = max(dists_repair) if dists_repair else 0.0
        random.shuffle(unassigned)
        for cust in unassigned:
            # decide if random insertion
            if random.random() < 0.2:
                # random feasible insertion
                feasible = []
                for t, route in enumerate(routes_repair):
                    for pos in range(1, len(route)):
                        delta = insertion_delta(route, pos, cust)
                        new_dist = dists_repair[t] + delta
                        feasible.append((t, pos, new_dist, delta))
                if feasible:
                    t, pos, new_dist, delta = random.choice(feasible)
                    route = routes_repair[t]
                    routes_repair[t] = route[:pos] + [cust] + route[pos:]
                    dists_repair[t] += delta
                    if dists_repair[t] > current_max_repair:
                        current_max_repair = dists_repair[t]
                continue
            # else greedy best
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
            if best_truck is not None:
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
        if delta < 0 or (delta == 0 and new_total < current_total) or random.random() < math.exp(-delta / max(T, 1e-9)):
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
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        # Restart if stuck
        if no_improve >= restart_threshold:
            # Perturb best solution: random swaps between routes
            perturbed_routes = [list(r) for r in best_routes]
            perturbed_dists = list(best_dists)
            for _ in range(5):
                # pick two non-empty routes
                non_empty = [t for t, r in enumerate(perturbed_routes) if len(r) > 2]
                if len(non_empty) < 2:
                    break
                t1, t2 = random.sample(non_empty, 2)
                # pick random customers (excluding depot)
                if len(perturbed_routes[t1]) <= 2 or len(perturbed_routes[t2]) <= 2:
                    continue
                p1 = random.randint(1, len(perturbed_routes[t1])-2)
                p2 = random.randint(1, len(perturbed_routes[t2])-2)
                c1 = perturbed_routes[t1][p1]
                c2 = perturbed_routes[t2][p2]
                # swap
                perturbed_routes[t1][p1] = c2
                perturbed_routes[t2][p2] = c1
                perturbed_dists[t1] = route_distance(perturbed_routes[t1])
                perturbed_dists[t2] = route_distance(perturbed_routes[t2])
            current_routes = perturbed_routes
            current_dists = perturbed_dists
            current_max = max(current_dists)
            current_total = sum(current_dists)
            no_improve = 0
            T = T0  # reheat

        # Linear cooling
        T = T0 * (1 - it / max_iter)

    # 2-opt post-optimization on best solution
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