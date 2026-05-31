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

    # Initial construction: random order greedy insertion minimizing max distance
    customers = list(range(1, n))
    random.shuffle(customers)
    routes = [[0, 0] for _ in range(truck_count)]
    for cust in customers:
        best_truck = None
        best_pos = None
        best_max = float('inf')
        best_total = float('inf')
        for t, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_routes = routes[:t] + [new_route] + routes[t+1:]
                new_max = max(route_distance(r) for r in new_routes)
                new_total = sum(route_distance(r) for r in new_routes)
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = pos
        routes[best_truck].insert(best_pos, cust)

    current_routes = [list(r) for r in routes]
    best_routes = [list(r) for r in routes]
    best_max = max(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)

    # ALNS parameters
    max_iter = 100 * n
    removal_fraction = 0.3
    num_removals = max(1, int(removal_fraction * (n-1)))
    T0 = best_max / 2.0
    T = T0

    # Adaptive weights: scores (rewards for success) and usage counters for exploration
    destroy_scores = [1.0, 1.0]
    repair_scores = [1.0, 1.0]
    destroy_usage = [1, 1]
    repair_usage = [1, 1]
    score_increase = 1.0
    score_decrease = 0.5
    min_score = 0.1
    max_score = 10.0

    # Elite set
    elite_set = []
    elite_size = 5

    def add_to_elite(routes, max_val):
        nonlocal elite_set
        if len(elite_set) < elite_size:
            elite_set.append((max_val, [list(r) for r in routes]))
            elite_set.sort(key=lambda x: x[0])
        elif max_val < elite_set[-1][0]:
            elite_set.pop()
            elite_set.append((max_val, [list(r) for r in routes]))
            elite_set.sort(key=lambda x: x[0])

    add_to_elite(best_routes, best_max)

    def worst_removal(routes, num_removals):
        all_contribs = []
        for t, route in enumerate(routes):
            if len(route) <= 2:
                continue
            for i in range(1, len(route)-1):
                prev = route[i-1]
                nxt = route[i+1]
                with_ = dist[prev, route[i]] + dist[route[i], nxt]
                without = dist[prev, nxt]
                contrib = with_ - without
                all_contribs.append((-contrib, t, i, route[i]))
        all_contribs.sort(key=lambda x: x[0])
        to_remove = set()
        for _, t, i, cust in all_contribs[:num_removals]:
            to_remove.add(cust)
        new_routes = []
        for t, route in enumerate(routes):
            new_routes.append([0] + [c for c in route[1:-1] if c not in to_remove] + [0])
        return list(to_remove), new_routes

    def random_removal(routes, num_removals):
        all_customers = [c for r in routes for c in r[1:-1]]
        random.shuffle(all_customers)
        to_remove = set(all_customers[:num_removals])
        new_routes = []
        for route in routes:
            new_routes.append([0] + [c for c in route[1:-1] if c not in to_remove] + [0])
        return list(to_remove), new_routes

    def greedy_repair(routes, unassigned):
        routes = [list(r) for r in routes]
        unassigned = list(unassigned)
        for cust in unassigned:
            best_max = float('inf')
            best_total = float('inf')
            best_truck = None
            best_pos = None
            for t, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_routes = routes[:t] + [new_route] + routes[t+1:]
                    new_max_val = max(route_distance(r) for r in new_routes)
                    new_total_val = sum(route_distance(r) for r in new_routes)
                    if new_max_val < best_max or (new_max_val == best_max and new_total_val < best_total):
                        best_max = new_max_val
                        best_total = new_total_val
                        best_truck = t
                        best_pos = pos
            routes[best_truck].insert(best_pos, cust)
        return routes

    def regret2_repair(routes, unassigned):
        routes = [list(r) for r in routes]
        unassigned = list(unassigned)
        while unassigned:
            best_info = None
            for cust in unassigned:
                best_max = float('inf')
                best_total = float('inf')
                second_best_max = float('inf')
                second_best_total = float('inf')
                best_truck = None
                best_pos = None
                for t, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_routes = routes[:t] + [new_route] + routes[t+1:]
                        new_max_val = max(route_distance(r) for r in new_routes)
                        new_total_val = sum(route_distance(r) for r in new_routes)
                        if new_max_val < best_max or (new_max_val == best_max and new_total_val < best_total):
                            second_best_max = best_max
                            second_best_total = best_total
                            best_max = new_max_val
                            best_total = new_total_val
                            best_truck = t
                            best_pos = pos
                        elif new_max_val < second_best_max or (new_max_val == second_best_max and new_total_val < second_best_total):
                            second_best_max = new_max_val
                            second_best_total = new_total_val
                if second_best_max == float('inf'):
                    regret = float('inf')
                else:
                    regret = second_best_max - best_max
                if best_info is None or (regret > best_info[0] or (regret == best_info[0] and (best_max > best_info[1] or (best_max == best_info[1] and cust < best_info[2])))):
                    best_info = (regret, best_max, cust, best_truck, best_pos)
            _, _, cust, best_truck, best_pos = best_info
            routes[best_truck].insert(best_pos, cust)
            unassigned.remove(cust)
        return routes

    no_improve_iter = 0
    restart_threshold = int(0.15 * max_iter)
    elite_inject_interval = int(0.05 * max_iter)
    iter_since_elite_inject = 0

    for it in range(max_iter):
        # Operator selection: combine scores and usage (inverse usage for exploration)
        # We use product of score and 1/usage as weight
        total_d = 0
        destroy_weights = []
        for i in range(2):
            w = destroy_scores[i] / destroy_usage[i]
            destroy_weights.append(w)
            total_d += w
        destroy_probs = [w / total_d for w in destroy_weights]

        total_r = 0
        repair_weights = []
        for i in range(2):
            w = repair_scores[i] / repair_usage[i]
            repair_weights.append(w)
            total_r += w
        repair_probs = [w / total_r for w in repair_weights]

        destroy_op = random.choices([0, 1], weights=destroy_probs)[0]
        repair_op = random.choices([0, 1], weights=repair_probs)[0]

        if destroy_op == 0:
            removed, partial = worst_removal(current_routes, num_removals)
        else:
            removed, partial = random_removal(current_routes, num_removals)

        if repair_op == 0:
            new_routes = greedy_repair(partial, removed)
        else:
            new_routes = regret2_repair(partial, removed)

        new_max = max(route_distance(r) for r in new_routes)
        new_total = sum(route_distance(r) for r in new_routes)
        current_max = max(route_distance(r) for r in current_routes)
        current_total = sum(route_distance(r) for r in current_routes)

        delta = new_max - current_max
        if delta < 0 or (delta == 0 and new_total < current_total) or random.random() < math.exp(-delta / max(T, 1e-9)):
            current_routes = [list(r) for r in new_routes]
            # Update scores for successful operators
            destroy_scores[destroy_op] = min(max_score, destroy_scores[destroy_op] + score_increase)
            repair_scores[repair_op] = min(max_score, repair_scores[repair_op] + score_increase)
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < sum(route_distance(r) for r in best_routes)):
                best_max = new_max
                best_routes = [list(r) for r in new_routes]
                report_best_vrp(best_routes)
                add_to_elite(new_routes, new_max)
                no_improve_iter = 0
            else:
                no_improve_iter += 1
        else:
            destroy_scores[destroy_op] = max(min_score, destroy_scores[destroy_op] - score_decrease)
            repair_scores[repair_op] = max(min_score, repair_scores[repair_op] - score_decrease)
            no_improve_iter += 1

        # Update usage counters
        destroy_usage[destroy_op] += 1
        repair_usage[repair_op] += 1

        # Temperature update
        T = T0 * (1 - it / max_iter)

        iter_since_elite_inject += 1
        # Elite injection: combine two elite solutions
        if iter_since_elite_inject >= elite_inject_interval and len(elite_set) >= 2:
            iter_since_elite_inject = 0
            # pick two different elite solutions
            idx1 = random.randint(0, len(elite_set)-1)
            idx2 = random.randint(0, len(elite_set)-1)
            while idx2 == idx1:
                idx2 = random.randint(0, len(elite_set)-1)
            _, routes1 = elite_set[idx1]
            _, routes2 = elite_set[idx2]
            # build a new solution by taking the union of customers from both, but we need a feasible solution
            # Instead, we do a path-relinking: for each customer, randomly choose which truck it goes to from one of the two elite routes
            # Then repair to ensure feasibility? Simpler: randomly select a truck assignment from either solution then run greedy repair
            # Actually, better: create a partial solution by copying a random subset of customers from each, then complete with ALNS
            # For simplicity, we do a random shuffle of customers and then greedy insertion (like restart) but using the best so far
            # We'll just perform a large perturbation by random removal and reinsertion
            large_removal_count = max(1, int(0.4 * (n-1)))
            removed, partial = random_removal(current_routes, large_removal_count)
            current_routes = greedy_repair(partial, removed)
            T = T0  # reset temperature
            no_improve_iter = 0

        # Restart if completely stuck
        if no_improve_iter >= restart_threshold:
            large_removal_count = max(1, int(0.5 * (n-1)))
            removed, partial = random_removal(current_routes, large_removal_count)
            current_routes = greedy_repair(partial, removed)
            T = T0
            no_improve_iter = 0

    # Final local search: relocate and swap with first improvement limited passes
    routes = [list(r) for r in best_routes]
    route_dists = [route_distance(r) for r in routes]
    max_dist = max(route_dists)
    max_passes = n * truck_count
    for _ in range(max_passes):
        improved = False
        # Relocate
        for t_from in range(truck_count):
            if not routes[t_from]:
                continue
            for i in range(len(routes[t_from])):
                cust = routes[t_from][i]
                for t_to in range(truck_count):
                    if t_to == t_from:
                        continue
                    for j in range(len(routes[t_to])+1):
                        new_route_from = routes[t_from][:i] + routes[t_from][i+1:]
                        new_dist_from = route_distance(new_route_from)
                        new_route_to = routes[t_to][:j] + [cust] + routes[t_to][j:]
                        new_dist_to = route_distance(new_route_to)
                        other_max = max(route_dists[t] for t in range(truck_count) if t != t_from and t != t_to)
                        new_max = max(new_dist_from, new_dist_to, other_max)
                        if new_max < max_dist - 1e-9:
                            routes[t_from] = new_route_from
                            routes[t_to] = new_route_to
                            route_dists[t_from] = new_dist_from
                            route_dists[t_to] = new_dist_to
                            max_dist = new_max
                            full_routes = [[0] + r + [0] for r in routes]
                            report_best_vrp(full_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Swap
        for t1 in range(truck_count):
            if not routes[t1]:
                continue
            for i in range(len(routes[t1])):
                for t2 in range(t1+1, truck_count):
                    if not routes[t2]:
                        continue
                    for j in range(len(routes[t2])):
                        new_route1 = routes[t1].copy()
                        new_route2 = routes[t2].copy()
                        new_route1[i], new_route2[j] = new_route2[j], new_route1[i]
                        new_dist1 = route_distance(new_route1)
                        new_dist2 = route_distance(new_route2)
                        other_max = max(route_dists[t] for t in range(truck_count) if t != t1 and t != t2)
                        new_max = max(new_dist1, new_dist2, other_max)
                        if new_max < max_dist - 1e-9:
                            routes[t1] = new_route1
                            routes[t2] = new_route2
                            route_dists[t1] = new_dist1
                            route_dists[t2] = new_dist2
                            max_dist = new_max
                            full_routes = [[0] + r + [0] for r in routes]
                            report_best_vrp(full_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    final_routes = [[0] + r + [0] for r in routes]
    return final_routes