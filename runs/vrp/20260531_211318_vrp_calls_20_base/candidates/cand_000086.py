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
        for i in range(len(route) - 1):
            d += dist[route[i], route[i + 1]]
        return d

    # Randomized farthest insertion: probabilistic selection based on insertion cost
    def randomized_farthest_insertion(randomize=True):
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        customers = sorted(range(1, n), key=lambda c: -dist[0][c])
        remaining = list(customers)
        while remaining:
            # among remaining, select the one farthest from depot (or random subset?)
            farthest = remaining[0]
            # compute insertion costs for this customer
            costs = []
            for t, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [farthest] + route[pos:]
                    new_dist = route_distance(new_route)
                    new_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                    new_total = sum(route_dists[:t]) + new_dist + sum(route_dists[t+1:])
                    costs.append((new_max, new_total, t, pos, new_dist))
            # select probabilistically using softmax if randomize
            if randomize:
                weights = [math.exp(-c[0]/max(1e-9, route_dists[0] if route_dists[0]>0 else 1.0)) for c in costs]
                total_w = sum(weights)
                if total_w == 0:
                    best_idx = 0
                else:
                    r = random.random() * total_w
                    cum = 0.0
                    best_idx = 0
                    for i, w in enumerate(weights):
                        cum += w
                        if r <= cum:
                            best_idx = i
                            break
                chosen = costs[best_idx]
            else:
                # deterministic: choose min max, then min total
                chosen = min(costs, key=lambda x: (x[0], x[1]))
            t = chosen[2]
            pos = chosen[3]
            new_dist = chosen[4]
            routes[t].insert(pos, farthest)
            route_dists[t] = new_dist
            remaining.pop(0)
        return routes, route_dists

    # Initial solution
    routes, route_dists = randomized_farthest_insertion(randomize=True)
    current_routes = [list(r) for r in routes]
    current_dists = list(route_dists)
    current_max = max(current_dists)
    current_total = sum(current_dists)
    best_routes = [list(r) for r in routes]
    best_dists = list(route_dists)
    best_max = current_max
    best_total = current_total
    report_best_vrp(best_routes)

    # ALNS parameters
    max_iter = min(5000, 30 * n)
    removal_fraction = 0.3  # increased for diversity
    num_removals = max(1, int(removal_fraction * (n - 1)))
    T0 = best_max / 2.0
    T = T0
    # Adaptive operator weights
    destroy_weights = [1.0, 1.0]  # 0: worst, 1: random
    repair_weights = [1.0, 1.0]   # 0: greedy, 1: regret-2
    destroy_scores = [0.0, 0.0]
    repair_scores = [0.0, 0.0]
    scores_used = 0
    # Restart tracking
    it_since_best = 0
    restart_period = 200

    # Helper functions (same as parent but modified for adaptation)
    def worst_removal(routes, dists, num_removals):
        all_contribs = []
        for t, route in enumerate(routes):
            if len(route) <= 2:
                continue
            base = dists[t]
            for i in range(1, len(route) - 1):
                prev = route[i - 1]
                nxt = route[i + 1]
                with_ = dist[prev, route[i]] + dist[route[i], nxt]
                without = dist[prev, nxt]
                contrib = with_ - without
                all_contribs.append((-contrib, t, i, route[i]))
        all_contribs.sort(key=lambda x: x[0])
        to_remove = set()
        for _, t, i, cust in all_contribs[:num_removals]:
            to_remove.add(cust)
        new_routes = []
        new_dists = []
        for t, route in enumerate(routes):
            new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
            new_routes.append(new_route)
            new_dists.append(route_distance(new_route))
        return list(to_remove), new_routes, new_dists

    def random_removal(routes, dists, num_removals):
        all_customers = [c for r in routes for c in r[1:-1]]
        random.shuffle(all_customers)
        to_remove = set(all_customers[:num_removals])
        new_routes = []
        new_dists = []
        for route in routes:
            new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
            new_routes.append(new_route)
            new_dists.append(route_distance(new_route))
        return list(to_remove), new_routes, new_dists

    def greedy_repair(routes, dists, unassigned):
        routes = [list(r) for r in routes]
        dists = list(dists)
        unassigned = list(unassigned)
        current_max_local = max(dists)
        for cust in unassigned:
            best_truck = None
            best_pos = None
            best_new_max = float('inf')
            best_new_total = float('inf')
            for t, route in enumerate(routes):
                old_dist = dists[t]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = route_distance(new_route)
                    delta_dist = new_dist - old_dist
                    new_max = max(current_max_local, new_dist)
                    new_total = sum(dists) + delta_dist
                    if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                        best_new_max = new_max
                        best_new_total = new_total
                        best_truck = t
                        best_pos = pos
                        best_new_dist = new_dist
            routes[best_truck].insert(best_pos, cust)
            dists[best_truck] = best_new_dist
            if best_new_dist > current_max_local:
                current_max_local = best_new_dist
        return routes, dists

    def regret2_repair(routes, dists, unassigned):
        routes = [list(r) for r in routes]
        dists = list(dists)
        unassigned = list(unassigned)
        current_max_local = max(dists)
        while unassigned:
            best_info = None
            for cust in unassigned:
                best_max = float('inf')
                best_total = float('inf')
                best_truck = None
                best_pos = None
                second_best_max = float('inf')
                second_best_total = float('inf')
                for t, route in enumerate(routes):
                    old_dist = dists[t]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_dist = route_distance(new_route)
                        delta_dist = new_dist - old_dist
                        new_max = max(current_max_local, new_dist)
                        new_total = sum(dists) + delta_dist
                        if new_max < best_max or (new_max == best_max and new_total < best_total):
                            second_best_max = best_max
                            second_best_total = best_total
                            best_max = new_max
                            best_total = new_total
                            best_truck = t
                            best_pos = pos
                            best_delta = delta_dist
                        elif new_max < second_best_max or (new_max == second_best_max and new_total < second_best_total):
                            second_best_max = new_max
                            second_best_total = new_total
                if second_best_max == float('inf'):
                    regret = float('inf')
                else:
                    regret = second_best_max - best_max
                if best_info is None:
                    best_info = (regret, best_max, best_total, cust, best_truck, best_pos, best_delta)
                else:
                    if regret > best_info[0] or (regret == best_info[0] and (
                        best_max < best_info[1] or (best_max == best_info[1] and cust < best_info[3]))):
                        best_info = (regret, best_max, best_total, cust, best_truck, best_pos, best_delta)
            regret, best_max, best_total, cust, best_truck, best_pos, best_delta = best_info
            routes[best_truck].insert(best_pos, cust)
            dists[best_truck] += best_delta
            if dists[best_truck] > current_max_local:
                current_max_local = dists[best_truck]
            unassigned.remove(cust)
        return routes, dists

    # Adaptive operator selection
    def select_operator(weights):
        total = sum(weights)
        r = random.random() * total
        cum = 0.0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                return i
        return len(weights)-1

    def update_weights(weights, scores, used):
        for i in range(len(weights)):
            if used[i] > 0:
                weights[i] = weights[i] * 0.9 + 0.1 * (scores[i] / used[i] if used[i] > 0 else 0.0)
        # reset scores and used
        return [1.0]*len(weights), [0.0]*len(weights), [0]*len(weights)

    used_destroy = [0, 0]
    used_repair = [0, 0]

    for it in range(max_iter):
        # Check for restart
        it_since_best += 1
        if it_since_best >= restart_period:
            # restart: perturb best solution
            # Remove 30% customers randomly from best solution and reinsert with randomized greedy
            all_customers = [c for r in best_routes for c in r[1:-1]]
            random.shuffle(all_customers)
            num_remove = max(1, int(0.3 * (n-1)))
            removed_set = set(all_customers[:num_remove])
            new_routes = []
            new_dists = []
            for t, route in enumerate(best_routes):
                new_route = [0] + [c for c in route[1:-1] if c not in removed_set] + [0]
                new_routes.append(new_route)
                new_dists.append(route_distance(new_route))
            # reinsert with randomized greedy (probabilistic)
            unassigned = list(removed_set)
            # use greedy but with randomness: select insertion as before but choose randomly among top 3
            for cust in unassigned:
                candidates = []
                for t, route in enumerate(new_routes):
                    for pos in range(1, len(route)):
                        new_route_try = route[:pos] + [cust] + route[pos:]
                        new_dist = route_distance(new_route_try)
                        new_max = max(new_dists[:t] + [new_dist] + new_dists[t+1:])
                        new_total = sum(new_dists[:t]) + new_dist + sum(new_dists[t+1:])
                        candidates.append((new_max, new_total, t, pos, new_dist))
                # select probabilistically from top 3 by max
                candidates.sort(key=lambda x: (x[0], x[1]))
                top = candidates[:min(3, len(candidates))]
                weights = [math.exp(-c[0]/max(1e-9, T0)) for c in top]
                total_w = sum(weights)
                if total_w == 0:
                    chosen = top[0]
                else:
                    r = random.random() * total_w
                    cum = 0.0
                    chosen = top[-1]
                    for i, w in enumerate(weights):
                        cum += w
                        if r <= cum:
                            chosen = top[i]
                            break
                t = chosen[2]
                pos = chosen[3]
                new_dist = chosen[4]
                new_routes[t].insert(pos, cust)
                new_dists[t] = new_dist
            # update current
            current_routes = new_routes
            current_dists = new_dists
            current_max = max(current_dists)
            current_total = sum(current_dists)
            it_since_best = 0
            continue

        # Select destroy and repair operators
        destroy_idx = select_operator(destroy_weights)
        repair_idx = select_operator(repair_weights)
        used_destroy[destroy_idx] += 1
        used_repair[repair_idx] += 1
        scores_used += 1

        if destroy_idx == 0:
            removed, partial, partial_dists = worst_removal(current_routes, current_dists, num_removals)
        else:
            removed, partial, partial_dists = random_removal(current_routes, current_dists, num_removals)

        if repair_idx == 0:
            new_routes, new_dists = greedy_repair(partial, partial_dists, removed)
        else:
            new_routes, new_dists = regret2_repair(partial, partial_dists, removed)

        new_max = max(new_dists)
        new_total = sum(new_dists)
        delta = new_max - current_max
        accept = False
        if delta < 0 or (delta == 0 and new_total < current_total) or random.random() < math.exp(-delta / max(T, 1e-9)):
            accept = True
            current_routes = [list(r) for r in new_routes]
            current_dists = list(new_dists)
            current_max = new_max
            current_total = new_total
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                best_max = new_max
                best_total = new_total
                best_routes = [list(r) for r in new_routes]
                best_dists = list(new_dists)
                report_best_vrp(best_routes)
                it_since_best = 0
        # Update scores based on acceptance and improvement
        if accept:
            if new_max < best_max:
                destroy_scores[destroy_idx] += 2.0
                repair_scores[repair_idx] += 2.0
            else:
                destroy_scores[destroy_idx] += 0.5
                repair_scores[repair_idx] += 0.5
        else:
            destroy_scores[destroy_idx] += 0.0
            repair_scores[repair_idx] += 0.0

        # Update weights periodically
        if scores_used >= 50:
            destroy_weights, destroy_scores, used_destroy = update_weights(destroy_weights, destroy_scores, used_destroy)
            repair_weights, repair_scores, used_repair = update_weights(repair_weights, repair_scores, used_repair)
            scores_used = 0

        T = T0 * (1 - it / max_iter)

    # Post-optimization: 2-opt
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
                        old_other = sum(best_dists[:t]) + sum(best_dists[t+1:])
                        new_total = old_other + new_dist
                        new_max = max(best_dists[:t] + [new_dist] + best_dists[t+1:])
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