import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)

    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def insert_cost(route, node):
        best_cost = float('inf')
        best_pos = -1
        for i in range(1, len(route)):
            cost = distance_matrix[route[i-1], node] + distance_matrix[node, route[i]] - distance_matrix[route[i-1], route[i]]
            if cost < best_cost:
                best_cost = cost
                best_pos = i
        return best_cost, best_pos

    def compute_penalty(current_dist, lambda_bal):
        return [lambda_bal * (dist ** 2) for dist in current_dist]

    def perturb_solution(routes, current_dist, removal_fraction):
        # Remove from longest and shortest routes
        route_lengths = [len(r) for r in routes]
        if min(route_lengths) < 3:
            return routes, current_dist, []
        sorted_indices = sorted(range(truck_count), key=lambda i: current_dist[i])
        longest_idx = sorted_indices[-1]
        shortest_idx = sorted_indices[0]
        removed = []
        # remove from longest
        remove_count_long = max(1, int(removal_fraction * (len(routes[longest_idx]) - 2)))
        if remove_count_long > len(routes[longest_idx]) - 2:
            remove_count_long = len(routes[longest_idx]) - 2
        if remove_count_long > 0 and len(routes[longest_idx]) > 2:
            candidates = list(range(1, len(routes[longest_idx])-1))
            random.shuffle(candidates)
            for idx in candidates[:remove_count_long]:
                cust = routes[longest_idx][idx]
                removed.append(cust)
            new_route = [0]
            for node in routes[longest_idx][1:-1]:
                if node not in removed:
                    new_route.append(node)
            new_route.append(0)
            routes[longest_idx] = new_route
        # remove from shortest
        remove_count_short = max(1, int(removal_fraction * (len(routes[shortest_idx]) - 2)))
        if remove_count_short > len(routes[shortest_idx]) - 2:
            remove_count_short = len(routes[shortest_idx]) - 2
        if remove_count_short > 0 and len(routes[shortest_idx]) > 2:
            candidates = list(range(1, len(routes[shortest_idx])-1))
            random.shuffle(candidates)
            for idx in candidates[:remove_count_short]:
                cust = routes[shortest_idx][idx]
                removed.append(cust)
            new_route = [0]
            for node in routes[shortest_idx][1:-1]:
                if node not in removed:
                    new_route.append(node)
            new_route.append(0)
            routes[shortest_idx] = new_route
        for r in range(truck_count):
            current_dist[r] = route_distance(routes[r])
        return routes, current_dist, removed

    def repair_regret2(routes, current_dist, unassigned):
        lambda_balance = 0.5
        while unassigned:
            best_cust = None
            best_regret = -1.0
            best_route_idx = None
            best_pos = None
            best_cost_val = None
            for cust in unassigned:
                costs_info = []
                for r in range(truck_count):
                    base_cost, pos = insert_cost(routes[r], cust)
                    pen = lambda_balance * (current_dist[r] ** 2)
                    total_cost = base_cost + pen
                    costs_info.append((total_cost, r, pos, base_cost))
                costs_info.sort(key=lambda x: x[0])
                best_cost = costs_info[0][0]
                second_best = costs_info[1][0] if len(costs_info) >= 2 else best_cost
                regret = second_best - best_cost
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = costs_info[0][1]
                    best_pos = costs_info[0][2]
                    best_cost_val = costs_info[0][3]
            routes[best_route_idx].insert(best_pos, best_cust)
            current_dist[best_route_idx] += best_cost_val
            unassigned.remove(best_cust)
            max_dist = max(current_dist)
            avg_dist = sum(current_dist) / truck_count
            imbalance = max_dist - avg_dist
            lambda_balance = min(1.0, max(0.1, imbalance / max(avg_dist, 1e-9)))
        return routes, current_dist

    def intra_route_2opt(routes, current_dist):
        improved = True
        while improved:
            improved = False
            for r in range(truck_count):
                route = routes[r]
                if len(route) <= 3:
                    continue
                best_improve = 0
                best_i = best_j = -1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < current_dist[r] - 1e-9:
                            improvement = current_dist[r] - new_dist
                            if improvement > best_improve:
                                best_improve = improvement
                                best_i, best_j = i, j
                if best_improve > 0:
                    routes[r] = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                    current_dist[r] = route_distance(routes[r])
                    improved = True
        return routes, current_dist

    global_best_max = float('inf')
    global_best_routes = None
    num_restarts = 30
    no_improve_restarts = 0
    elite_solutions = []
    visited = set()

    for restart in range(num_restarts):
        customers = list(range(1, n))
        if no_improve_restarts >= 5 and elite_solutions:
            # reinitialize from elite
            elite_routes = random.choice(elite_solutions)
            deep_routes = [list(r) for r in elite_routes]
            # apply random perturbation
            removal_fraction = 0.4
            deep_routes, current_dist, removed = perturb_solution(deep_routes, [route_distance(r) for r in deep_routes], removal_fraction)
            unassigned = removed
            random.shuffle(unassigned)
            deep_routes, current_dist = repair_regret2(deep_routes, current_dist, unassigned)
            routes = deep_routes
            current_dist = [route_distance(r) for r in routes]
        else:
            random.shuffle(customers)
            routes = [[0, 0] for _ in range(truck_count)]
            current_dist = [0.0 for _ in range(truck_count)]
            unassigned = customers[:]
            lambda_balance = 0.5
            while unassigned:
                best_cust = None
                best_regret = -1.0
                best_route_idx = None
                best_pos = None
                best_cost_val = None
                for cust in unassigned:
                    costs_info = []
                    for r in range(truck_count):
                        base_cost, pos = insert_cost(routes[r], cust)
                        pen = lambda_balance * (current_dist[r] ** 2)
                        total_cost = base_cost + pen
                        costs_info.append((total_cost, r, pos, base_cost))
                    costs_info.sort(key=lambda x: x[0])
                    best_cost = costs_info[0][0]
                    second_best = costs_info[1][0] if len(costs_info) >= 2 else best_cost
                    regret = second_best - best_cost
                    if regret > best_regret:
                        best_regret = regret
                        best_cust = cust
                        best_route_idx = costs_info[0][1]
                        best_pos = costs_info[0][2]
                        best_cost_val = costs_info[0][3]
                routes[best_route_idx].insert(best_pos, best_cust)
                current_dist[best_route_idx] += best_cost_val
                unassigned.remove(best_cust)
                max_dist = max(current_dist)
                avg_dist = sum(current_dist) / truck_count
                imbalance = max_dist - avg_dist
                lambda_balance = min(1.0, max(0.1, imbalance / max(avg_dist, 1e-9)))

        # Post-construction imbalance reduction: relocate from longest to shortest
        for _ in range(5):
            max_idx = max(range(truck_count), key=lambda i: current_dist[i])
            min_idx = min(range(truck_count), key=lambda i: current_dist[i])
            if current_dist[max_idx] - current_dist[min_idx] < 1e-6:
                break
            best_improvement = 0
            best_cust = None
            best_pos_new = None
            best_pos_old = None
            route_long = routes[max_idx]
            route_short = routes[min_idx]
            for idx in range(1, len(route_long)-1):
                cust = route_long[idx]
                new_long = route_long[:idx] + route_long[idx+1:]
                new_long_dist = route_distance(new_long)
                cost, pos = insert_cost(route_short, cust)
                new_short_dist = current_dist[min_idx] + cost
                other_dists = [current_dist[k] for k in range(truck_count) if k not in (max_idx, min_idx)]
                new_max = max(new_long_dist, new_short_dist, *other_dists)
                if new_max < max(current_dist):
                    improvement = max(current_dist) - new_max
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_cust = cust
                        best_pos_old = idx
                        best_pos_new = pos
            if best_cust is not None:
                routes[max_idx] = routes[max_idx][:best_pos_old] + routes[max_idx][best_pos_old+1:]
                routes[min_idx] = routes[min_idx][:best_pos_new] + [best_cust] + routes[min_idx][best_pos_new:]
                current_dist[max_idx] = route_distance(routes[max_idx])
                current_dist[min_idx] = route_distance(routes[min_idx])
            else:
                break

        best_max = max(current_dist)
        local_best_routes = [list(r) for r in routes]
        if best_max < global_best_max:
            global_best_max = best_max
            global_best_routes = [list(r) for r in routes]
            report_best_vrp(routes)
            no_improve_restarts = 0
            elite_solutions.append([tuple(r) for r in routes])
            if len(elite_solutions) > 3:
                elite_solutions.sort(key=lambda s: max(route_distance(list(r)) for r in s))
                elite_solutions = elite_solutions[:3]
        else:
            no_improve_restarts += 1

        # Perturbation and improvement cycles
        removal_fraction = 0.4
        for cycle in range(5):
            # Perturb
            routes, current_dist, removed = perturb_solution(routes, current_dist, removal_fraction)
            if not removed:
                break
            unassigned = removed[:]
            random.shuffle(unassigned)
            routes, current_dist = repair_regret2(routes, current_dist, unassigned)
            # Additional intra-route 2-opt pass
            routes, current_dist = intra_route_2opt(routes, current_dist)
            # Local search with best-improvement and tabu
            local_visited = set()
            max_iters = 10 * (n - 1) * truck_count
            improved = True
            iters = 0
            while improved and iters < max_iters:
                improved = False
                iters += 1
                # Relocate best-improvement
                best_move = None
                best_new_max = float('inf')
                for r1 in range(truck_count):
                    route1 = routes[r1]
                    if len(route1) <= 2:
                        continue
                    for idx in range(1, len(route1)-1):
                        cust = route1[idx]
                        new_route1 = route1[:idx] + route1[idx+1:]
                        new_dist1 = route_distance(new_route1)
                        for r2 in range(truck_count):
                            if r2 == r1:
                                continue
                            route2 = routes[r2]
                            cost, pos = insert_cost(route2, cust)
                            new_dist2 = current_dist[r2] + cost
                            other_dists = [current_dist[i] for i in range(truck_count) if i not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_max and new_max < best_new_max:
                                # check tabu
                                new_routes = [list(r) for r in routes]
                                new_routes[r1] = new_route1
                                new_routes[r2] = route2[:pos] + [cust] + route2[pos:]
                                hash_key = tuple(tuple(r) for r in new_routes)
                                if hash_key not in visited and hash_key not in local_visited:
                                    best_new_max = new_max
                                    best_move = (r1, idx, r2, pos, new_route1, new_dist1, new_dist2, new_routes)
                if best_move is not None and best_new_max < best_max:
                    r1, idx, r2, pos, new_route1, new_dist1, new_dist2, new_routes = best_move
                    routes = new_routes
                    current_dist[r1] = new_dist1
                    current_dist[r2] = new_dist2
                    best_max = best_new_max
                    local_best_routes = [list(r) for r in routes]
                    if best_max < global_best_max:
                        global_best_max = best_max
                        global_best_routes = [list(r) for r in routes]
                        report_best_vrp(routes)
                        elite_solutions.append([tuple(r) for r in routes])
                        if len(elite_solutions) > 3:
                            elite_solutions.sort(key=lambda s: max(route_distance(list(r)) for r in s))
                            elite_solutions = elite_solutions[:3]
                    visited.add(tuple(tuple(r) for r in routes))
                    local_visited.add(tuple(tuple(r) for r in routes))
                    improved = True
                    continue

                # Swap best-improvement
                best_move = None
                best_new_max = float('inf')
                for r1 in range(truck_count):
                    route1 = routes[r1]
                    if len(route1) <= 2:
                        continue
                    for idx1 in range(1, len(route1)-1):
                        cust1 = route1[idx1]
                        for r2 in range(r1+1, truck_count):
                            route2 = routes[r2]
                            if len(route2) <= 2:
                                continue
                            for idx2 in range(1, len(route2)-1):
                                cust2 = route2[idx2]
                                new_route1 = route1[:idx1] + [cust2] + route1[idx1+1:]
                                new_route2 = route2[:idx2] + [cust1] + route2[idx2+1:]
                                new_dist1 = route_distance(new_route1)
                                new_dist2 = route_distance(new_route2)
                                other_dists = [current_dist[i] for i in range(truck_count) if i not in (r1, r2)]
                                new_max = max(new_dist1, new_dist2, *other_dists)
                                if new_max < best_max and new_max < best_new_max:
                                    new_routes = [list(r) for r in routes]
                                    new_routes[r1] = new_route1
                                    new_routes[r2] = new_route2
                                    hash_key = tuple(tuple(r) for r in new_routes)
                                    if hash_key not in visited and hash_key not in local_visited:
                                        best_new_max = new_max
                                        best_move = (new_routes, r1, r2, new_dist1, new_dist2)
                if best_move is not None and best_new_max < best_max:
                    new_routes, r1, r2, new_dist1, new_dist2 = best_move
                    routes = new_routes
                    current_dist[r1] = new_dist1
                    current_dist[r2] = new_dist2
                    best_max = best_new_max
                    local_best_routes = [list(r) for r in routes]
                    if best_max < global_best_max:
                        global_best_max = best_max
                        global_best_routes = [list(r) for r in routes]
                        report_best_vrp(routes)
                        elite_solutions.append([tuple(r) for r in routes])
                        if len(elite_solutions) > 3:
                            elite_solutions.sort(key=lambda s: max(route_distance(list(r)) for r in s))
                            elite_solutions = elite_solutions[:3]
                    visited.add(tuple(tuple(r) for r in routes))
                    local_visited.add(tuple(tuple(r) for r in routes))
                    improved = True
                    continue

                # Intra-route 2-opt best-improvement
                best_improve = 0
                best_i = best_j = -1
                best_r = -1
                for r in range(truck_count):
                    route = routes[r]
                    if len(route) <= 3:
                        continue
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            new_dist = route_distance(new_route)
                            if new_dist < current_dist[r] - 1e-9:
                                improvement = current_dist[r] - new_dist
                                if improvement > best_improve:
                                    best_improve = improvement
                                    best_i, best_j = i, j
                                    best_r = r
                if best_improve > 0:
                    r = best_r
                    route = routes[r]
                    new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                    routes[r] = new_route
                    current_dist[r] = route_distance(new_route)
                    new_max = max(current_dist)
                    if new_max < best_max:
                        best_max = new_max
                        local_best_routes = [list(r) for r in routes]
                        if best_max < global_best_max:
                            global_best_max = best_max
                            global_best_routes = [list(r) for r in routes]
                            report_best_vrp(routes)
                            elite_solutions.append([tuple(r) for r in routes])
                            if len(elite_solutions) > 3:
                                elite_solutions.sort(key=lambda s: max(route_distance(list(r)) for r in s))
                                elite_solutions = elite_solutions[:3]
                    visited.add(tuple(tuple(r) for r in routes))
                    local_visited.add(tuple(tuple(r) for r in routes))
                    improved = True
                    continue

                # Cross-route 2-opt best-improvement
                best_move = None
                best_new_max = float('inf')
                for r1 in range(truck_count):
                    route1 = routes[r1]
                    if len(route1) <= 2:
                        continue
                    for r2 in range(r1+1, truck_count):
                        route2 = routes[r2]
                        if len(route2) <= 2:
                            continue
                        for i in range(1, len(route1)-1):
                            for j in range(1, len(route2)-1):
                                new1 = route1[:i+1] + route2[j+1:]
                                new2 = route2[:j+1] + route1[i+1:]
                                new_dist1 = route_distance(new1)
                                new_dist2 = route_distance(new2)
                                other_dists = [current_dist[k] for k in range(truck_count) if k not in (r1, r2)]
                                new_max = max(new_dist1, new_dist2, *other_dists)
                                if new_max < best_max and new_max < best_new_max:
                                    new_routes = [list(r) for r in routes]
                                    new_routes[r1] = new1
                                    new_routes[r2] = new2
                                    hash_key = tuple(tuple(r) for r in new_routes)
                                    if hash_key not in visited and hash_key not in local_visited:
                                        best_new_max = new_max
                                        best_move = (new_routes, r1, r2, new_dist1, new_dist2)
                if best_move is not None and best_new_max < best_max:
                    new_routes, r1, r2, new_dist1, new_dist2 = best_move
                    routes = new_routes
                    current_dist[r1] = new_dist1
                    current_dist[r2] = new_dist2
                    best_max = best_new_max
                    local_best_routes = [list(r) for r in routes]
                    if best_max < global_best_max:
                        global_best_max = best_max
                        global_best_routes = [list(r) for r in routes]
                        report_best_vrp(routes)
                        elite_solutions.append([tuple(r) for r in routes])
                        if len(elite_solutions) > 3:
                            elite_solutions.sort(key=lambda s: max(route_distance(list(r)) for r in s))
                            elite_solutions = elite_solutions[:3]
                    visited.add(tuple(tuple(r) for r in routes))
                    local_visited.add(tuple(tuple(r) for r in routes))
                    improved = True
                    continue

                if not improved:
                    break

            removal_fraction *= 0.95
            if best_max < global_best_max:
                global_best_max = best_max
                global_best_routes = [list(r) for r in routes]
                report_best_vrp(global_best_routes)

    return global_best_routes