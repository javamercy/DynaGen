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

    global_best_max = float('inf')
    global_best_routes = None
    num_restarts = 30
    no_improve_restarts = 0

    for restart in range(num_restarts):
        customers = list(range(1, n))
        if restart >= 5 and no_improve_restarts >= 5:
            random.shuffle(customers)
        else:
            random.shuffle(customers)

        routes = [[0, 0] for _ in range(truck_count)]
        current_dist = [0.0 for _ in range(truck_count)]
        unassigned = customers[:]
        lambda_balance = 0.5

        # Regret-2 construction with penalty
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
        else:
            no_improve_restarts += 1

        # Perturbation and improvement cycles
        removal_fraction = 0.4  # increased from 0.3 to 0.4
        for cycle in range(5):
            # Perturb: remove customers from longest and shortest routes
            route_lengths = [len(r) for r in routes]
            if min(route_lengths) < 3:
                break
            sorted_indices = sorted(range(truck_count), key=lambda i: current_dist[i])
            longest_idx = sorted_indices[-1]
            shortest_idx = sorted_indices[0]
            remove_count_long = max(1, int(removal_fraction * (len(routes[longest_idx]) - 2)))
            if remove_count_long > len(routes[longest_idx]) - 2:
                remove_count_long = len(routes[longest_idx]) - 2
            remove_count_short = max(1, int(removal_fraction * (len(routes[shortest_idx]) - 2)))
            if remove_count_short > len(routes[shortest_idx]) - 2:
                remove_count_short = len(routes[shortest_idx]) - 2
            removed = []
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
            # Repair using regret-2
            unassigned = removed[:]
            random.shuffle(unassigned)
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

            # Additional intra-route 2-opt pass after each perturbation cycle
            for r in range(truck_count):
                route = routes[r]
                if len(route) <= 3:
                    continue
                improved = True
                while improved:
                    improved = False
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
                        route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                        routes[r] = route
                        current_dist[r] = route_distance(route)
                        improved = True
                        new_max = max(current_dist)
                        if new_max < best_max:
                            best_max = new_max
                            local_best_routes = [list(r) for r in routes]
                            if new_max < global_best_max:
                                global_best_max = new_max
                                global_best_routes = [list(r) for r in routes]
                                report_best_vrp(routes)

            # Local search (same as before but with possible shake)
            max_iters = 10 * (n - 1) * truck_count
            improved = True
            iters = 0
            plateau_count = 0
            while improved and iters < max_iters:
                improved = False
                iters += 1

                # Shake if plateau
                if not improved and plateau_count >= 5:
                    r1 = random.randint(0, truck_count-1)
                    r2 = random.randint(0, truck_count-1)
                    if r1 != r2 and len(routes[r1]) > 2 and len(routes[r2]) > 2:
                        idx1 = random.randint(1, len(routes[r1])-2)
                        cust = routes[r1][idx1]
                        new_route1 = routes[r1][:idx1] + routes[r1][idx1+1:]
                        cost, pos = insert_cost(routes[r2], cust)
                        if cost != float('inf'):
                            routes[r1] = new_route1
                            routes[r2] = routes[r2][:pos] + [cust] + routes[r2][pos:]
                            current_dist[r1] = route_distance(routes[r1])
                            current_dist[r2] = route_distance(routes[r2])
                            new_max = max(current_dist)
                            if new_max < best_max:
                                best_max = new_max
                                local_best_routes = [list(r) for r in routes]
                                if new_max < global_best_max:
                                    global_best_max = new_max
                                    global_best_routes = [list(r) for r in routes]
                                    report_best_vrp(routes)
                            improved = True
                    plateau_count = 0

                # Relocate
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
                            if new_max < best_max:
                                routes[r1] = new_route1
                                routes[r2] = route2[:pos] + [cust] + route2[pos:]
                                current_dist[r1] = new_dist1
                                current_dist[r2] = new_dist2
                                best_max = new_max
                                local_best_routes = [list(r) for r in routes]
                                if new_max < global_best_max:
                                    global_best_max = new_max
                                    global_best_routes = [list(r) for r in routes]
                                    report_best_vrp(routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    plateau_count = 0
                    continue

                # Swap
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
                                if new_max < best_max:
                                    routes[r1] = new_route1
                                    routes[r2] = new_route2
                                    current_dist[r1] = new_dist1
                                    current_dist[r2] = new_dist2
                                    best_max = new_max
                                    local_best_routes = [list(r) for r in routes]
                                    if new_max < global_best_max:
                                        global_best_max = new_max
                                        global_best_routes = [list(r) for r in routes]
                                        report_best_vrp(routes)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    plateau_count = 0
                    continue

                # Intra-route 2-opt
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
                        new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                        routes[r] = new_route
                        current_dist[r] = route_distance(new_route)
                        new_max = max(current_dist)
                        if new_max < best_max:
                            best_max = new_max
                            local_best_routes = [list(r) for r in routes]
                            if new_max < global_best_max:
                                global_best_max = new_max
                                global_best_routes = [list(r) for r in routes]
                                report_best_vrp(routes)
                        improved = True
                        break
                if improved:
                    plateau_count = 0
                    continue

                # Cross-route 2-opt
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
                                if new_max < best_max:
                                    routes[r1] = new1
                                    routes[r2] = new2
                                    current_dist[r1] = new_dist1
                                    current_dist[r2] = new_dist2
                                    best_max = new_max
                                    local_best_routes = [list(r) for r in routes]
                                    if new_max < global_best_max:
                                        global_best_max = new_max
                                        global_best_routes = [list(r) for r in routes]
                                        report_best_vrp(routes)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    plateau_count = 0
                else:
                    plateau_count += 1

            # Update decay of removal fraction
            removal_fraction *= 0.95

            if best_max < global_best_max:
                global_best_max = best_max
                global_best_routes = [list(r) for r in routes]
                report_best_vrp(global_best_routes)

    return global_best_routes