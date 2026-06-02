import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    random.seed(0)

    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def best_k_insertions(customer, routes, route_distances, k=3):
        costs = []
        for r_idx, route in enumerate(routes):
            cur_dist = route_distances[r_idx]
            for i in range(1, len(route)):
                new_dist = cur_dist - distance_matrix[route[i-1], route[i]] \
                           + distance_matrix[route[i-1], customer] \
                           + distance_matrix[customer, route[i]]
                other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                cand_max = max(new_dist, other_max)
                costs.append(cand_max)
        costs.sort()
        if len(costs) < k:
            return costs
        return costs[:k]

    def regret_insertion(selected_customer, routes, route_distances):
        best_val = float('inf')
        best_pos = None
        best_costs = []
        for r_idx, route in enumerate(routes):
            cur_dist = route_distances[r_idx]
            for i in range(1, len(route)):
                new_dist = cur_dist - distance_matrix[route[i-1], route[i]] \
                           + distance_matrix[route[i-1], selected_customer] \
                           + distance_matrix[selected_customer, route[i]]
                other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                cand_max = max(new_dist, other_max)
                if cand_max < best_val - 1e-12:
                    best_val = cand_max
                    best_pos = (r_idx, i)
        return best_pos, best_val

    def construction():
        routes = [[0, 0] for _ in range(truck_count)]
        route_distances = [0.0 for _ in range(truck_count)]
        unassigned = set(range(1, n))

        while unassigned:
            best_customer = None
            best_regret = -float('inf')
            best_insertion_cost = float('inf')
            best_pos = None
            for c in sorted(unassigned):
                costs = best_k_insertions(c, routes, route_distances, k=3)
                if not costs:
                    continue
                best_cost = costs[0]
                if len(costs) >= 3:
                    regret = (costs[1] - costs[0]) + (costs[2] - costs[0])
                elif len(costs) == 2:
                    regret = costs[1] - costs[0]
                else:
                    regret = 0.0
                if regret > best_regret + 1e-12 or (abs(regret - best_regret) < 1e-12 and best_cost < best_insertion_cost - 1e-12) or (abs(regret - best_regret) < 1e-12 and abs(best_cost - best_insertion_cost) < 1e-12 and c < best_customer):
                    best_regret = regret
                    best_insertion_cost = best_cost
                    best_customer = c
            # get best position for best_customer
            best_pos, _ = regret_insertion(best_customer, routes, route_distances)
            r_idx, i = best_pos
            routes[r_idx].insert(i, best_customer)
            route_distances[r_idx] = compute_route_distance(routes[r_idx])
            unassigned.remove(best_customer)
        return routes, route_distances

    def local_search(routes, route_distances):
        current_routes = [list(r) for r in routes]
        current_distances = list(route_distances)
        current_max = max(current_distances)
        improved = True
        max_iters = n * n
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = current_routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_distance(new_route)
                        if new_dist < current_distances[r_idx] - 1e-12:
                            current_distances[r_idx] = new_dist
                            current_routes[r_idx] = new_route
                            new_max = max(current_distances)
                            if new_max < current_max - 1e-12:
                                current_max = new_max
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Inter-swap
            for r1 in range(truck_count):
                for r2 in range(r1+1, truck_count):
                    route1 = current_routes[r1]
                    route2 = current_routes[r2]
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i] + [route2[j]] + route1[i+1:]
                            new2 = route2[:j] + [route1[i]] + route2[j+1:]
                            new_dist1 = compute_route_distance(new1)
                            new_dist2 = compute_route_distance(new2)
                            other_max = max(current_distances[k] for k in range(truck_count) if k not in (r1, r2))
                            cand_max = max(new_dist1, new_dist2, other_max)
                            if cand_max < current_max - 1e-12:
                                current_routes[r1] = new1
                                current_routes[r2] = new2
                                current_distances[r1] = new_dist1
                                current_distances[r2] = new_dist2
                                current_max = cand_max
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
            # Inter-relocate
            for r1 in range(truck_count):
                for r2 in range(truck_count):
                    if r1 == r2:
                        continue
                    route1 = current_routes[r1]
                    route2 = current_routes[r2]
                    for i in range(1, len(route1)-1):
                        c = route1[i]
                        new1 = route1[:i] + route1[i+1:]
                        new_dist1 = compute_route_distance(new1)
                        best_new2 = None
                        best_new_dist2 = float('inf')
                        for j in range(1, len(route2)):
                            new2 = route2[:j] + [c] + route2[j:]
                            d2 = compute_route_distance(new2)
                            if d2 < best_new_dist2 - 1e-12:
                                best_new_dist2 = d2
                                best_new2 = new2
                        other_max = max([current_distances[k] for k in range(truck_count) if k not in (r1, r2)] or [0])
                        cand_max = max(new_dist1, best_new_dist2, other_max)
                        if cand_max < current_max - 1e-12:
                            current_routes[r1] = new1
                            current_routes[r2] = best_new2
                            current_distances[r1] = new_dist1
                            current_distances[r2] = best_new_dist2
                            current_max = cand_max
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Inter-route 2-opt*
            for r1 in range(truck_count):
                for r2 in range(r1+1, truck_count):
                    route1 = current_routes[r1]
                    route2 = current_routes[r2]
                    if len(route1) <= 2 or len(route2) <= 2:
                        continue
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i] + route2[j:]
                            new2 = route2[:j] + route1[i:]
                            new_dist1 = compute_route_distance(new1)
                            new_dist2 = compute_route_distance(new2)
                            other_max = max(current_distances[k] for k in range(truck_count) if k not in (r1, r2))
                            cand_max = max(new_dist1, new_dist2, other_max)
                            if cand_max < current_max - 1e-12:
                                current_routes[r1] = new1
                                current_routes[r2] = new2
                                current_distances[r1] = new_dist1
                                current_distances[r2] = new_dist2
                                current_max = cand_max
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return current_routes, current_distances, current_max

    def ruin_recreate(routes, route_distances, frac=0.5):
        new_routes = [list(r) for r in routes]
        new_distances = list(route_distances)
        sorted_indices = sorted(range(truck_count), key=lambda i: new_distances[i], reverse=True)
        removed_customers = []
        for idx in sorted_indices[:min(3, truck_count)]:
            route = new_routes[idx]
            if len(route) <= 2:
                continue
            num_remove = max(1, int(frac * (len(route) - 2)))
            removable = list(range(1, len(route)-1))
            random.shuffle(removable)
            to_remove = removable[:num_remove]
            to_remove.sort(reverse=True)
            for pos in to_remove:
                removed_customers.append(route.pop(pos))
            new_distances[idx] = compute_route_distance(route)
        if not removed_customers:
            return new_routes, new_distances
        unassigned = set(removed_customers)
        while unassigned:
            best_customer = None
            best_regret = -float('inf')
            best_insertion_cost = float('inf')
            best_pos = None
            for c in sorted(unassigned):
                costs = best_k_insertions(c, new_routes, new_distances, k=3)
                if not costs:
                    continue
                best_cost = costs[0]
                if len(costs) >= 3:
                    regret = (costs[1] - costs[0]) + (costs[2] - costs[0])
                elif len(costs) == 2:
                    regret = costs[1] - costs[0]
                else:
                    regret = 0.0
                if regret > best_regret + 1e-12 or (abs(regret - best_regret) < 1e-12 and best_cost < best_insertion_cost - 1e-12) or (abs(regret - best_regret) < 1e-12 and abs(best_cost - best_insertion_cost) < 1e-12 and c < best_customer):
                    best_regret = regret
                    best_insertion_cost = best_cost
                    best_customer = c
            best_pos, _ = regret_insertion(best_customer, new_routes, new_distances)
            r_idx, i = best_pos
            new_routes[r_idx].insert(i, best_customer)
            new_distances[r_idx] = compute_route_distance(new_routes[r_idx])
            unassigned.remove(best_customer)
        return new_routes, new_distances

    best_routes = None
    best_max_val = float('inf')
    max_restarts = 20

    routes, route_distances = construction()
    routes, route_distances, current_max = local_search(routes, route_distances)
    if current_max < best_max_val - 1e-12:
        best_max_val = current_max
        best_routes = routes
        report_best_vrp(best_routes)

    stagnation = 0
    for restart in range(max_restarts - 1):
        if stagnation >= 3:
            frac = 0.7
        else:
            frac = 0.5
        routes, route_distances = ruin_recreate(best_routes, [compute_route_distance(r) for r in best_routes], frac)
        routes, route_distances, current_max = local_search(routes, route_distances)
        if current_max < best_max_val - 1e-12:
            best_max_val = current_max
            best_routes = routes
            stagnation = 0
            report_best_vrp(best_routes)
        else:
            stagnation += 1

    return best_routes