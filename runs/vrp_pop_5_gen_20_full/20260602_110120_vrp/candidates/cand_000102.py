import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    random.seed(0)

    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def construction():
        routes = [[0, 0] for _ in range(truck_count)]
        route_distances = [0.0 for _ in range(truck_count)]
        unassigned = set(range(1, n))

        def best_max_position(customer):
            best_val = float('inf')
            best_pos = None
            second_val = float('inf')
            for r_idx, route in enumerate(routes):
                curr_dist = route_distances[r_idx]
                for i in range(1, len(route)):
                    new_dist = curr_dist - distance_matrix[route[i-1], route[i]] \
                               + distance_matrix[route[i-1], customer] \
                               + distance_matrix[customer, route[i]]
                    other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                    cand_max = max(new_dist, other_max)
                    if cand_max < best_val - 1e-12:
                        second_val = best_val
                        best_val = cand_max
                        best_pos = (r_idx, i)
                    elif cand_max < second_val - 1e-12 and abs(cand_max - best_val) > 1e-12:
                        second_val = cand_max
            if best_pos is None:
                return float('inf'), float('inf'), (0, len(routes[0])-1)
            return best_val, second_val, best_pos

        while unassigned:
            regrets = []
            for c in unassigned:
                best_val, second_val, _ = best_max_position(c)
                regret = second_val - best_val if second_val != float('inf') else 0.0
                regrets.append((regret, best_val, c))
            regrets.sort(key=lambda x: (-x[0], -x[1], x[2]))
            selected = regrets[0][2]
            _, _, best_pos = best_max_position(selected)
            r_idx, i = best_pos
            route = routes[r_idx]
            route.insert(i, selected)
            route_distances[r_idx] = compute_route_distance(route)
            unassigned.remove(selected)
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
                            if d2 < best_new_dist2:
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
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i] + route2[j:]
                            new2 = route2[:j] + route1[i:]
                            if new1[-1] != 0 or new2[-1] != 0:
                                continue
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

    def regret2_insertion(removed_customers, routes, route_distances):
        unassigned = set(removed_customers)
        while unassigned:
            regrets = []
            for c in unassigned:
                best_val = float('inf')
                second_val = float('inf')
                best_pos = None
                for r_idx, r in enumerate(routes):
                    for i in range(1, len(r)):
                        new_dist = route_distances[r_idx] - distance_matrix[r[i-1], r[i]] \
                                   + distance_matrix[r[i-1], c] \
                                   + distance_matrix[c, r[i]]
                        other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                        cand_max = max(new_dist, other_max)
                        if cand_max < best_val - 1e-12:
                            second_val = best_val
                            best_val = cand_max
                            best_pos = (r_idx, i)
                        elif cand_max < second_val - 1e-12 and abs(cand_max - best_val) > 1e-12:
                            second_val = cand_max
                regret = second_val - best_val if second_val != float('inf') else 0.0
                regrets.append((regret, best_val, c, best_pos))
            regrets.sort(key=lambda x: (-x[0], -x[1], x[2]))
            selected = regrets[0][2]
            r_idx, i = regrets[0][3]
            routes[r_idx].insert(i, selected)
            route_distances[r_idx] = compute_route_distance(routes[r_idx])
            unassigned.remove(selected)

    def ruin_recreate(routes, route_distances):
        new_routes = [list(r) for r in routes]
        new_distances = list(route_distances)
        sorted_indices = sorted(range(truck_count), key=lambda i: new_distances[i], reverse=True)
        removed_customers = []
        num_routes = min(3, truck_count)
        for idx in sorted_indices[:num_routes]:
            route = new_routes[idx]
            if len(route) <= 2:
                continue
            num_remove = max(1, int(0.3 * (len(route) - 2)))
            removable = list(range(1, len(route)-1))
            random.shuffle(removable)
            to_remove = removable[:num_remove]
            to_remove.sort(reverse=True)
            for pos in to_remove:
                removed_customers.append(route.pop(pos))
            new_distances[idx] = compute_route_distance(route)
        if not removed_customers:
            return new_routes, new_distances
        regret2_insertion(removed_customers, new_routes, new_distances)
        return new_routes, new_distances

    def deterministic_perturbation(routes, route_distances):
        new_routes = [list(r) for r in routes]
        new_distances = list(route_distances)
        max_idx = max(range(truck_count), key=lambda i: new_distances[i])
        route = new_routes[max_idx]
        if len(route) <= 2:
            return new_routes, new_distances
        interior = list(range(1, len(route)-1))
        if not interior:
            return new_routes, new_distances
        dist_to_depot = [(distance_matrix[0, route[i]], i) for i in interior]
        dist_to_depot.sort(reverse=True, key=lambda x: x[0])
        num_remove = max(1, int(0.2 * len(interior)))
        to_remove = [i for _, i in dist_to_depot[:num_remove]]
        to_remove.sort(reverse=True)
        removed_customers = []
        for pos in to_remove:
            removed_customers.append(route.pop(pos))
        new_distances[max_idx] = compute_route_distance(route)
        regret2_insertion(removed_customers, new_routes, new_distances)
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

    for restart in range(max_restarts - 1):
        if restart % 2 == 0:
            routes, route_distances = ruin_recreate(best_routes, [compute_route_distance(r) for r in best_routes])
        else:
            routes, route_distances = deterministic_perturbation(best_routes, [compute_route_distance(r) for r in best_routes])
        routes, route_distances, current_max = local_search(routes, route_distances)
        if current_max < best_max_val - 1e-12:
            best_max_val = current_max
            best_routes = routes
            report_best_vrp(best_routes)

    return best_routes