import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]

    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def construction():
        routes = [[0, 0] for _ in range(truck_count)]
        route_distances = [0.0 for _ in range(truck_count)]
        unassigned = sorted(range(1, n))

        while unassigned:
            best_customer = None
            best_max_val = float('inf')
            best_pos = None
            best_delta = None

            for c in unassigned:
                # find best insertion for this customer
                best_local_max = float('inf')
                best_local_pos = None
                best_local_delta = None
                for r_idx, route in enumerate(routes):
                    curr_dist = route_distances[r_idx]
                    for i in range(1, len(route)):
                        new_dist = curr_dist - distance_matrix[route[i-1], route[i]] \
                                   + distance_matrix[route[i-1], c] + distance_matrix[c, route[i]]
                        other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                        cand_max = max(new_dist, other_max)
                        delta = new_dist - curr_dist
                        if cand_max < best_local_max - 1e-12:
                            best_local_max = cand_max
                            best_local_pos = (r_idx, i)
                            best_local_delta = delta
                        elif abs(cand_max - best_local_max) < 1e-12:
                            if delta < best_local_delta - 1e-12:
                                best_local_max = cand_max
                                best_local_pos = (r_idx, i)
                                best_local_delta = delta
                if best_local_pos is None:
                    continue
                if best_local_max < best_max_val - 1e-12:
                    best_max_val = best_local_max
                    best_customer = c
                    best_pos = best_local_pos
                    best_delta = best_local_delta
                elif abs(best_local_max - best_max_val) < 1e-12:
                    if best_local_delta < best_delta - 1e-12:
                        best_max_val = best_local_max
                        best_customer = c
                        best_pos = best_local_pos
                        best_delta = best_local_delta

            if best_customer is None:
                break
            r_idx, i = best_pos
            route = routes[r_idx]
            route.insert(i, best_customer)
            route_distances[r_idx] = compute_route_distance(route)
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
                                report_best_vrp(current_routes)
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
                                report_best_vrp(current_routes)
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
                        other_max = max(current_distances[:r1] + current_distances[r1+1:r2] + current_distances[r2+1:], default=0.0)
                        cand_max = max(new_dist1, best_new_dist2, other_max)
                        if cand_max < current_max - 1e-12:
                            current_routes[r1] = new1
                            current_routes[r2] = best_new2
                            current_distances[r1] = new_dist1
                            current_distances[r2] = best_new_dist2
                            current_max = cand_max
                            report_best_vrp(current_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return current_routes, current_distances, current_max

    def ruin_and_recreate(routes, route_distances):
        all_customers = list(range(1, n))
        num_remove = max(1, int(0.2 * (n-1)))
        max_dist = max(route_distances)
        longest_indices = [i for i, d in enumerate(route_distances) if abs(d - max_dist) < 1e-12]
        num_from_longest = (num_remove + 1) // 2
        longest_customers = []
        for idx in longest_indices:
            for node in routes[idx][1:-1]:
                longest_customers.append(node)
        random.shuffle(longest_customers)
        selected_longest = longest_customers[:min(num_from_longest, len(longest_customers))]
        remaining_needed = num_remove - len(selected_longest)
        other_customers = [c for c in all_customers if c not in selected_longest]
        random.shuffle(other_customers)
        selected_other = other_customers[:remaining_needed]
        customers_to_remove = set(selected_longest + selected_other)

        new_routes = []
        new_distances = []
        for route in routes:
            new_route = [0]
            for node in route[1:-1]:
                if node not in customers_to_remove:
                    new_route.append(node)
            new_route.append(0)
            new_routes.append(new_route)
            new_distances.append(compute_route_distance(new_route))

        unassigned = customers_to_remove.copy()

        def best_max_insert(customer):
            best_val = float('inf')
            best_pos = None
            second_val = float('inf')
            for r_idx, route in enumerate(new_routes):
                curr_dist = new_distances[r_idx]
                for i in range(1, len(route)):
                    new_dist = curr_dist - distance_matrix[route[i-1], route[i]] \
                               + distance_matrix[route[i-1], customer] \
                               + distance_matrix[customer, route[i]]
                    other_max = max(new_distances[:r_idx] + new_distances[r_idx+1:], default=0.0)
                    cand_max = max(new_dist, other_max)
                    if cand_max < best_val - 1e-12:
                        second_val = best_val
                        best_val = cand_max
                        best_pos = (r_idx, i)
                    elif cand_max < second_val - 1e-12 and abs(cand_max - best_val) > 1e-12:
                        second_val = cand_max
            return best_val, second_val, best_pos

        while unassigned:
            regrets = []
            for c in unassigned:
                best_val, second_val, _ = best_max_insert(c)
                regret = second_val - best_val if second_val != float('inf') else 0.0
                regrets.append((regret, best_val, c))
            regrets.sort(key=lambda x: (-x[0], -x[1], x[2]))
            selected = regrets[0][2]
            _, _, best_pos = best_max_insert(selected)
            r_idx, i = best_pos
            route = new_routes[r_idx]
            route.insert(i, selected)
            new_distances[r_idx] = compute_route_distance(route)
            unassigned.remove(selected)
        return new_routes, new_distances

    routes, route_distances = construction()
    routes, route_distances, best_max_val = local_search(routes, route_distances)
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)

    max_restarts = min(10, n)
    for _ in range(1, max_restarts):
        new_routes, new_distances = ruin_and_recreate(best_routes, [compute_route_distance(r) for r in best_routes])
        new_routes, new_distances, current_max = local_search(new_routes, new_distances)
        if current_max < best_max_val - 1e-12:
            best_max_val = current_max
            best_routes = [list(r) for r in new_routes]
            report_best_vrp(best_routes)

    return best_routes