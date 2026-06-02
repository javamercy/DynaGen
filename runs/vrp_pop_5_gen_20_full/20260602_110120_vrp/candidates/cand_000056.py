import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
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

        def best_max(customer):
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
            return best_val, second_val, best_pos

        while unassigned:
            regrets = []
            for c in unassigned:
                best_val, second_val, _ = best_max(c)
                regret = second_val - best_val if second_val != float('inf') else 0.0
                regrets.append((regret, best_val, c))
            regrets.sort(key=lambda x: (-x[0], -x[1], x[2]))
            selected = regrets[0][2]
            _, _, best_pos = best_max(selected)
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

        # Intra-route 2-opt (bounded)
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            improved = True
            max_iters = len(route) * 10
            it = 0
            while improved and it < max_iters:
                improved = False
                it += 1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_distance(new_route)
                        if new_dist < current_distances[r_idx] - 1e-12:
                            current_distances[r_idx] = new_dist
                            current_routes[r_idx] = new_route
                            route = new_route
                            improved = True
                            new_max = max(current_distances)
                            if new_max < current_max - 1e-12:
                                current_max = new_max
                                report_best_vrp(current_routes)
                            break
                    if improved:
                        break

        # Inter-route improvements (swap+relocate) bounded
        improved = True
        max_iters = n * n
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            # Swap
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
                            other_max = max([current_distances[k] for k in range(truck_count) if k != r1 and k != r2] + [new_dist1, new_dist2])
                            if other_max < current_max - 1e-12:
                                current_routes[r1] = new1
                                current_routes[r2] = new2
                                current_distances[r1] = new_dist1
                                current_distances[r2] = new_dist2
                                current_max = other_max
                                improved = True
                                report_best_vrp(current_routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Relocate
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
                            improved = True
                            report_best_vrp(current_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
        return current_routes, current_distances, current_max

    def destroy_and_repair(routes):
        # Identify longest routes
        dists = [compute_route_distance(r) for r in routes]
        sorted_indices = sorted(range(truck_count), key=lambda i: -dists[i])
        # Remove customers from top 2 longest (if they have customers)
        removed_customers = []
        new_routes = [list(r) for r in routes]
        for idx in sorted_indices[:2]:
            route = new_routes[idx]
            if len(route) <= 2:
                continue
            # Remove a random number between 1 and max(1, len(route)-2)
            remove_count = random.randint(1, max(1, len(route)-2))
            # Randomly select positions (excluding depot)
            positions = list(range(1, len(route)-1))
            random.shuffle(positions)
            positions_to_remove = positions[:remove_count]
            # Sort descending to remove safely
            positions_to_remove.sort(reverse=True)
            for pos in positions_to_remove:
                removed_customers.append(route.pop(pos))
        # Reinsert removed customers using regret-2
        unassigned = set(removed_customers)
        while unassigned:
            regrets = []
            for c in unassigned:
                best_val = float('inf')
                second_val = float('inf')
                for r_idx, route in enumerate(new_routes):
                    for i in range(1, len(route)):
                        new_dist = compute_route_distance(route[:i] + [c] + route[i:])
                        if new_dist < best_val:
                            second_val = best_val
                            best_val = new_dist
                            best_pos = (r_idx, i)
                        elif new_dist < second_val:
                            second_val = new_dist
                regret = second_val - best_val if second_val != float('inf') else 0.0
                regrets.append((regret, best_val, c, best_pos))
            regrets.sort(key=lambda x: (-x[0], -x[1], x[2]))
            _, _, selected, (r_idx, i) = regrets[0]
            new_routes[r_idx].insert(i, selected)
            unassigned.remove(selected)
        return new_routes

    # Main
    best_routes = None
    best_max_val = float('inf')
    max_restarts = min(10, n)

    routes, route_distances = construction()
    routes, route_distances, current_max = local_search(routes, route_distances)
    if current_max < best_max_val - 1e-12:
        best_max_val = current_max
        best_routes = routes
        report_best_vrp(routes)

    for restart in range(1, max_restarts):
        new_routes = destroy_and_repair(best_routes)
        new_distances = [compute_route_distance(r) for r in new_routes]
        new_routes, new_distances, new_max = local_search(new_routes, new_distances)
        if new_max < best_max_val - 1e-12:
            best_max_val = new_max
            best_routes = new_routes
            report_best_vrp(new_routes)

    return best_routes