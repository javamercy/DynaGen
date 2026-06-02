import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    best_routes = None
    best_max = float('inf')

    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    # Multi-start with different random seeds for tie-breaking
    for restart in range(5):
        random.seed(restart)
        routes = [[0, 0] for _ in range(truck_count)]
        route_distances = [0.0 for _ in range(truck_count)]
        for r in range(truck_count):
            route_distances[r] = compute_route_distance(routes[r])

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
                # Use random tie-breaking
                regrets.append((regret, best_val, random.random(), c))
            regrets.sort(key=lambda x: (-x[0], -x[1], x[2]))
            selected = regrets[0][3]
            _, _, best_pos = best_max(selected)
            r_idx, i = best_pos
            route = routes[r_idx]
            route.insert(i, selected)
            route_distances[r_idx] = compute_route_distance(route)
            unassigned.remove(selected)

        current_routes = [list(r) for r in routes]
        current_max = max(route_distances)

        def report_best_vrp(routes):
            nonlocal best_routes, best_max
            cand_max = max(compute_route_distance(r) for r in routes)
            if cand_max < best_max - 1e-12:
                best_max = cand_max
                best_routes = [list(r) for r in routes]

        # Initial best
        report_best_vrp(current_routes)

        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            improved = True
            max_iters = len(route) * 10
            it = 0
            while improved and it < max_iters:
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_distance(new_route)
                        if new_dist < route_distances[r_idx] - 1e-12:
                            route_distances[r_idx] = new_dist
                            current_routes[r_idx] = new_route
                            improved = True
                            new_max = max(route_distances)
                            if new_max < current_max - 1e-12:
                                current_max = new_max
                                report_best_vrp(current_routes)
                            break
                    if improved:
                        break
                it += 1

        # Inter-route improvement loop (swap + relocate)
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
                            other_max = max([route_distances[k] for k in range(truck_count) if k != r1 and k != r2] + [new_dist1, new_dist2])
                            if other_max < current_max - 1e-12:
                                current_routes[r1] = new1
                                current_routes[r2] = new2
                                route_distances[r1] = new_dist1
                                route_distances[r2] = new_dist2
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
            # Relocate (move customer from one route to another)
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
                        other_max = max(route_distances[:r1] + route_distances[r1+1:r2] + route_distances[r2+1:], default=0.0)
                        cand_max = max(new_dist1, best_new_dist2, other_max)
                        if cand_max < current_max - 1e-12:
                            current_routes[r1] = new1
                            current_routes[r2] = best_new2
                            route_distances[r1] = new_dist1
                            route_distances[r2] = best_new_dist2
                            current_max = cand_max
                            improved = True
                            report_best_vrp(current_routes)
                            break
                    if improved:
                        break
                if improved:
                    break

        # Load-balancing restart: move one customer from longest to shortest route
        for _ in range(10):
            # Find longest and shortest routes with customers
            max_len = -1
            max_r = -1
            min_len = float('inf')
            min_r = -1
            for r in range(truck_count):
                if len(current_routes[r]) > 2:
                    d = route_distances[r]
                    if d > max_len:
                        max_len = d
                        max_r = r
                    if d < min_len:
                        min_len = d
                        min_r = r
            if max_r == -1 or min_r == -1 or max_r == min_r:
                break
            # Try to move a customer from max_r to min_r
            route_max = current_routes[max_r]
            route_min = current_routes[min_r]
            best_move = None
            best_new_max = current_max
            for i in range(1, len(route_max)-1):
                c = route_max[i]
                new_max_route = route_max[:i] + route_max[i+1:]
                new_max_dist = compute_route_distance(new_max_route)
                # Find best insertion in min route
                best_min_dist = float('inf')
                best_pos = None
                for j in range(1, len(route_min)):
                    new_min_route = route_min[:j] + [c] + route_min[j:]
                    d_min = compute_route_distance(new_min_route)
                    if d_min < best_min_dist:
                        best_min_dist = d_min
                        best_pos = j
                other_max = max([route_distances[k] for k in range(truck_count) if k != max_r and k != min_r] + [new_max_dist, best_min_dist])
                if other_max < best_new_max - 1e-12:
                    best_new_max = other_max
                    best_move = (i, best_pos)
            if best_move is not None:
                i, j = best_move
                c = current_routes[max_r][i]
                # Remove from max route
                current_routes[max_r] = current_routes[max_r][:i] + current_routes[max_r][i+1:]
                route_distances[max_r] = compute_route_distance(current_routes[max_r])
                # Insert into min route
                current_routes[min_r] = current_routes[min_r][:j] + [c] + current_routes[min_r][j:]
                route_distances[min_r] = compute_route_distance(current_routes[min_r])
                current_max = max(route_distances)
                report_best_vrp(current_routes)
                # Re-run improvement loops after balance move
                improved = True
                it = 0
                while improved and it < max_iters:
                    # Repeat swap and relocate (same code as above)
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
                                    other_max = max([route_distances[k] for k in range(truck_count) if k != r1 and k != r2] + [new_dist1, new_dist2])
                                    if other_max < current_max - 1e-12:
                                        current_routes[r1] = new1
                                        current_routes[r2] = new2
                                        route_distances[r1] = new_dist1
                                        route_distances[r2] = new_dist2
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
                                other_max = max(route_distances[:r1] + route_distances[r1+1:r2] + route_distances[r2+1:], default=0.0)
                                cand_max = max(new_dist1, best_new_dist2, other_max)
                                if cand_max < current_max - 1e-12:
                                    current_routes[r1] = new1
                                    current_routes[r2] = best_new2
                                    route_distances[r1] = new_dist1
                                    route_distances[r2] = best_new_dist2
                                    current_max = cand_max
                                    improved = True
                                    report_best_vrp(current_routes)
                                    break
                            if improved:
                                break
                        if improved:
                            break
            else:
                break
        # End of restart

    # Return best found
    if best_routes is None:
        return routes
    return best_routes