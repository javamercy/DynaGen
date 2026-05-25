import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]

    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def compute_new_max(routes, route_dists, cust, r, pos):
        route = routes[r]
        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
        new_dist = route_dists[r] + inc
        other_max = max(route_dists[i] for i in range(truck_count) if i != r)
        return max(new_dist, other_max)

    def insert_best(routes, route_dists, cust, best_r, best_pos):
        routes[best_r].insert(best_pos, cust)
        route_dists[best_r] = route_distance(routes[best_r])

    def construct_regret2():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0 for _ in range(truck_count)]
        customers = list(range(1, n))
        # Precompute best insertion for each customer per route
        while customers:
            regret = []
            best_positions = []
            best_routes = []
            for cust in customers:
                best_inc = float('inf')
                second_best_inc = float('inf')
                best_r = -1
                best_pos = -1
                for r in range(truck_count):
                    route = routes[r]
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        if inc < best_inc:
                            second_best_inc = best_inc
                            best_inc = inc
                            best_r = r
                            best_pos = pos
                        elif inc < second_best_inc:
                            second_best_inc = inc
                reg = second_best_inc - best_inc if second_best_inc != float('inf') else best_inc
                regret.append(reg)
                best_positions.append((best_r, best_pos))
                best_routes.append(best_r)
            # Select customer with max regret, tie-break by smaller customer index
            max_regret = -1
            best_idx = -1
            for idx, reg in enumerate(regret):
                if reg > max_regret + 1e-9:
                    max_regret = reg
                    best_idx = idx
                elif abs(reg - max_regret) < 1e-9:
                    if customers[idx] < customers[best_idx]:
                        best_idx = idx
            cust = customers.pop(best_idx)
            best_r, best_pos = best_positions[best_idx]
            insert_best(routes, route_dists, cust, best_r, best_pos)
        return routes, route_dists

    def local_search(routes, route_dists):
        best_routes = [list(r) for r in routes]
        best_max = max(route_dists)
        try:
            report_best_vrp(best_routes)
        except NameError:
            pass
        max_iters = 10 * n * truck_count
        for _ in range(max_iters):
            improved = False
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
                        for pos in range(1, len(route2)):
                            inc = distance_matrix[route2[pos-1], cust] + distance_matrix[cust, route2[pos]] - distance_matrix[route2[pos-1], route2[pos]]
                            new_dist2 = route_dists[r2] + inc
                            other_dists = [route_dists[i] for i in range(truck_count) if i not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_max - 1e-9:
                                routes[r1] = new_route1
                                route_dists[r1] = new_dist1
                                routes[r2].insert(pos, cust)
                                route_dists[r2] = new_dist2
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                improved = True
                                try:
                                    report_best_vrp(best_routes)
                                except NameError:
                                    pass
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
                            other_dists = [route_dists[i] for i in range(truck_count) if i not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_max - 1e-9:
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                route_dists[r1] = new_dist1
                                route_dists[r2] = new_dist2
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                improved = True
                                try:
                                    report_best_vrp(best_routes)
                                except NameError:
                                    pass
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Intra-route 2-opt
            for r in range(truck_count):
                route = routes[r]
                if len(route) <= 3:
                    continue
                best_improve = 0.0
                best_i = best_j = -1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < route_dists[r] - 1e-9:
                            improvement = route_dists[r] - new_dist
                            if improvement > best_improve:
                                best_improve = improvement
                                best_i, best_j = i, j
                if best_improve > 0:
                    new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                    routes[r] = new_route
                    route_dists[r] = route_distance(new_route)
                    new_max = max(route_dists)
                    if new_max < best_max - 1e-9:
                        best_max = new_max
                        best_routes = [list(r) for r in routes]
                        try:
                            report_best_vrp(best_routes)
                        except NameError:
                            pass
                    improved = True
                    break
            if improved:
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
                            other_dists = [route_dists[k] for k in range(truck_count) if k not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_max - 1e-9:
                                routes[r1] = new1
                                routes[r2] = new2
                                route_dists[r1] = new_dist1
                                route_dists[r2] = new_dist2
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                improved = True
                                try:
                                    report_best_vrp(best_routes)
                                except NameError:
                                    pass
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return best_routes, best_max

    def perturbation(routes, route_dists):
        # Randomly relocate a few customers
        num_customers = n - 1
        num_to_move = max(1, num_customers // 20)
        for _ in range(num_to_move):
            # choose a random customer from a random route
            nonempty = [r for r in range(truck_count) if len(routes[r]) > 2]
            if not nonempty:
                break
            r1 = random.choice(nonempty)
            route1 = routes[r1]
            idx = random.randint(1, len(route1)-2)
            cust = route1.pop(idx)
            route_dists[r1] = route_distance(route1)
            # insert into random route at random position
            r2 = random.randint(0, truck_count-1)
            route2 = routes[r2]
            pos = random.randint(1, len(route2)-1)
            route2.insert(pos, cust)
            route_dists[r2] = route_distance(route2)
        return routes, route_dists

    # Main
    num_restarts = 10
    best_routes = []
    best_max = float('inf')
    for _ in range(num_restarts):
        routes, route_dists = construct_regret2()
        routes, max_dist = local_search(routes, route_dists)
        # Perturb and local search again
        for _ in range(3):
            routes, route_dists = perturbation(routes, route_dists)
            routes, max_dist2 = local_search(routes, route_dists)
            if max_dist2 < max_dist - 1e-9:
                max_dist = max_dist2
        if max_dist < best_max - 1e-9:
            best_routes = routes
            best_max = max_dist
    # Ensure empty routes
    for r in range(truck_count):
        if len(best_routes[r]) == 0:
            best_routes[r] = [0, 0]
    return best_routes