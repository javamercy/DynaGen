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
        inc = (distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]])
        new_dist = route_dists[r] + inc
        other_max = max(route_dists[i] for i in range(truck_count) if i != r)
        return max(new_dist, other_max)

    def insert_best(routes, route_dists, cust, best_r, best_pos):
        routes[best_r].insert(best_pos, cust)
        route_dists[best_r] = route_distance(routes[best_r])

    def construct_regret():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0 for _ in range(truck_count)]
        customers = list(range(1, n))
        while customers:
            best_regret = -float('inf')
            best_cust = None
            best_r = -1
            best_pos = -1
            best_second_r = -1
            best_second_pos = -1
            for cust in customers:
                best_new_max = float('inf')
                best_r_c = -1
                best_pos_c = -1
                second_new_max = float('inf')
                second_r_c = -1
                second_pos_c = -1
                for r in range(truck_count):
                    route = routes[r]
                    for pos in range(1, len(route)):
                        new_max = compute_new_max(routes, route_dists, cust, r, pos)
                        if new_max < best_new_max - 1e-9:
                            second_new_max = best_new_max
                            second_r_c = best_r_c
                            second_pos_c = best_pos_c
                            best_new_max = new_max
                            best_r_c = r
                            best_pos_c = pos
                        elif new_max < second_new_max - 1e-9:
                            second_new_max = new_max
                            second_r_c = r
                            second_pos_c = pos
                regret = second_new_max - best_new_max
                if regret > best_regret + 1e-9:
                    best_regret = regret
                    best_cust = cust
                    best_r = best_r_c
                    best_pos = best_pos_c
                    best_second_r = second_r_c
                    best_second_pos = second_pos_c
                elif abs(regret - best_regret) < 1e-9:
                    # tie-break by best_new_max, then truck index, then position
                    if best_new_max < compute_new_max(routes, route_dists, best_cust, best_r, best_pos) - 1e-9:
                        best_cust = cust
                        best_r = best_r_c
                        best_pos = best_pos_c
            # Insert best_cust
            insert_best(routes, route_dists, best_cust, best_r, best_pos)
            customers.remove(best_cust)
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
                            inc = (distance_matrix[route2[pos-1], cust] + distance_matrix[cust, route2[pos]] - distance_matrix[route2[pos-1], route2[pos]])
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

    def perturb(routes, route_dists):
        # Randomly relocate a small number of customers (up to 2)
        num_perturb = min(2, n-1)
        all_customers = list(range(1, n))
        random.shuffle(all_customers)
        for cust in all_customers[:num_perturb]:
            # Find current route
            for r in range(truck_count):
                if cust in routes[r]:
                    idx = routes[r].index(cust)
                    routes[r].pop(idx)
                    route_dists[r] = route_distance(routes[r])
                    break
            # Reinsert in random position of a random route (could be same, but we force different with 0.5 prob)
            if random.random() < 0.5:
                # choose different truck randomly
                other_trucks = [t for t in range(truck_count) if t != r]
                if other_trucks:
                    new_r = random.choice(other_trucks)
                else:
                    new_r = r
            else:
                new_r = r
            pos = random.randint(1, len(routes[new_r])-1)
            routes[new_r].insert(pos, cust)
            route_dists[new_r] = route_distance(routes[new_r])
        return routes, route_dists

    # Main
    num_restarts = 10
    best_routes = []
    best_max = float('inf')
    for _ in range(num_restarts):
        routes, route_dists = construct_regret()
        # Run local search, then perturb and re-run up to 2 times
        for _ in range(3):  # first run plus up to 2 perturbations
            routes, max_dist = local_search(routes, route_dists)
            if max_dist < best_max - 1e-9:
                best_max = max_dist
                best_routes = [list(r) for r in routes]
            # Perturb and continue if we haven't reached the last iteration
            if _ < 2:
                routes, route_dists = perturb(routes, route_dists)
    # Ensure empty routes
    for r in range(truck_count):
        if len(best_routes[r]) == 0:
            best_routes[r] = [0, 0]
    return best_routes