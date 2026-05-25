import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def best_insertion(cust, route):
        best_cost = float('inf')
        best_pos = -1
        for pos in range(1, len(route)):
            i = route[pos-1]
            j = route[pos]
            cost = distance_matrix[i, cust] + distance_matrix[cust, j] - distance_matrix[i, j]
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        return best_cost, best_pos

    def regret_construction():
        unassigned = set(customers)
        routes = [[depot, depot] for _ in range(truck_count)]
        while unassigned:
            best_regret = -1
            best_candidates = []
            for cust in unassigned:
                costs = []
                for r_idx, route in enumerate(routes):
                    cost, pos = best_insertion(cust, route)
                    costs.append((cost, r_idx, pos))
                costs.sort(key=lambda x: x[0])
                if len(costs) == 1:
                    regret = costs[0][0] * 2
                else:
                    regret = costs[1][0] - costs[0][0]
                if regret > best_regret:
                    best_regret = regret
                    best_candidates = [(cust, costs[0][0], costs[0][1], costs[0][2])]
                elif regret == best_regret:
                    best_candidates.append((cust, costs[0][0], costs[0][1], costs[0][2]))
            chosen = random.choice(best_candidates)
            cust, _, best_route_idx, best_pos = chosen
            route = routes[best_route_idx]
            route.insert(best_pos, cust)
            unassigned.remove(cust)
        return routes

    def local_search(routes):
        n_customers = n - 1
        max_iters = 2 * n_customers
        best_routes = [list(r) for r in routes]
        best_max = max(route_dist(r) for r in best_routes)
        for _ in range(max_iters):
            improved = False
            # Inter-route relocate
            for r_idx, route in enumerate(routes):
                if len(route) <= 3:
                    continue
                customers_in_route = route[1:-1]
                cust_list = list(customers_in_route)
                random.shuffle(cust_list)
                for cust in cust_list:
                    new_route = [x for x in route if x != cust]
                    other_indices = [i for i in range(truck_count) if i != r_idx]
                    random.shuffle(other_indices)
                    for other_idx in other_indices:
                        other_route = routes[other_idx]
                        cost, pos = best_insertion(cust, other_route)
                        candidate_routes = [list(r) for r in routes]
                        candidate_routes[r_idx] = new_route
                        other_new = list(other_route)
                        other_new.insert(pos, cust)
                        candidate_routes[other_idx] = other_new
                        dists = [route_dist(r) for r in candidate_routes]
                        new_max = max(dists)
                        if new_max < best_max:
                            best_max = new_max
                            routes = candidate_routes
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                # Intra-route 2-opt
                for r_idx, route in enumerate(routes):
                    if len(route) <= 4:
                        continue
                    n_nodes = len(route)
                    best_imp = False
                    indices = [(i, j) for i in range(1, n_nodes-2) for j in range(i+1, n_nodes-1)]
                    random.shuffle(indices)
                    for i, j in indices:
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if route_dist(new_route) < route_dist(route):
                            routes[r_idx] = new_route
                            best_imp = True
                            break
                    if best_imp:
                        dists = [route_dist(r) for r in routes]
                        new_max = max(dists)
                        if new_max < best_max:
                            best_max = new_max
                        improved = True
                        break
            if not improved:
                break
        # Finalize routes
        result = []
        for r in routes:
            if len(r) <= 2:
                result.append([0, 0])
            else:
                if r[0] != 0:
                    r.insert(0, 0)
                if r[-1] != 0:
                    r.append(0)
                result.append(r)
        while len(result) < truck_count:
            result.append([0, 0])
        return result, best_max

    best_overall_routes = None
    best_overall_max = float('inf')
    num_restarts = 3
    for _ in range(num_restarts):
        routes = regret_construction()
        routes, cur_max = local_search(routes)
        # Perturbation and re-optimization
        n_customers = n - 1
        num_to_perturb = max(1, n_customers // 10)
        for _ in range(2):
            all_customers = list(range(1, n))
            random.shuffle(all_customers)
            to_remove = set(all_customers[:num_to_perturb])
            new_routes = [[0, 0] for _ in range(truck_count)]
            remaining = list(range(1, n))
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for cust in route[1:-1]:
                    if cust not in to_remove:
                        new_routes[r_idx].insert(-1, cust)
                        if cust in remaining:
                            remaining.remove(cust)
            unassigned = to_remove
            while unassigned:
                best_regret = -1
                best_candidates = []
                for cust in unassigned:
                    costs = []
                    for r_idx, route in enumerate(new_routes):
                        cost, pos = best_insertion(cust, route)
                        costs.append((cost, r_idx, pos))
                    costs.sort(key=lambda x: x[0])
                    if len(costs) == 1:
                        regret = costs[0][0] * 2
                    else:
                        regret = costs[1][0] - costs[0][0]
                    if regret > best_regret:
                        best_regret = regret
                        best_candidates = [(cust, costs[0][0], costs[0][1], costs[0][2])]
                    elif regret == best_regret:
                        best_candidates.append((cust, costs[0][0], costs[0][1], costs[0][2]))
                chosen = random.choice(best_candidates)
                cust, _, best_route_idx, best_pos = chosen
                route = new_routes[best_route_idx]
                route.insert(best_pos, cust)
                unassigned.remove(cust)
            routes, cur_max = local_search(new_routes)
        if cur_max < best_overall_max:
            best_overall_max = cur_max
            best_overall_routes = routes
            report_best_vrp(routes)
    return best_overall_routes