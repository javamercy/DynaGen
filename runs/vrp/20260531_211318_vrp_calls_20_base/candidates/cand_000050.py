import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    dist = distance_matrix

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    customers = list(range(1, n))
    random.shuffle(customers)
    routes = [[0, 0] for _ in range(truck_count)]
    for cust in customers:
        best_truck = None
        best_pos = None
        best_max = float('inf')
        best_total = float('inf')
        for t, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_routes = routes[:t] + [new_route] + routes[t+1:]
                new_max = max(route_distance(r) for r in new_routes)
                new_total = sum(route_distance(r) for r in new_routes)
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = pos
        routes[best_truck].insert(best_pos, cust)

    current_routes = [list(r) for r in routes]
    best_routes = [list(r) for r in routes]
    best_max = max(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)

    max_iter = max(100, 2 * n)
    removal_fraction = 0.3
    num_removals = max(1, int(removal_fraction * (n-1)))
    T0 = best_max * 2.0  # higher initial temperature
    T = T0
    no_improve = 0

    def worst_removal(routes, num_removals):
        all_contribs = []
        for t, route in enumerate(routes):
            if len(route) <= 2:
                continue
            base = route_distance(route)
            for i in range(1, len(route)-1):
                prev = route[i-1]
                nxt = route[i+1]
                with_ = dist[prev, route[i]] + dist[route[i], nxt]
                without = dist[prev, nxt]
                contrib = with_ - without
                all_contribs.append((-contrib, t, i, route[i]))
        all_contribs.sort(key=lambda x: x[0])
        to_remove = set()
        for _, t, i, cust in all_contribs[:num_removals]:
            to_remove.add(cust)
        new_routes = []
        for t, route in enumerate(routes):
            new_routes.append([0] + [c for c in route[1:-1] if c not in to_remove] + [0])
        return list(to_remove), new_routes

    def random_removal(routes, num_removals):
        all_customers = [c for r in routes for c in r[1:-1]]
        random.shuffle(all_customers)
        to_remove = set(all_customers[:num_removals])
        new_routes = []
        for route in routes:
            new_routes.append([0] + [c for c in route[1:-1] if c not in to_remove] + [0])
        return list(to_remove), new_routes

    def route_removal(routes, num_removals):
        # Remove entire routes except depot, then remove some customers to reach num_removals
        routes_with_index = [(t, r) for t, r in enumerate(routes) if len(r) > 2]
        if not routes_with_index:
            return [], [[0,0] for _ in routes]
        random.shuffle(routes_with_index)
        to_remove_set = set()
        new_routes = [list(r) for r in routes]
        for t, r in routes_with_index:
            if len(to_remove_set) >= num_removals:
                break
            # Remove all customers from this route (except depot)
            custs = r[1:-1]
            to_remove_set.update(custs)
            new_routes[t] = [0,0]
        # If still not enough, add random customers from remaining
        remaining_customers = [c for r in new_routes for c in r[1:-1] if c not in to_remove_set]
        random.shuffle(remaining_customers)
        while len(to_remove_set) < num_removals and remaining_customers:
            c = remaining_customers.pop()
            to_remove_set.add(c)
        # Rebuild routes without removed customers
        final_routes = []
        for route in routes:
            final_routes.append([0] + [c for c in route[1:-1] if c not in to_remove_set] + [0])
        return list(to_remove_set), final_routes

    def greedy_repair(routes, unassigned):
        routes = [list(r) for r in routes]
        unassigned = list(unassigned)
        for cust in unassigned:
            best_max = float('inf')
            best_total = float('inf')
            best_truck = None
            best_pos = None
            for t, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_routes = routes[:t] + [new_route] + routes[t+1:]
                    new_max_val = max(route_distance(r) for r in new_routes)
                    new_total_val = sum(route_distance(r) for r in new_routes)
                    if new_max_val < best_max or (new_max_val == best_max and new_total_val < best_total):
                        best_max = new_max_val
                        best_total = new_total_val
                        best_truck = t
                        best_pos = pos
            routes[best_truck].insert(best_pos, cust)
        return routes

    def regret2_repair(routes, unassigned):
        routes = [list(r) for r in routes]
        unassigned = list(unassigned)
        while unassigned:
            best_info = None
            for cust in unassigned:
                best_max = float('inf')
                best_total = float('inf')
                second_best_max = float('inf')
                second_best_total = float('inf')
                best_truck = None
                best_pos = None
                for t, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_routes = routes[:t] + [new_route] + routes[t+1:]
                        new_max_val = max(route_distance(r) for r in new_routes)
                        new_total_val = sum(route_distance(r) for r in new_routes)
                        if new_max_val < best_max or (new_max_val == best_max and new_total_val < best_total):
                            second_best_max = best_max
                            second_best_total = best_total
                            best_max = new_max_val
                            best_total = new_total_val
                            best_truck = t
                            best_pos = pos
                        elif new_max_val < second_best_max or (new_max_val == second_best_max and new_total_val < second_best_total):
                            second_best_max = new_max_val
                            second_best_total = new_total_val
                if second_best_max == float('inf') or best_max == float('inf'):
                    regret = 0.0
                else:
                    regret = second_best_max - best_max
                if best_info is None:
                    best_info = (regret, best_max, best_total, cust, best_truck, best_pos)
                else:
                    if regret > best_info[0] or (regret == best_info[0] and (best_max < best_info[1] or (best_max == best_info[1] and best_total < best_info[2]))):
                        best_info = (regret, best_max, best_total, cust, best_truck, best_pos)
            regret, _, _, cust, best_truck, best_pos = best_info
            routes[best_truck].insert(best_pos, cust)
            unassigned.remove(cust)
        return routes

    for it in range(max_iter):
        destroy_op = random.choice([0, 1, 2])
        repair_op = random.choice([0, 1])

        if destroy_op == 0:
            removed, partial = worst_removal(current_routes, num_removals)
        elif destroy_op == 1:
            removed, partial = random_removal(current_routes, num_removals)
        else:
            removed, partial = route_removal(current_routes, num_removals)

        if repair_op == 0:
            new_routes = greedy_repair(partial, removed)
        else:
            new_routes = regret2_repair(partial, removed)

        new_max = max(route_distance(r) for r in new_routes)
        new_total = sum(route_distance(r) for r in new_routes)
        current_max = max(route_distance(r) for r in current_routes)
        current_total = sum(route_distance(r) for r in current_routes)

        delta = new_max - current_max
        if delta < 0 or (delta == 0 and new_total < current_total) or random.random() < math.exp(-delta / max(T, 1e-9)):
            current_routes = [list(r) for r in new_routes]
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < sum(route_distance(r) for r in best_routes)):
                best_max = new_max
                best_routes = [list(r) for r in new_routes]
                report_best_vrp(best_routes)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        # Restart if no improvement for 50 iterations (shake best solution)
        if no_improve >= 50:
            # Perturb best solution: randomly reinsert a few customers
            perturbed = [list(r) for r in best_routes]
            num_perturb = max(1, int(0.2 * (n-1)))
            to_reinsert = random.sample([c for r in perturbed for c in r[1:-1]], num_perturb)
            for r in perturbed:
                for c in to_reinsert:
                    while c in r[1:-1]:
                        r.remove(c)
            for cust in to_reinsert:
                best_max_tmp = float('inf')
                best_total_tmp = float('inf')
                best_truck_tmp = None
                best_pos_tmp = None
                for t, route in enumerate(perturbed):
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_routes = perturbed[:t] + [new_route] + perturbed[t+1:]
                        new_max_val = max(route_distance(r) for r in new_routes)
                        new_total_val = sum(route_distance(r) for r in new_routes)
                        if new_max_val < best_max_tmp or (new_max_val == best_max_tmp and new_total_val < best_total_tmp):
                            best_max_tmp = new_max_val
                            best_total_tmp = new_total_val
                            best_truck_tmp = t
                            best_pos_tmp = pos
                perturbed[best_truck_tmp].insert(best_pos_tmp, cust)
            current_routes = perturbed
            no_improve = 0
            T = T0  # reset temperature for restart

        T = T0 * (1 - it / max_iter)  # slower cooling? Actually same linear cooling, but with restart it resets

    return best_routes