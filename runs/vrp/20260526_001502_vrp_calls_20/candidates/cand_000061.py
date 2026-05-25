import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    unvisited = list(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count

    # Helper: compute insertion cost
    def insertion_cost(route, customer, pos):
        return (distance_matrix[route[pos-1]][customer] +
                distance_matrix[customer][route[pos]] -
                distance_matrix[route[pos-1]][route[pos]])

    # Biased cheapest insertion
    while unvisited:
        best_cost = float('inf')
        best_route = None
        best_pos = None
        best_cust = None
        max_len = max(route_lengths) if any(route_lengths) else 1.0
        for cust in unvisited:
            for r in range(truck_count):
                route = routes[r]
                if len(route) == 2 and route_lengths[r] == 0:
                    # empty route: just depot
                    cost = 0.0
                else:
                    # Only consider positions that don't exceed some limit? All positions
                    # Find best position for this customer in this route
                    best_pos_local = None
                    best_cost_local = float('inf')
                    for pos in range(1, len(route)):
                        cost = insertion_cost(route, cust, pos)
                        # bias: penalize long routes
                        alpha = 0.5
                        bias = 1.0 + alpha * (route_lengths[r] / max_len)
                        cost_biased = cost * bias
                        if cost_biased < best_cost_local:
                            best_cost_local = cost_biased
                            best_pos_local = pos
                    # Use actual cost for comparison? Use biased cost for selection
                    if best_cost_local < best_cost:
                        best_cost = best_cost_local
                        best_route = r
                        best_pos = best_pos_local
                        best_cust = cust
        # Insert best customer
        routes[best_route].insert(best_pos, best_cust)
        # Update route_lengths (simple delta)
        route_lengths[best_route] += distance_matrix[routes[best_route][best_pos-1]][best_cust] + \
                                      distance_matrix[best_cust][routes[best_route][best_pos+1]] - \
                                      distance_matrix[routes[best_route][best_pos-1]][routes[best_route][best_pos+1]]
        unvisited.remove(best_cust)

    # Compute actual route lengths
    for r in range(truck_count):
        dist = 0.0
        for i in range(len(routes[r])-1):
            dist += distance_matrix[routes[r][i]][routes[r][i+1]]
        route_lengths[r] = dist

    best_routes = [r[:] for r in routes]
    best_max = max(route_lengths)
    report_best_vrp(best_routes)

    # Improvement: relocate from longest + 2-opt
    def improve_solution(routes, route_lengths):
        n_cust = n - 1
        max_iter = n_cust * truck_count * 2
        for _ in range(max_iter):
            improved = False
            # Relocate from longest route
            max_len = max(route_lengths)
            longest_routes = [i for i, l in enumerate(route_lengths) if l == max_len]
            for r in longest_routes:
                route = routes[r]
                for idx in range(1, len(route)-1):
                    cust = route[idx]
                    # removal cost
                    removed_cost = (distance_matrix[route[idx-1]][cust] +
                                    distance_matrix[cust][route[idx+1]] -
                                    distance_matrix[route[idx-1]][route[idx+1]])
                    new_route = route[:idx] + route[idx+1:]
                    new_len_original = route_lengths[r] - removed_cost
                    for r2 in range(truck_count):
                        if r2 == r:
                            continue
                        route2 = routes[r2]
                        for pos in range(1, len(route2)):
                            cost_ins = (distance_matrix[route2[pos-1]][cust] +
                                        distance_matrix[cust][route2[pos]] -
                                        distance_matrix[route2[pos-1]][route2[pos]])
                            new_len_r2 = route_lengths[r2] + cost_ins
                            new_max = max(new_len_original, new_len_r2)
                            if new_max < best_max:
                                # apply
                                routes[r] = new_route
                                route_lengths[r] = new_len_original
                                routes[r2].insert(pos, cust)
                                route_lengths[r2] = new_len_r2
                                best_max = new_max
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
            # 2-opt on each route
            for r in range(truck_count):
                route = routes[r]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old_dist = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                        new_dist = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                        if new_dist < old_dist - 1e-9:
                            route[i:j+1] = reversed(route[i:j+1])
                            new_len = 0.0
                            for k in range(len(route)-1):
                                new_len += distance_matrix[route[k]][route[k+1]]
                            route_lengths[r] = new_len
                            current_max = max(route_lengths)
                            if current_max < best_max:
                                best_max = current_max
                                improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
        return routes, route_lengths, best_max, best_routes

    # Apply improvement
    routes, route_lengths, best_max, best_routes = improve_solution(routes, route_lengths)
    report_best_vrp(best_routes)

    # Perturbation and improvement cycles
    max_cycles = min(10, n // 10)
    for _ in range(max_cycles):
        # Random perturbation: move a customer from longest route to a random shorter route
        max_len = max(route_lengths)
        longest_routes = [i for i, l in enumerate(route_lengths) if l == max_len]
        r_long = random.choice(longest_routes)
        route_long = routes[r_long]
        if len(route_long) <= 2:
            continue
        # pick random customer from longest route (excluding depots)
        idx = random.randint(1, len(route_long)-2)
        cust = route_long[idx]
        # remove customer
        removal_cost = (distance_matrix[route_long[idx-1]][cust] +
                        distance_matrix[cust][route_long[idx+1]] -
                        distance_matrix[route_long[idx-1]][route_long[idx+1]])
        new_route_long = route_long[:idx] + route_long[idx+1:]
        new_len_long = route_lengths[r_long] - removal_cost
        # choose a random other route
        other_routes = [r for r in range(truck_count) if r != r_long]
        r_short = random.choice(other_routes)
        route_short = routes[r_short]
        pos = random.randint(1, len(route_short))
        ins_cost = (distance_matrix[route_short[pos-1]][cust] +
                    distance_matrix[cust][route_short[pos]] -
                    distance_matrix[route_short[pos-1]][route_short[pos]])
        new_len_short = route_lengths[r_short] + ins_cost
        # apply
        routes[r_long] = new_route_long
        route_lengths[r_long] = new_len_long
        routes[r_short].insert(pos, cust)
        route_lengths[r_short] = new_len_short
        # improve from perturbed state
        routes, route_lengths, current_max, current_best = improve_solution(routes, route_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        else:
            # revert perturbation? Not necessary since improvement may have changed state; we keep best
            pass

    # Ensure empty trucks
    for r in range(truck_count):
        if len(best_routes[r]) == 0:
            best_routes[r] = [0, 0]
    return best_routes