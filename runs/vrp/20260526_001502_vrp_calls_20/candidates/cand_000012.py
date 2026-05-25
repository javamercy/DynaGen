import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    # Deterministic cheapest insertion construction
    def cheapest_insertion():
        unvisited = list(range(1, n))
        routes = [[0, 0] for _ in range(truck_count)]
        route_lengths = [0.0] * truck_count
        while unvisited:
            best_cost = float('inf')
            best_route = None
            best_pos = None
            best_cust = None
            for cust in unvisited:
                for r in range(truck_count):
                    route = routes[r]
                    for pos in range(1, len(route)):
                        cost = (distance_matrix[route[pos-1]][cust] +
                                distance_matrix[cust][route[pos]] -
                                distance_matrix[route[pos-1]][route[pos]])
                        if cost < best_cost:
                            best_cost = cost
                            best_route = r
                            best_pos = pos
                            best_cust = cust
            routes[best_route].insert(best_pos, best_cust)
            route_lengths[best_route] += best_cost
            unvisited.remove(best_cust)
        # recompute lengths accurately
        for r in range(truck_count):
            dist = 0.0
            route = routes[r]
            for i in range(len(route)-1):
                dist += distance_matrix[route[i]][route[i+1]]
            route_lengths[r] = dist
        return routes, route_lengths

    # Improvement: relocate from longest route + 2-opt
    def improve(routes, route_lengths):
        max_iter = n * truck_count
        best_routes = [r[:] for r in routes]
        best_max = max(route_lengths)
        for _ in range(max_iter):
            improved = False
            # relocate from longest
            max_len = max(route_lengths)
            if max_len == 0:
                break
            longest = [i for i, l in enumerate(route_lengths) if l == max_len]
            for r in longest:
                route = routes[r]
                for idx in range(1, len(route)-1):
                    cust = route[idx]
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
                            if new_max < best_max - 1e-9:
                                routes[r] = new_route
                                route_lengths[r] = new_len_original
                                routes[r2].insert(pos, cust)
                                route_lengths[r2] = new_len_r2
                                best_max = max(route_lengths)
                                best_routes = [r[:] for r in routes]
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
            # 2-opt
            for r in range(truck_count):
                route = routes[r]
                if len(route) <= 3:
                    continue
                improved_2opt = True
                while improved_2opt:
                    improved_2opt = False
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            old = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                            new = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                            if new < old - 1e-9:
                                route[i:j+1] = reversed(route[i:j+1])
                                new_len = 0.0
                                for k in range(len(route)-1):
                                    new_len += distance_matrix[route[k]][route[k+1]]
                                route_lengths[r] = new_len
                                improved_2opt = True
                                current_max = max(route_lengths)
                                if current_max < best_max - 1e-9:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                break
                        if improved_2opt:
                            break
            # update best after 2-opt
            current_max = max(route_lengths)
            if current_max < best_max - 1e-9:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                improved = True
            if not improved:
                break
        return best_routes, best_max

    # Initial solution
    best_routes, best_lengths = cheapest_insertion()
    best_max = max(best_lengths)
    report_best_vrp([r[:] for r in best_routes])
    # Multi-start with perturbation
    num_restarts = max(5, n // 20)
    for _ in range(num_restarts):
        # Perturb: remove a few customers randomly and reinsert with penalty
        routes, lengths = cheapest_insertion()
        # remove up to 5 customers
        num_remove = min(5, n-1)
        remove_set = set(random.sample(range(1, n), num_remove))
        # Remove them
        for r in range(truck_count):
            route = routes[r]
            new_route = [0]
            for node in route[1:-1]:
                if node not in remove_set:
                    new_route.append(node)
            new_route.append(0)
            routes[r] = new_route
            # recompute length
            dist = 0.0
            for i in range(len(new_route)-1):
                dist += distance_matrix[new_route[i]][new_route[i+1]]
            lengths[r] = dist
        # Reinsert removed customers with penalty on long routes
        unvisited = list(remove_set)
        while unvisited:
            best_cost = float('inf')
            best_route = None
            best_pos = None
            best_cust = None
            max_len = max(lengths)
            for cust in unvisited:
                for r in range(truck_count):
                    route = routes[r]
                    penalty = 1.0 + 0.5 * (lengths[r] / (max_len + 1e-9))
                    for pos in range(1, len(route)):
                        cost = (distance_matrix[route[pos-1]][cust] +
                                distance_matrix[cust][route[pos]] -
                                distance_matrix[route[pos-1]][route[pos]])
                        cost *= penalty
                        if cost < best_cost:
                            best_cost = cost
                            best_route = r
                            best_pos = pos
                            best_cust = cust
            routes[best_route].insert(best_pos, best_cust)
            lengths[best_route] += (distance_matrix[routes[best_route][best_pos-1]][best_cust] +
                                    distance_matrix[best_cust][routes[best_route][best_pos+1]] -
                                    distance_matrix[routes[best_route][best_pos-1]][routes[best_route][best_pos+1]])
            unvisited.remove(best_cust)
        # Recompute all lengths accurately
        for r in range(truck_count):
            dist = 0.0
            route = routes[r]
            for i in range(len(route)-1):
                dist += distance_matrix[route[i]][route[i+1]]
            lengths[r] = dist
        # Improve
        routes, current_max = improve(routes, lengths)
        if current_max < best_max - 1e-9:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp([r[:] for r in best_routes])
    # Ensure empty trucks
    for r in range(truck_count):
        if len(best_routes[r]) == 0 or (len(best_routes[r])==2 and best_routes[r][0]==0 and best_routes[r][1]==0):
            best_routes[r] = [0, 0]
    # Ensure exactly one depot per route
    for r in range(truck_count):
        route = best_routes[r]
        if route[0] != 0:
            route.insert(0, 0)
        if route[-1] != 0:
            route.append(0)
    # Verify feasibility (optional, for safety)
    return best_routes