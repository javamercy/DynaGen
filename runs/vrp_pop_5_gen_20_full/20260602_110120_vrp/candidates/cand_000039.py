import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0 for _ in range(truck_count)]
    
    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    for r in range(truck_count):
        route_distances[r] = compute_route_distance(routes[r])
    unassigned = set(range(1, n))
    
    def get_insertion_data(customer):
        data = []
        for r_idx, route in enumerate(routes):
            curr_dist = route_distances[r_idx]
            for i in range(1, len(route)):
                new_dist = curr_dist - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], customer] + distance_matrix[customer, route[i]]
                other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                cand_max = max(new_dist, other_max)
                data.append((cand_max, (r_idx, i)))
        data.sort(key=lambda x: x[0])
        return data
    
    # Construction with k = sqrt(remaining)
    while unassigned:
        regrets = []
        k = max(2, min(int(math.sqrt(len(unassigned))), len(unassigned)-1))
        for c in unassigned:
            data = get_insertion_data(c)
            m = len(data)
            if m >= k:
                regret = sum(data[i][0] - data[0][0] for i in range(1, k))
            elif m > 1:
                regret = data[1][0] - data[0][0]
            else:
                regret = 0.0
            tie_breaker = distance_matrix[0, c]
            regrets.append((regret, tie_breaker, c, data[0][1]))
        regrets.sort(key=lambda x: (-x[0], x[1]))
        selected = regrets[0][2]
        best_pos = regrets[0][3]
        r_idx, i = best_pos
        route = routes[r_idx]
        route.insert(i, selected)
        route_distances[r_idx] = compute_route_distance(route)
        unassigned.remove(selected)
    
    current_routes = [list(r) for r in routes]
    current_max = max(route_distances)
    best_routes = [list(r) for r in current_routes]
    best_max = current_max
    
    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        maxd = max(compute_route_distance(r) for r in routes)
        if maxd < best_max:
            best_max = maxd
            best_routes = [list(r) for r in routes]
    
    report_best_vrp(current_routes)
    
    def intra_2opt(routes, distances):
        improved = True
        it = 0
        while improved and it < n*10:
            improved = False
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_distance(new_route)
                        if new_dist < distances[r_idx]:
                            distances[r_idx] = new_dist
                            routes[r_idx] = new_route
                            improved = True
                            new_max = max(distances)
                            if new_max < best_max:
                                report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            it += 1
        return routes, distances
    
    def inter_relocate(routes, distances):
        improved = True
        it = 0
        while improved and it < n:
            improved = False
            avg_dist = sum(distances) / truck_count
            candidate_routes = [r for r in range(truck_count) if distances[r] > avg_dist]
            if not candidate_routes:
                break
            for src_idx in candidate_routes:
                src_route = routes[src_idx]
                if len(src_route) <= 2:
                    continue
                for idx in range(1, len(src_route)-1):
                    cust = src_route[idx]
                    new_src_route = src_route[:idx] + src_route[idx+1:]
                    new_dist_src = compute_route_distance(new_src_route)
                    for dest_idx in range(truck_count):
                        if dest_idx == src_idx:
                            continue
                        dest_route = routes[dest_idx]
                        for i in range(1, len(dest_route)):
                            new_dest_route = dest_route[:i] + [cust] + dest_route[i:]
                            new_dist_dest = compute_route_distance(new_dest_route)
                            other_max = max([distances[r] for r in range(truck_count) if r != src_idx and r != dest_idx], default=0.0)
                            new_max = max(other_max, new_dist_src, new_dist_dest)
                            if new_max < best_max:
                                routes[src_idx] = new_src_route
                                distances[src_idx] = new_dist_src
                                routes[dest_idx] = new_dest_route
                                distances[dest_idx] = new_dist_dest
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            it += 1
        return routes, distances
    
    def inter_swap(routes, distances):
        improved = True
        it = 0
        while improved and it < n*n:
            improved = False
            for r1 in range(truck_count):
                for r2 in range(r1+1, truck_count):
                    route1 = routes[r1]
                    route2 = routes[r2]
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i] + [route2[j]] + route1[i+1:]
                            new2 = route2[:j] + [route1[i]] + route2[j+1:]
                            new_dist1 = compute_route_distance(new1)
                            new_dist2 = compute_route_distance(new2)
                            other_max = max([distances[r] for r in range(truck_count) if r != r1 and r != r2], default=0.0)
                            new_max = max(other_max, new_dist1, new_dist2)
                            if new_max < best_max:
                                routes[r1] = new1
                                routes[r2] = new2
                                distances[r1] = new_dist1
                                distances[r2] = new_dist2
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            it += 1
        return routes, distances
    
    # Dynamic restart: decreasing fractions
    fractions = [0.3, 0.2, 0.1]
    for frac in fractions:
        # Improvement cycle
        current_routes, route_distances = intra_2opt(current_routes, route_distances)
        current_routes, route_distances = inter_relocate(current_routes, route_distances)
        current_routes, route_distances = inter_swap(current_routes, route_distances)
        current_routes, route_distances = inter_relocate(current_routes, route_distances)
        # Restart: remove fraction from longest route
        max_idx = max(range(truck_count), key=lambda r: route_distances[r])
        longest_route = current_routes[max_idx]
        if len(longest_route) > 2:
            customers = longest_route[1:-1]
            if customers:
                sorted_cust = sorted(customers, key=lambda c: -distance_matrix[0, c])
                remove_count = max(1, int(len(customers) * frac))
                removed = sorted_cust[:remove_count]
                new_route = [0] + [c for c in longest_route[1:-1] if c not in removed] + [0]
                current_routes[max_idx] = new_route
                route_distances[max_idx] = compute_route_distance(new_route)
                for cust in removed:
                    data = []
                    for r_idx, route in enumerate(current_routes):
                        curr_dist = route_distances[r_idx]
                        for i in range(1, len(route)):
                            new_dist = curr_dist - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], cust] + distance_matrix[cust, route[i]]
                            other_max = max([route_distances[r] for r in range(truck_count) if r != r_idx], default=0.0)
                            cand_max = max(new_dist, other_max)
                            data.append((cand_max, (r_idx, i)))
                    data.sort(key=lambda x: x[0])
                    best_pos = data[0][1]
                    r_idx, i = best_pos
                    current_routes[r_idx].insert(i, cust)
                    route_distances[r_idx] = compute_route_distance(current_routes[r_idx])
                report_best_vrp(current_routes)
    
    # Final improvement
    current_routes, route_distances = intra_2opt(current_routes, route_distances)
    current_routes, route_distances = inter_relocate(current_routes, route_distances)
    current_routes, route_distances = inter_swap(current_routes, route_distances)
    
    return best_routes