import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def initial_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        route_distances = [0.0 for _ in range(truck_count)]
        unassigned = set(range(1, n))
        k = min(max(3, int(math.log2(n))), n-1)
        
        def insertion_data(customer):
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
        
        while unassigned:
            regrets = []
            for c in unassigned:
                data = insertion_data(c)
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
            r_idx, i = regrets[0][3]
            route = routes[r_idx]
            route.insert(i, selected)
            route_distances[r_idx] = compute_route_distance(route)
            unassigned.remove(selected)
        return routes, route_distances
    
    def two_opt(routes, route_distances, best_routes, best_max):
        improved = True
        it = 0
        max_iters = n * 10
        while improved and it < max_iters:
            improved = False
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_distance(new_route)
                        if new_dist < route_distances[r_idx]:
                            routes[r_idx] = new_route
                            route_distances[r_idx] = new_dist
                            improved = True
                            new_max = max(route_distances)
                            if new_max < best_max:
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            it += 1
        return routes, route_distances, best_routes, best_max
    
    def inter_relocate(routes, route_distances, best_routes, best_max):
        improved = True
        it = 0
        while improved and it < n:
            improved = False
            max_route_idx = max(range(truck_count), key=lambda r: route_distances[r])
            route_max = routes[max_route_idx]
            if len(route_max) <= 2:
                break
            for idx in range(1, len(route_max)-1):
                cust = route_max[idx]
                new_route_max = route_max[:idx] + route_max[idx+1:]
                new_dist_max = compute_route_distance(new_route_max)
                for r_idx in range(truck_count):
                    if r_idx == max_route_idx:
                        continue
                    route = routes[r_idx]
                    for i in range(1, len(route)):
                        new_route_other = route[:i] + [cust] + route[i:]
                        new_dist_other = compute_route_distance(new_route_other)
                        other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:max_route_idx] + route_distances[max_route_idx+1:])
                        new_max = max(other_max, new_dist_max, new_dist_other)
                        if new_max < best_max:
                            routes[max_route_idx] = new_route_max
                            route_distances[max_route_idx] = new_dist_max
                            routes[r_idx] = new_route_other
                            route_distances[r_idx] = new_dist_other
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            it += 1
        return routes, route_distances, best_routes, best_max
    
    def inter_swap(routes, route_distances, best_routes, best_max):
        improved = True
        it = 0
        max_iters = n * n
        while improved and it < max_iters:
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
                            other_max = max(route_distances[:r1] + route_distances[r1+1:r2] + route_distances[r2+1:])
                            new_max = max(other_max, new_dist1, new_dist2)
                            if new_max < best_max:
                                routes[r1] = new1
                                routes[r2] = new2
                                route_distances[r1] = new_dist1
                                route_distances[r2] = new_dist2
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                report_best_vrp(best_routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            it += 1
        return routes, route_distances, best_routes, best_max
    
    def perturbation(routes, route_distances):
        # Remove a random subset of customers (20%) and reinsert with regret-k
        unassigned = []
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) > 2:
                # Remove interior customers randomly
                interior = list(range(1, len(route)-1))
                random.shuffle(interior)
                num_remove = max(1, int(0.2 * (len(route)-2)))
                to_remove = interior[:num_remove]
                to_remove.sort(reverse=True)
                for idx in to_remove:
                    cust = route.pop(idx)
                    unassigned.append(cust)
                route_distances[r_idx] = compute_route_distance(route)
        random.shuffle(unassigned)
        k = min(max(3, int(math.log2(n))), n-1)
        def insertion_data(customer, routes, route_distances):
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
        while unassigned:
            regrets = []
            for c in unassigned:
                data = insertion_data(c, routes, route_distances)
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
            r_idx, i = regrets[0][3]
            routes[r_idx].insert(i, selected)
            route_distances[r_idx] = compute_route_distance(routes[r_idx])
            unassigned.remove(selected)
        return routes, route_distances
    
    # Initial construction
    routes, route_distances = initial_construction()
    current_max = max(route_distances)
    best_routes = [list(r) for r in routes]
    best_max = current_max
    report_best_vrp(best_routes)
    
    # Main improvement loops with restarts
    for restart in range(3):  # up to 3 restarts
        # Local search
        routes, route_distances = [list(r) for r in best_routes], [compute_route_distance(r) for r in best_routes]
        current_max = max(route_distances)
        
        # Apply 2-opt, relocate, swap sequentially
        routes, route_distances, best_routes, best_max = two_opt(routes, route_distances, best_routes, best_max)
        routes, route_distances, best_routes, best_max = inter_relocate(routes, route_distances, best_routes, best_max)
        routes, route_distances, best_routes, best_max = inter_swap(routes, route_distances, best_routes, best_max)
        
        # If not last restart, perturb
        if restart < 2:
            routes, route_distances = perturbation(routes, route_distances)
            current_max = max(route_distances)
            if current_max < best_max:
                best_max = current_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
    
    return best_routes