import numpy as np
from itertools import permutations

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
    
    while unassigned:
        regrets = []
        for c in unassigned:
            data = get_insertion_data(c)
            if len(data) >= 3:
                regret = (data[1][0] - data[0][0]) + (data[2][0] - data[0][0])
            elif len(data) == 2:
                regret = data[1][0] - data[0][0]
            else:
                regret = 0.0
            regrets.append((regret, c, data[0][1]))
        regrets.sort(key=lambda x: (-x[0], x[1]))
        selected = regrets[0][1]
        best_pos = regrets[0][2]
        r_idx, i = best_pos
        route = routes[r_idx]
        route.insert(i, selected)
        route_distances[r_idx] = compute_route_distance(route)
        unassigned.remove(selected)
    
    current_routes = [list(r) for r in routes]
    current_max = max(route_distances)
    
    def report_best_vrp(routes):
        pass
    
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
                    if new_dist < route_distances[r_idx]:
                        route_distances[r_idx] = new_dist
                        current_routes[r_idx] = new_route
                        improved = True
                        new_max = max(route_distances)
                        if new_max < current_max:
                            current_max = new_max
                            report_best_vrp(current_routes)
                        break
                if improved:
                    break
            it += 1
    
    # Inter-route swap focused on longest route
    improved = True
    max_iters_swap = n * truck_count
    it_swap = 0
    while improved and it_swap < max_iters_swap:
        improved = False
        longest = np.argmax(route_distances)
        for r2 in range(truck_count):
            if r2 == longest:
                continue
            route1 = current_routes[longest]
            route2 = current_routes[r2]
            for i in range(1, len(route1)-1):
                for j in range(1, len(route2)-1):
                    new1 = route1[:i] + [route2[j]] + route1[i+1:]
                    new2 = route2[:j] + [route1[i]] + route2[j+1:]
                    new_dist1 = compute_route_distance(new1)
                    new_dist2 = compute_route_distance(new2)
                    new_max = max(route_distances[:longest] + route_distances[longest+1:r2] + route_distances[r2+1:], new_dist1, new_dist2)
                    if new_max < current_max:
                        current_routes[longest] = new1
                        current_routes[r2] = new2
                        route_distances[longest] = new_dist1
                        route_distances[r2] = new_dist2
                        current_max = new_max
                        improved = True
                        report_best_vrp(current_routes)
                        break
                if improved:
                    break
            if improved:
                break
        it_swap += 1
    
    # Inter-route relocation from longest route
    improved = True
    max_iters_reloc = n * truck_count
    it_reloc = 0
    while improved and it_reloc < max_iters_reloc:
        improved = False
        longest = np.argmax(route_distances)
        route_long = current_routes[longest]
        # iterate over customers in longest route (excluding depots)
        for idx in range(1, len(route_long)-1):
            customer = route_long[idx]
            # try to insert into any other route
            best_improvement = 0.0
            best_tuple = None
            for r2 in range(truck_count):
                if r2 == longest:
                    continue
                route2 = current_routes[r2]
                for pos in range(1, len(route2)):
                    # remove customer from longest route
                    new_long = route_long[:idx] + route_long[idx+1:]
                    new_dist_long = compute_route_distance(new_long)
                    # insert customer into route2
                    new_r2 = route2[:pos] + [customer] + route2[pos:]
                    new_dist_r2 = compute_route_distance(new_r2)
                    other_max = max(route_distances[:longest] + route_distances[longest+1:r2] + route_distances[r2+1:], default=0.0)
                    new_max = max(new_dist_long, new_dist_r2, other_max)
                    if new_max < current_max:
                        if current_max - new_max > best_improvement:
                            best_improvement = current_max - new_max
                            best_tuple = (r2, pos, new_long, new_r2, new_dist_long, new_dist_r2)
            if best_tuple is not None:
                r2, pos, new_long, new_r2, new_dist_long, new_dist_r2 = best_tuple
                current_routes[longest] = new_long
                current_routes[r2] = new_r2
                route_distances[longest] = new_dist_long
                route_distances[r2] = new_dist_r2
                current_max = max(route_distances)
                improved = True
                report_best_vrp(current_routes)
                break  # restart after first improvement
        it_reloc += 1
    
    return current_routes