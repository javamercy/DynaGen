import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    
    def route_dist(route):
        if len(route) <= 2:
            return 0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def max_route_dist(routes):
        return max(route_dist(r) for r in routes)
    
    def repair(routes):
        for r in routes:
            if r[0] != 0:
                r.insert(0, 0)
            if r[-1] != 0:
                r.append(0)
        return routes
    
    # Construction: greedy insertion minimizing new max
    def construct():
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in customers:
            best_increase = float('inf')
            best_moves = []
            current_max = max_route_dist(routes)
            for r_idx in range(truck_count):
                r = routes[r_idx]
                for pos in range(1, len(r)):
                    prev = r[pos-1]
                    nxt = r[pos]
                    added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_route_dist = route_dist(r) + added
                    other_max = max(route_dist(routes[i]) for i in range(truck_count) if i != r_idx)
                    new_max = max(new_route_dist, other_max)
                    increase = new_max - current_max
                    if increase < best_increase - 1e-12:
                        best_increase = increase
                        best_moves = [(r_idx, pos)]
                    elif abs(increase - best_increase) < 1e-12:
                        best_moves.append((r_idx, pos))
            r_idx, pos = random.choice(best_moves)
            routes[r_idx].insert(pos, cust)
        return routes
    
    # Local search: focus on longest route
    def local_search(routes):
        best_routes = [r[:] for r in routes]
        best_max = max_route_dist(routes)
        improved = True
        it = 0
        max_iter = (n - 1) * truck_count * 2
        while improved and it < max_iter:
            improved = False
            it += 1
            # Identify longest route
            dists = [route_dist(r) for r in routes]
            longest_idx = max(range(truck_count), key=lambda i: dists[i])
            
            # 2-opt on longest route
            route = routes[longest_idx]
            if len(route) > 3:
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        old_dist = dists[longest_idx]
                        new_dist = route_dist(new_route)
                        if new_dist >= old_dist:
                            continue
                        other_max = max(dists[k] for k in range(truck_count) if k != longest_idx)
                        new_max = max(new_dist, other_max)
                        if new_max < best_max - 1e-12:
                            routes[longest_idx] = new_route
                            best_routes = [r[:] for r in routes]
                            best_max = new_max
                            improved = True
                            report_best_vrp(best_routes)
                            break
                    if improved:
                        break
            if improved:
                continue
            
            # Relocate a customer from longest route to another to reduce max
            route = routes[longest_idx]
            if len(route) > 2:
                for pos in range(1, len(route)-1):
                    cust = route[pos]
                    temp = route[:pos] + route[pos+1:]
                    temp_dist = route_dist(temp)
                    for other_idx in range(truck_count):
                        if other_idx == longest_idx:
                            continue
                        other_route = routes[other_idx]
                        for other_pos in range(1, len(other_route)):
                            new_other = other_route[:other_pos] + [cust] + other_route[other_pos:]
                            new_other_dist = route_dist(new_other)
                            other_max = max(route_dist(routes[k]) for k in range(truck_count) if k != longest_idx and k != other_idx)
                            new_max = max(temp_dist, new_other_dist, other_max)
                            if new_max < best_max - 1e-12:
                                routes[longest_idx] = temp
                                routes[other_idx] = new_other
                                best_routes = [r[:] for r in routes]
                                best_max = new_max
                                improved = True
                                report_best_vrp(best_routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
        return best_routes
    
    # Perturbation: move a random customer from longest route to another random route
    def perturb(routes):
        dists = [route_dist(r) for r in routes]
        longest_idx = max(range(truck_count), key=lambda i: dists[i])
        route = routes[longest_idx]
        if len(route) <= 2:
            return routes
        # pick a random customer from longest route (excluding depot)
        pos = random.randint(1, len(route)-2)
        cust = route[pos]
        temp = route[:pos] + route[pos+1:]
        # insert into a random other route at best position to minimize new max
        other_idx = random.choice([i for i in range(truck_count) if i != longest_idx])
        other_route = routes[other_idx]
        current_max = max_route_dist(routes)
        best_increase = float('inf')
        best_pos = -1
        for other_pos in range(1, len(other_route)):
            prev = other_route[other_pos-1]
            nxt = other_route[other_pos]
            added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
            new_other_dist = route_dist(other_route) + added
            other_max = max(route_dist(routes[k]) for k in range(truck_count) if k != longest_idx and k != other_idx)
            new_max = max(route_dist(temp), new_other_dist, other_max)
            increase = new_max - current_max
            if increase < best_increase - 1e-12:
                best_increase = increase
                best_pos = other_pos
        if best_pos != -1:
            new_other = other_route[:best_pos] + [cust] + other_route[best_pos:]
            routes[longest_idx] = temp
            routes[other_idx] = new_other
        return routes
    
    global_best = None
    global_best_max = float('inf')
    restarts = min(3, n // 10 + 1)
    for _ in range(restarts):
        routes = construct()
        routes = local_search(routes)
        current_max = max_route_dist(routes)
        if current_max < global_best_max - 1e-12:
            global_best_max = current_max
            global_best = [r[:] for r in routes]
            report_best_vrp(global_best)
        # perturb and improve again
        if restarts > 1:
            routes = perturb(routes)
            routes = local_search(routes)
            current_max = max_route_dist(routes)
            if current_max < global_best_max - 1e-12:
                global_best_max = current_max
                global_best = [r[:] for r in routes]
                report_best_vrp(global_best)
    if global_best is None:
        global_best = [[0, 0] for _ in range(truck_count)]
    repair(global_best)
    return global_best