import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    random.seed(0)
    
    def route_distance(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    global_best_routes = None
    global_best_max = float('inf')
    
    restarts = 10
    for restart in range(restarts):
        # Construction: random order greedy insertion minimizing max increase
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in customers:
            best_increase = float('inf')
            best_route = -1
            best_pos = -1
            current_max = max_distance(routes)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_route_dist = route_distance(route) + added
                    other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                    new_max = max(new_route_dist, other_max)
                    increase = new_max - current_max
                    if increase < best_increase or (increase == best_increase and r_idx < best_route):
                        best_increase = increase
                        best_route = r_idx
                        best_pos = pos
            routes[best_route].insert(best_pos, cust)
        
        best_routes = [r[:] for r in routes]
        best_max = max_distance(routes)
        report_best_vrp(best_routes)
        
        # Simulated Annealing local search
        T = best_max * 0.1
        cooling_rate = 0.99
        max_iter = (n - 1) * truck_count * 50
        current_routes = [r[:] for r in routes]
        current_max = best_max
        
        for iteration in range(max_iter):
            # Choose a random neighborhood
            neighborhood = random.randint(0, 3)
            improved = False
            if neighborhood == 0:  # Intra-route 2-opt
                r_idx = random.randrange(truck_count)
                route = current_routes[r_idx]
                if len(route) <= 3:
                    continue
                i = random.randrange(1, len(route)-2)
                j = random.randrange(i+1, len(route)-1)
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_dist = route_distance(new_route)
                old_dist = route_distance(route)
                if new_dist < old_dist:
                    delta = 0  # improve, always accept
                else:
                    delta = new_dist - old_dist
                if delta <= 0 or random.random() < np.exp(-delta/T):
                    current_routes[r_idx] = new_route
                    new_max = max_distance(current_routes)
                    if new_max < current_max or (new_max >= current_max and delta > 0):
                        # We accepted, update current_max
                        current_max = new_max
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = [r[:] for r in current_routes]
                            report_best_vrp(best_routes)
                    improved = True
            elif neighborhood == 1:  # Inter-route relocate
                src = random.randrange(truck_count)
                route_src = current_routes[src]
                if len(route_src) <= 2:
                    continue
                pos_src = random.randrange(1, len(route_src)-1)
                cust = route_src[pos_src]
                temp_src = route_src[:pos_src] + route_src[pos_src+1:]
                dist_src = route_distance(temp_src)
                dst = random.randrange(truck_count)
                if dst == src:
                    continue
                route_dst = current_routes[dst]
                pos_dst = random.randrange(1, len(route_dst))
                new_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                dist_dst = route_distance(new_dst)
                other_max = max(route_distance(current_routes[x]) for x in range(truck_count) if x != src and x != dst)
                new_max = max(dist_src, dist_dst, other_max)
                delta = new_max - current_max
                if delta <= 0 or random.random() < np.exp(-delta/T):
                    current_routes[src] = temp_src
                    current_routes[dst] = new_dst
                    current_max = new_max
                    if new_max < best_max:
                        best_max = new_max
                        best_routes = [r[:] for r in current_routes]
                        report_best_vrp(best_routes)
                    improved = True
            elif neighborhood == 2:  # Inter-route swap
                t1 = random.randrange(truck_count)
                route1 = current_routes[t1]
                if len(route1) <= 2:
                    continue
                t2 = random.randrange(truck_count)
                if t2 == t1:
                    continue
                route2 = current_routes[t2]
                if len(route2) <= 2:
                    continue
                i = random.randrange(1, len(route1)-1)
                j = random.randrange(1, len(route2)-1)
                cust1 = route1[i]
                cust2 = route2[j]
                new_route1 = route1[:i] + [cust2] + route1[i+1:]
                new_route2 = route2[:j] + [cust1] + route2[j+1:]
                dist1 = route_distance(new_route1)
                dist2 = route_distance(new_route2)
                other_max = max(route_distance(current_routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                new_max = max(dist1, dist2, other_max)
                delta = new_max - current_max
                if delta <= 0 or random.random() < np.exp(-delta/T):
                    current_routes[t1] = new_route1
                    current_routes[t2] = new_route2
                    current_max = new_max
                    if new_max < best_max:
                        best_max = new_max
                        best_routes = [r[:] for r in current_routes]
                        report_best_vrp(best_routes)
                    improved = True
            else:  # Cross-route 2-opt*
                t1 = random.randrange(truck_count)
                route1 = current_routes[t1]
                if len(route1) <= 2:
                    continue
                t2 = random.randrange(truck_count)
                if t2 == t1:
                    continue
                route2 = current_routes[t2]
                if len(route2) <= 2:
                    continue
                i = random.randrange(1, len(route1)-1)
                j = random.randrange(1, len(route2)-1)
                new_route1 = route1[:i] + route2[j:]
                new_route2 = route2[:j] + route1[i:]
                dist1 = route_distance(new_route1)
                dist2 = route_distance(new_route2)
                other_max = max(route_distance(current_routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                new_max = max(dist1, dist2, other_max)
                delta = new_max - current_max
                if delta <= 0 or random.random() < np.exp(-delta/T):
                    current_routes[t1] = new_route1
                    current_routes[t2] = new_route2
                    current_max = new_max
                    if new_max < best_max:
                        best_max = new_max
                        best_routes = [r[:] for r in current_routes]
                        report_best_vrp(best_routes)
                    improved = True
            if improved:
                T *= cooling_rate
        # Update global best after restart
        if best_max < global_best_max:
            global_best_max = best_max
            global_best_routes = [r[:] for r in best_routes]
            report_best_vrp(global_best_routes)
    
    if global_best_routes is None:
        global_best_routes = best_routes
    return global_best_routes