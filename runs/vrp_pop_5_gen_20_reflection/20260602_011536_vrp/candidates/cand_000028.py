import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()
    random.seed(42)
    
    def route_distance(route):
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))
    
    def objective(routes):
        return max(route_distance(r) for r in routes)
    
    def cheapest_insertion(unassigned, routes):
        while unassigned:
            best_max = float('inf')
            best_total = float('inf')
            best_node = None
            best_route = None
            best_pos = None
            for node in unassigned:
                for r in range(truck_count):
                    route = routes[r]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [node] + route[pos:]
                        new_route_dist = sum(dist[new_route[i]][new_route[i+1]] for i in range(len(new_route)-1))
                        current_max = 0
                        for rr in range(truck_count):
                            if rr == r:
                                d = new_route_dist
                            else:
                                d = sum(dist[routes[rr][i]][routes[rr][i+1]] for i in range(len(routes[rr])-1))
                            if d > current_max:
                                current_max = d
                        if current_max < best_max or (current_max == best_max and new_route_dist < best_total):
                            best_max = current_max
                            best_total = new_route_dist
                            best_node = node
                            best_route = r
                            best_pos = pos
            routes[best_route].insert(best_pos, best_node)
            unassigned.remove(best_node)
        return routes
    
    def local_search(routes):
        best_obj = objective(routes)
        improved = True
        for _ in range(20):
            if not improved:
                break
            improved = False
            # Relocate
            for node in range(1, n):
                curr_route = None
                curr_pos = None
                for r, route in enumerate(routes):
                    for pos, cust in enumerate(route):
                        if cust == node:
                            curr_route = r
                            curr_pos = pos
                            break
                    if curr_route is not None:
                        break
                if curr_route is None:
                    continue
                best_new_obj = float('inf')
                best_r = None
                best_pos = None
                for r in range(truck_count):
                    if r == curr_route:
                        continue
                    route = routes[r]
                    for pos in range(1, len(route)):
                        new_route_r = route[:pos] + [node] + route[pos:]
                        new_route_curr = routes[curr_route][:curr_pos] + routes[curr_route][curr_pos+1:]
                        if len(new_route_curr) < 2:
                            new_route_curr = [0, 0]
                        new_routes = [list(routes[i]) for i in range(truck_count)]
                        new_routes[r] = new_route_r
                        new_routes[curr_route] = new_route_curr
                        obj = max(route_distance(rr) for rr in new_routes)
                        if obj < best_new_obj:
                            best_new_obj = obj
                            best_r = r
                            best_pos = pos
                if best_new_obj < best_obj:
                    routes[curr_route].pop(curr_pos)
                    if len(routes[curr_route]) < 2:
                        routes[curr_route] = [0, 0]
                    routes[best_r].insert(best_pos, node)
                    best_obj = best_new_obj
                    improved = True
                    report_best_vrp([list(r) for r in routes])
                    break
            if improved:
                continue
            # Swap
            for i in range(1, n):
                ri = None
                pi = None
                for r, route in enumerate(routes):
                    for p, cust in enumerate(route):
                        if cust == i:
                            ri = r
                            pi = p
                            break
                    if ri is not None:
                        break
                if ri is None:
                    continue
                for j in range(i+1, n):
                    rj = None
                    pj = None
                    for r, route in enumerate(routes):
                        for p, cust in enumerate(route):
                            if cust == j:
                                rj = r
                                pj = p
                                break
                        if rj is not None:
                            break
                    if rj is None or rj == ri:
                        continue
                    route_i_without = routes[ri][:pi] + routes[ri][pi+1:]
                    if len(route_i_without) < 2:
                        route_i_without = [0, 0]
                    route_j_without = routes[rj][:pj] + routes[rj][pj+1:]
                    if len(route_j_without) < 2:
                        route_j_without = [0, 0]
                    best_obj_swap = float('inf')
                    best_pos_i = None
                    best_pos_j = None
                    for pos_i in range(1, len(route_i_without)):
                        for pos_j in range(1, len(route_j_without)):
                            new_route_i = route_i_without[:pos_i] + [j] + route_i_without[pos_i:]
                            new_route_j = route_j_without[:pos_j] + [i] + route_j_without[pos_j:]
                            new_routes = [list(routes[k]) for k in range(truck_count)]
                            new_routes[ri] = new_route_i
                            new_routes[rj] = new_route_j
                            obj = max(route_distance(rr) for rr in new_routes)
                            if obj < best_obj_swap:
                                best_obj_swap = obj
                                best_pos_i = pos_i
                                best_pos_j = pos_j
                    if best_obj_swap < best_obj:
                        routes[ri] = route_i_without[:best_pos_i] + [j] + route_i_without[best_pos_i:]
                        routes[rj] = route_j_without[:best_pos_j] + [i] + route_j_without[best_pos_j:]
                        best_obj = best_obj_swap
                        improved = True
                        report_best_vrp([list(r) for r in routes])
                        break
                if improved:
                    break
            if improved:
                continue
            # 2-opt within each route
            for r in range(truck_count):
                route = routes[r]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        other_max = 0
                        for rr in range(truck_count):
                            if rr != r:
                                d = route_distance(routes[rr])
                                if d > other_max:
                                    other_max = d
                        new_max = max(new_dist, other_max)
                        if new_max < best_obj:
                            routes[r] = new_route
                            best_obj = new_max
                            improved = True
                            report_best_vrp([list(r) for r in routes])
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Cross-route 2-opt*
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 3:
                    continue
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    if len(route2) <= 3:
                        continue
                    for a in range(1, len(route1)-2):
                        for b in range(1, len(route2)-2):
                            new_route1 = route1[:a+1] + route2[b+1:]
                            new_route2 = route2[:b+1] + route1[a+1:]
                            new_dist1 = route_distance(new_route1)
                            new_dist2 = route_distance(new_route2)
                            other_max = 0
                            for rr in range(truck_count):
                                if rr != r1 and rr != r2:
                                    d = route_distance(routes[rr])
                                    if d > other_max:
                                        other_max = d
                            new_max = max(new_dist1, new_dist2, other_max)
                            if new_max < best_obj:
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                best_obj = new_max
                                improved = True
                                report_best_vrp([list(r) for r in routes])
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes, best_obj
    
    # Initial build
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    routes = cheapest_insertion(unassigned, routes)
    best_solution = [list(r) for r in routes]
    best_obj = objective(routes)
    report_best_vrp(best_solution)
    
    max_restarts = 5
    for restart in range(max_restarts):
        # Local search
        routes, obj = local_search(routes)
        if obj < best_obj:
            best_obj = obj
            best_solution = [list(r) for r in routes]
            report_best_vrp(best_solution)
        if restart < max_restarts - 1:
            # Perturbation: remove customers from the worst route based on detour contribution
            max_dist = -1
            worst_route_idx = 0
            for r in range(truck_count):
                d = route_distance(routes[r])
                if d > max_dist:
                    max_dist = d
                    worst_route_idx = r
            worst_route = routes[worst_route_idx]
            # Compute savings for each customer in worst route
            customers = worst_route[1:-1]
            savings = []
            for i, node in enumerate(customers):
                prev = worst_route[i]  # actually index in route, adjust
                # get previous and next in route
                idx = worst_route.index(node)
                prev = worst_route[idx-1]
                nxt = worst_route[idx+1]
                saving = dist[prev][node] + dist[node][nxt] - dist[prev][nxt]
                savings.append((saving, node))
            savings.sort(reverse=True, key=lambda x: x[0])
            # Remove some customers with highest savings (detour) but add randomness
            remove_count = max(1, int(len(customers) * 0.3))
            # Randomly pick from top portion to avoid determinism
            top_indices = list(range(min(remove_count*2, len(savings))))
            random.shuffle(top_indices)
            selected_indices = top_indices[:remove_count]
            to_remove = [savings[i][1] for i in selected_indices]
            # Additionally remove random customers from any route to reach total 25%
            all_customers = []
            for r in range(truck_count):
                all_customers.extend(routes[r][1:-1])
            total_remove_target = max(1, int(n * 0.25))
            current_remove = len(to_remove)
            extra_needed = total_remove_target - current_remove
            if extra_needed > 0:
                remaining = [c for c in all_customers if c not in to_remove]
                random.shuffle(remaining)
                to_remove.extend(remaining[:extra_needed])
            # Remove to_remove from routes
            for node in to_remove:
                for r in range(truck_count):
                    if node in routes[r]:
                        routes[r].remove(node)
                        break
            for r in range(truck_count):
                if len(routes[r]) < 2:
                    routes[r] = [0, 0]
            # Reinsert
            unassigned = to_remove[:]
            routes = cheapest_insertion(unassigned, routes)
    return best_solution