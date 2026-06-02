import numpy as np
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    
    def compute_route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d
    
    # construction
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    unassigned.sort(key=lambda c: (-distance_matrix[0][c], c))
    route_dists = [compute_route_dist(r) for r in routes]
    for cust in unassigned:
        best_new_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                prev = route[pos-1]
                succ = route[pos]
                increase = distance_matrix[prev][cust] + distance_matrix[cust][succ] - distance_matrix[prev][succ]
                new_route_dist = route_dists[r_idx] + increase
                new_max = new_route_dist
                for other_idx, d in enumerate(route_dists):
                    if other_idx != r_idx and d > new_max:
                        new_max = d
                if new_max < best_new_max or (new_max == best_new_max and r_idx < best_route_idx):
                    best_new_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
        route_dists[best_route_idx] = compute_route_dist(route)
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)
    
    # helpers for moves
    def apply_relocate(routes, r_idx, pos, cust, new_r_idx, new_pos):
        routes[r_idx].pop(pos)
        routes[new_r_idx].insert(new_pos, cust)
        route_dists[r_idx] = compute_route_dist(routes[r_idx])
        route_dists[new_r_idx] = compute_route_dist(routes[new_r_idx])
    
    def apply_swap(routes, r1, pos1, r2, pos2):
        cust1 = routes[r1][pos1]
        cust2 = routes[r2][pos2]
        routes[r1][pos1] = cust2
        routes[r2][pos2] = cust1
        route_dists[r1] = compute_route_dist(routes[r1])
        route_dists[r2] = compute_route_dist(routes[r2])
    
    def apply_two_opt(routes, r_idx, i, j):
        route = routes[r_idx]
        route[i:j+1] = reversed(route[i:j+1])
        route_dists[r_idx] = compute_route_dist(route)
    
    def apply_cross(routes, r1, pos1, r2, pos2):
        tail1 = routes[r1][pos1+1:]
        tail2 = routes[r2][pos2+1:]
        new_route1 = routes[r1][:pos1+1] + tail2
        new_route2 = routes[r2][:pos2+1] + tail1
        routes[r1] = new_route1
        routes[r2] = new_route2
        route_dists[r1] = compute_route_dist(new_route1)
        route_dists[r2] = compute_route_dist(new_route2)
    
    # deterministic perturbation modes
    def perturb_mode0(routes, route_dists):
        current_max = max(route_dists)
        max_routes = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
        r_idx = max_routes[0]
        route = routes[r_idx]
        customers = [node for node in route if node != 0]
        if not customers:
            return
        cust = max(customers)
        pos = route.index(cust)
        route.pop(pos)
        route_dists[r_idx] = compute_route_dist(route)
        min_route_idx = min(range(truck_count), key=lambda i: (route_dists[i], i))
        best_increase = float('inf')
        best_pos = -1
        for pos2 in range(1, len(routes[min_route_idx])):
            prev2 = routes[min_route_idx][pos2-1]
            succ2 = routes[min_route_idx][pos2]
            increase = distance_matrix[prev2][cust] + distance_matrix[cust][succ2] - distance_matrix[prev2][succ2]
            if increase < best_increase:
                best_increase = increase
                best_pos = pos2
        routes[min_route_idx].insert(best_pos, cust)
        route_dists[min_route_idx] = compute_route_dist(routes[min_route_idx])
    
    def perturb_mode1(routes, route_dists):
        current_max = max(route_dists)
        max_routes = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
        r_idx = max_routes[0]
        route = routes[r_idx]
        customers = [node for node in route if node != 0]
        if not customers:
            return
        farthest_dist = -1
        farthest_cust = -1
        for c in customers:
            d = distance_matrix[0][c]
            if d > farthest_dist or (abs(d - farthest_dist) < 1e-12 and c > farthest_cust):
                farthest_dist = d
                farthest_cust = c
        cust = farthest_cust
        pos = route.index(cust)
        route.pop(pos)
        route_dists[r_idx] = compute_route_dist(route)
        min_route_idx = min(range(truck_count), key=lambda i: (route_dists[i], i))
        best_increase = float('inf')
        best_pos = -1
        for pos2 in range(1, len(routes[min_route_idx])):
            prev2 = routes[min_route_idx][pos2-1]
            succ2 = routes[min_route_idx][pos2]
            increase = distance_matrix[prev2][cust] + distance_matrix[cust][succ2] - distance_matrix[prev2][succ2]
            if increase < best_increase:
                best_increase = increase
                best_pos = pos2
        routes[min_route_idx].insert(best_pos, cust)
        route_dists[min_route_idx] = compute_route_dist(routes[min_route_idx])
    
    def best_improve_loop(routes, route_dists):
        # best-improvement: find the best move that reduces max distance
        improved = True
        while improved:
            improved = False
            current_max = max(route_dists)
            best_move = None
            best_new_max = current_max
            # relocate
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)-1):
                    cust = route[pos]
                    prev = route[pos-1]
                    succ = route[pos+1]
                    removal_change = distance_matrix[prev][succ] - (distance_matrix[prev][cust] + distance_matrix[cust][succ])
                    new_dist_r = route_dists[r_idx] + removal_change
                    for other_idx in range(truck_count):
                        if other_idx == r_idx:
                            continue
                        other_route = routes[other_idx]
                        for insert_pos in range(1, len(other_route)):
                            prev2 = other_route[insert_pos-1]
                            succ2 = other_route[insert_pos]
                            insertion_change = distance_matrix[prev2][cust] + distance_matrix[cust][succ2] - distance_matrix[prev2][succ2]
                            new_dist_other = route_dists[other_idx] + insertion_change
                            new_max = max(new_dist_r, new_dist_other)
                            for idx, d in enumerate(route_dists):
                                if idx != r_idx and idx != other_idx and d > new_max:
                                    new_max = d
                            if new_max < best_new_max - 1e-12:
                                best_new_max = new_max
                                best_move = ('relocate', r_idx, pos, cust, other_idx, insert_pos)
            # swap
            for r1 in range(truck_count):
                route1 = routes[r1]
                for pos1 in range(1, len(route1)-1):
                    cust1 = route1[pos1]
                    for r2 in range(truck_count):
                        if r2 == r1:
                            continue
                        route2 = routes[r2]
                        for pos2 in range(1, len(route2)-1):
                            cust2 = route2[pos2]
                            prev1 = route1[pos1-1]
                            succ1 = route1[pos1+1]
                            remove1_change = distance_matrix[prev1][succ1] - (distance_matrix[prev1][cust1] + distance_matrix[cust1][succ1])
                            prev2 = route2[pos2-1]
                            succ2 = route2[pos2+1]
                            remove2_change = distance_matrix[prev2][succ2] - (distance_matrix[prev2][cust2] + distance_matrix[cust2][succ2])
                            insert1_change = distance_matrix[prev1][cust2] + distance_matrix[cust2][succ1] - distance_matrix[prev1][succ1]
                            insert2_change = distance_matrix[prev2][cust1] + distance_matrix[cust1][succ2] - distance_matrix[prev2][succ2]
                            new_dist_r1 = route_dists[r1] + remove1_change + insert1_change
                            new_dist_r2 = route_dists[r2] + remove2_change + insert2_change
                            new_max = max(new_dist_r1, new_dist_r2)
                            for idx, d in enumerate(route_dists):
                                if idx != r1 and idx != r2 and d > new_max:
                                    new_max = d
                            if new_max < best_new_max - 1e-12:
                                best_new_max = new_max
                                best_move = ('swap', r1, pos1, r2, pos2)
            # 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old_edges = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                        new_edges = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                        change = new_edges - old_edges
                        new_dist = route_dists[r_idx] + change
                        new_max = new_dist
                        for idx, d in enumerate(route_dists):
                            if idx != r_idx and d > new_max:
                                new_max = d
                        if new_max < best_new_max - 1e-12:
                            best_new_max = new_max
                            best_move = ('2opt', r_idx, i, j)
            # cross
            for r1 in range(truck_count):
                route1 = routes[r1]
                for pos1 in range(0, len(route1)-1):
                    if pos1 == len(route1)-1:
                        continue
                    for r2 in range(truck_count):
                        if r2 == r1:
                            continue
                        route2 = routes[r2]
                        for pos2 in range(0, len(route2)-1):
                            if pos2 == len(route2)-1:
                                continue
                            delta1 = distance_matrix[route1[pos1]][route2[pos2+1]] - distance_matrix[route1[pos1]][route1[pos1+1]]
                            delta2 = distance_matrix[route2[pos2]][route1[pos1+1]] - distance_matrix[route2[pos2]][route2[pos2+1]]
                            new_dist_r1 = route_dists[r1] + delta1
                            new_dist_r2 = route_dists[r2] + delta2
                            new_max = max(new_dist_r1, new_dist_r2)
                            for idx, d in enumerate(route_dists):
                                if idx != r1 and idx != r2 and d > new_max:
                                    new_max = d
                            if new_max < best_new_max - 1e-12:
                                best_new_max = new_max
                                best_move = ('cross', r1, pos1, r2, pos2)
            if best_move is not None:
                improved = True
                if best_move[0] == 'relocate':
                    _, r_idx, pos, cust, other_idx, insert_pos = best_move
                    apply_relocate(routes, r_idx, pos, cust, other_idx, insert_pos)
                elif best_move[0] == 'swap':
                    _, r1, pos1, r2, pos2 = best_move
                    apply_swap(routes, r1, pos1, r2, pos2)
                elif best_move[0] == '2opt':
                    _, r_idx, i, j = best_move
                    apply_two_opt(routes, r_idx, i, j)
                elif best_move[0] == 'cross':
                    _, r1, pos1, r2, pos2 = best_move
                    apply_cross(routes, r1, pos1, r2, pos2)
                current_max = max(route_dists)
                if current_max < best_max - 1e-12:
                    best_max = current_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
        return routes, route_dists
    
    max_restarts = 5
    for restart in range(max_restarts + 1):
        if restart > 0:
            routes = [list(r) for r in best_routes]
            route_dists = [compute_route_dist(r) for r in routes]
            if restart % 2 == 1:
                perturb_mode0(routes, route_dists)
            else:
                perturb_mode1(routes, route_dists)
        else:
            routes = [list(r) for r in best_routes]
            route_dists = [compute_route_dist(r) for r in routes]
        
        max_iter = min(n * 100, 5000)
        for _ in range(max_iter):
            old_max = max(route_dists)
            routes, route_dists = best_improve_loop(routes, route_dists)
            new_max = max(route_dists)
            if new_max < best_max - 1e-12:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
            if abs(new_max - old_max) < 1e-12:
                break
    return best_routes