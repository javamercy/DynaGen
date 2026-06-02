import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]

    def route_distance(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    global_best_routes = None
    global_best_max = float('inf')
    restarts = max(10, n // 5)

    for restart in range(restarts):
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]

        # Greedy insertion
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
                    if increase < best_increase - 1e-12 or (abs(increase - best_increase) < 1e-12 and (r_idx < best_route or (r_idx == best_route and pos < best_pos))):
                        best_increase = increase
                        best_route = r_idx
                        best_pos = pos
            routes[best_route].insert(best_pos, cust)

        best_routes = [r[:] for r in routes]
        best_max = max_distance(routes)
        report_best_vrp(best_routes)

        # Local search cycles
        for cycle in range(3):
            max_iter = (n - 1) * truck_count * 10
            no_improve_count = 0
            for iteration in range(max_iter):
                improved = False
                max_dist = max_distance(routes)
                longest_routes = [i for i, r in enumerate(routes) if route_distance(r) == max_dist]
                phases = ['2opt', 'relocate', 'swap', 'cross']
                random.shuffle(phases)
                for phase in phases:
                    if improved:
                        break
                    if phase == '2opt':
                        for r_idx in longest_routes:
                            route = routes[r_idx]
                            if len(route) <= 3:
                                continue
                            for i in range(1, len(route)-2):
                                for j in range(i+1, len(route)-1):
                                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                    old_dist = route_distance(route)
                                    new_dist = route_distance(new_route)
                                    if new_dist >= old_dist:
                                        continue
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != r_idx)
                                    new_max = max(new_dist, other_max)
                                    if new_max < best_max - 1e-12:
                                        routes[r_idx] = new_route
                                        best_routes = [r[:] for r in routes]
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                    elif phase == 'relocate':
                        for src in longest_routes:
                            route_src = routes[src]
                            if len(route_src) <= 2:
                                continue
                            for pos_src in range(1, len(route_src)-1):
                                cust = route_src[pos_src]
                                temp_src = route_src[:pos_src] + route_src[pos_src+1:]
                                dist_src = route_distance(temp_src)
                                for dst in range(truck_count):
                                    if dst == src:
                                        continue
                                    route_dst = routes[dst]
                                    for pos_dst in range(1, len(route_dst)):
                                        new_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                                        dist_dst = route_distance(new_dst)
                                        other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != src and x != dst)
                                        new_max = max(dist_src, dist_dst, other_max)
                                        if new_max < best_max - 1e-12:
                                            routes[src] = temp_src
                                            routes[dst] = new_dst
                                            best_routes = [r[:] for r in routes]
                                            best_max = new_max
                                            improved = True
                                            report_best_vrp(best_routes)
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                    elif phase == 'swap':
                        for t1 in longest_routes:
                            route1 = routes[t1]
                            if len(route1) <= 2:
                                continue
                            for t2 in range(truck_count):
                                if t2 == t1:
                                    continue
                                route2 = routes[t2]
                                if len(route2) <= 2:
                                    continue
                                for i in range(1, len(route1)-1):
                                    for j in range(1, len(route2)-1):
                                        cust1 = route1[i]
                                        cust2 = route2[j]
                                        new_route1 = route1[:i] + [cust2] + route1[i+1:]
                                        new_route2 = route2[:j] + [cust1] + route2[j+1:]
                                        dist1 = route_distance(new_route1)
                                        dist2 = route_distance(new_route2)
                                        other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                        new_max = max(dist1, dist2, other_max)
                                        if new_max < best_max - 1e-12:
                                            routes[t1] = new_route1
                                            routes[t2] = new_route2
                                            best_routes = [r[:] for r in routes]
                                            best_max = new_max
                                            improved = True
                                            report_best_vrp(best_routes)
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                    elif phase == 'cross':
                        for t1 in longest_routes:
                            route1 = routes[t1]
                            if len(route1) <= 2:
                                continue
                            for t2 in range(truck_count):
                                if t2 == t1:
                                    continue
                                route2 = routes[t2]
                                if len(route2) <= 2:
                                    continue
                                for i in range(1, len(route1)-1):
                                    for j in range(1, len(route2)-1):
                                        new_route1 = route1[:i] + route2[j:]
                                        new_route2 = route2[:j] + route1[i:]
                                        dist1 = route_distance(new_route1)
                                        dist2 = route_distance(new_route2)
                                        other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                        new_max = max(dist1, dist2, other_max)
                                        if new_max < best_max - 1e-12:
                                            routes[t1] = new_route1
                                            routes[t2] = new_route2
                                            best_routes = [r[:] for r in routes]
                                            best_max = new_max
                                            improved = True
                                            report_best_vrp(best_routes)
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                if improved:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= 5:
                        break

            # Adaptive shake: with probability 0.7 eliminate longest route; else remove half from random route
            if random.random() < 0.7:
                # Eliminate longest route entirely
                longest_route_idx = max(range(truck_count), key=lambda i: route_distance(routes[i]))
                longest_route = routes[longest_route_idx]
                if len(longest_route) > 2:
                    removed_customers = longest_route[1:-1]
                    routes[longest_route_idx] = [0, 0]
                    for cust in removed_customers:
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
                                if increase < best_increase - 1e-12 or (abs(increase - best_increase) < 1e-12 and (r_idx < best_route or (r_idx == best_route and pos < best_pos))):
                                    best_increase = increase
                                    best_route = r_idx
                                    best_pos = pos
                        routes[best_route].insert(best_pos, cust)
            else:
                # Remove half of customers from a random route (not necessarily longest)
                candidate_routes = [i for i, r in enumerate(routes) if len(r) > 2]
                if candidate_routes:
                    src_idx = random.choice(candidate_routes)
                    route = routes[src_idx]
                    inner = list(range(1, len(route)-1))
                    if len(inner) >= 2:
                        k = max(1, len(inner)//2)
                        remove_positions = sorted(random.sample(inner, k))
                        removed_customers = [route[p] for p in remove_positions]
                        # Remove customers in reverse order to preserve indices
                        for p in reversed(remove_positions):
                            del route[p]
                        # Reinsert removed customers greedily
                        for cust in removed_customers:
                            best_increase = float('inf')
                            best_route = -1
                            best_pos = -1
                            current_max = max_distance(routes)
                            for r_idx in range(truck_count):
                                # Skip route if it is the source? Actually allow reinsert into same route
                                route_dst = routes[r_idx]
                                for pos in range(1, len(route_dst)):
                                    prev = route_dst[pos-1]
                                    nxt = route_dst[pos]
                                    added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                                    new_route_dist = route_distance(route_dst) + added
                                    other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                                    new_max = max(new_route_dist, other_max)
                                    increase = new_max - current_max
                                    if increase < best_increase - 1e-12 or (abs(increase - best_increase) < 1e-12 and (r_idx < best_route or (r_idx == best_route and pos < best_pos))):
                                        best_increase = increase
                                        best_route = r_idx
                                        best_pos = pos
                            routes[best_route].insert(best_pos, cust)

            cur_max = max_distance(routes)
            if cur_max < best_max - 1e-12:
                best_routes = [r[:] for r in routes]
                best_max = cur_max
                report_best_vrp(best_routes)

        if best_max < global_best_max - 1e-12:
            global_best_max = best_max
            global_best_routes = [r[:] for r in best_routes]
            report_best_vrp(global_best_routes)

    if global_best_routes is None:
        global_best_routes = [r[:] for r in routes]
    return global_best_routes