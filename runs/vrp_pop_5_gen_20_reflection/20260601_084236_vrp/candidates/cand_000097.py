import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count >= n:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        if len(route) < 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    def total_distance(routes):
        return sum(route_distance(r) for r in routes)

    def two_opt(route):
        if len(route) <= 3:
            return route
        improved = True
        best = route[:]
        best_d = route_distance(route)
        max_iter = len(route) * 2
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d = route_distance(new_route)
                    if d < best_d - 1e-12:
                        best_d = d
                        best = new_route
                        improved = True
            route = best
            iteration += 1
        return best

    def farthest_insertion_construction():
        unvisited = set(range(1, n))
        routes = [[] for _ in range(truck_count)]
        # Initialize each route with a seed customer farthest from depot
        distances_from_depot = [distance_matrix[0, i] for i in range(1, n)]
        sorted_customers = sorted(range(1, n), key=lambda i: distances_from_depot[i-1], reverse=True)
        for idx, cust in enumerate(sorted_customers[:truck_count]):
            routes[idx].append(cust)
            unvisited.remove(cust)
        # For remaining customers, insert into the route that minimizes increase in route distance
        for cust in list(unvisited):
            best_route = 0
            best_increase = float('inf')
            best_pos = 0
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if not route:
                    # Insert as only customer
                    increase = 2 * distance_matrix[0, cust]
                    if increase < best_increase - 1e-12:
                        best_increase = increase
                        best_route = r_idx
                        best_pos = 0
                else:
                    for pos in range(len(route)+1):
                        before = distance_matrix[0, route[0]] if pos == 0 else distance_matrix[route[pos-1], route[pos]]
                        after = distance_matrix[route[-1], 0] if pos == len(route) else distance_matrix[route[pos], route[pos+1]]
                        # Actually compute increase properly
                        if pos == 0:
                            increase = distance_matrix[0, cust] + distance_matrix[cust, route[0]] - distance_matrix[0, route[0]]
                        elif pos == len(route):
                            increase = distance_matrix[route[-1], cust] + distance_matrix[cust, 0] - distance_matrix[route[-1], 0]
                        else:
                            increase = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        if increase < best_increase - 1e-12:
                            best_increase = increase
                            best_route = r_idx
                            best_pos = pos
            routes[best_route].insert(best_pos, cust)
        # Build full tours with depot
        tours = []
        for route in routes:
            if not route:
                tours.append([0, 0])
            else:
                tour = [0] + route + [0]
                tours.append(two_opt(tour))
        return tours

    best_routes = None
    best_max = float('inf')
    restarts = min(truck_count * 5, 30)
    for restart in range(restarts):
        routes = farthest_insertion_construction()
        current_max = max_distance(routes)
        current_total = total_distance(routes)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Intra-route 2-opt one pass
        for t in range(truck_count):
            routes[t] = two_opt(routes[t])
        cur_max = max_distance(routes)
        cur_total = total_distance(routes)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Inter-route 2-opt* best improvement (bounded)
        max_iter = n * 2
        iteration = 0
        improved = True
        while improved and iteration < max_iter:
            improved = False
            best_improv = None
            best_new_max = float('inf')
            best_new_total = float('inf')
            for t1 in range(truck_count):
                for t2 in range(t1+1, truck_count):
                    r1 = routes[t1]
                    r2 = routes[t2]
                    if len(r1) <= 2 or len(r2) <= 2:
                        continue
                    for i in range(1, len(r1)-1):
                        for j in range(1, len(r2)-1):
                            new_r1 = r1[:i+1] + r2[j+1:]
                            new_r2 = r2[:j+1] + r1[i+1:]
                            d1 = route_distance(new_r1)
                            d2 = route_distance(new_r2)
                            other_max = 0.0
                            other_total = 0.0
                            for idx, r in enumerate(routes):
                                if idx not in (t1, t2):
                                    d = route_distance(r)
                                    if d > other_max:
                                        other_max = d
                                    other_total += d
                            cand_max = max(d1, d2, other_max)
                            cand_total = d1 + d2 + other_total
                            if cand_max < best_new_max - 1e-12 or (abs(cand_max - best_new_max) < 1e-12 and cand_total < best_new_total - 1e-12):
                                best_new_max = cand_max
                                best_new_total = cand_total
                                best_improv = (t1, t2, i, j, new_r1, new_r2)
            if best_improv is not None and (best_new_max < cur_max - 1e-12 or (abs(best_new_max - cur_max) < 1e-12 and best_new_total < cur_total - 1e-12)):
                t1, t2, i, j, new_r1, new_r2 = best_improv
                routes[t1] = two_opt(new_r1)
                routes[t2] = two_opt(new_r2)
                cur_max = max_distance(routes)
                cur_total = total_distance(routes)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                improved = True
            iteration += 1

        # Max-route reduction via relocation (bounded)
        for _ in range(n):
            max_idx = max(range(truck_count), key=lambda t: route_distance(routes[t]))
            max_route = routes[max_idx]
            if len(max_route) <= 2:
                break
            found = False
            for idx in range(1, len(max_route)-1):
                cust = max_route[idx]
                new_max_route = max_route[:idx] + max_route[idx+1:]
                d_max_new = route_distance(new_max_route)
                for t2 in range(truck_count):
                    if t2 == max_idx:
                        continue
                    r2 = routes[t2]
                    for pos in range(1, len(r2)):
                        new_r2 = r2[:pos] + [cust] + r2[pos:]
                        d2_new = route_distance(new_r2)
                        other_max = 0.0
                        other_total = 0.0
                        for idx2, r in enumerate(routes):
                            if idx2 not in (max_idx, t2):
                                d = route_distance(r)
                                if d > other_max:
                                    other_max = d
                                other_total += d
                        cand_max = max(d_max_new, d2_new, other_max)
                        cand_total = d_max_new + d2_new + other_total
                        if cand_max < cur_max - 1e-12 or (abs(cand_max - cur_max) < 1e-12 and cand_total < cur_total - 1e-12):
                            routes[max_idx] = two_opt(new_max_route)
                            routes[t2] = two_opt(new_r2)
                            cur_max = max_distance(routes)
                            cur_total = total_distance(routes)
                            if cur_max < best_max - 1e-12:
                                best_max = cur_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(best_routes)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break

    # Optional final 2-opt on best routes
    if best_routes:
        for t in range(truck_count):
            best_routes[t] = two_opt(best_routes[t])
        best_max = max_distance(best_routes)
        report_best_vrp(best_routes)

    return best_routes