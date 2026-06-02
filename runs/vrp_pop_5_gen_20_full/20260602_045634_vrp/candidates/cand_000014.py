import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # Helper functions
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    # Construction: greedy min-max insertion
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
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

    # Local search phases (same as parent but with perturbation)
    max_restarts = (n - 1) * truck_count
    for restart in range(max_restarts):
        improved = False
        for _ in range(10 * (n - 1) * truck_count):
            # Phase 1: Intra-route 2-opt
            for r_idx in range(truck_count):
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
                        if new_max < best_max:
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
            if improved:
                continue
            # Phase 2: Inter-route relocate
            for src in range(truck_count):
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
                            if new_max < best_max:
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
            if improved:
                continue
            # Phase 3: Inter-route swap
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for t2 in range(t1+1, truck_count):
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
                            if new_max < best_max:
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
                continue
            # Phase 4: Cross-route 2-opt*
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for t2 in range(t1+1, truck_count):
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
                            if new_max < best_max:
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
        if not improved:
            # Perturbation: relocate up to 3 random customers
            num_perturb = min(3, n-1)
            cust_indices = random.sample(range(1, n), num_perturb)
            for cust in cust_indices:
                # Find current location
                found = False
                for r_idx, route in enumerate(routes):
                    for pos, c in enumerate(route):
                        if c == cust:
                            # Remove
                            route.pop(pos)
                            found = True
                            break
                    if found:
                        break
                # Insert into random route at random position (not first or last)
                target_route = random.randrange(truck_count)
                insert_pos = random.randint(1, len(routes[target_route])-1)
                routes[target_route].insert(insert_pos, cust)
            # Note: perturbation may worsen solution, but we keep it
            # Re-evaluate best (if perturbation accidentally improves, report)
            current_max = max_distance(routes)
            if current_max < best_max:
                best_routes = [r[:] for r in routes]
                best_max = current_max
                report_best_vrp(best_routes)
        else:
            # improvement found, continue next restart
            pass
    return best_routes