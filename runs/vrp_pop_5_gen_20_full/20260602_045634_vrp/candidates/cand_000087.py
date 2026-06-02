import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    
    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    best_routes = None
    best_max = float('inf')
    restarts = max(10, n // 5)

    for restart in range(restarts):
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]

        # Construction: regret insertion
        for cust in customers:
            best_route = -1
            best_pos = -1
            best_increase = float('inf')
            current_max = max_dist(routes)
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_route_dist = route_dist(route) + added
                    other_max = max(route_dist(routes[i]) for i in range(truck_count) if i != r_idx) if truck_count > 1 else 0
                    new_max = max(new_route_dist, other_max)
                    increase = new_max - current_max
                    if increase < best_increase - 1e-12 or (abs(increase - best_increase) < 1e-12 and (r_idx < best_route or (r_idx == best_route and pos < best_pos))):
                        best_increase = increase
                        best_route = r_idx
                        best_pos = pos
            routes[best_route].insert(best_pos, cust)

        cur_max = max_dist(routes)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Local search cycles
        for cycle in range(3):
            max_iter = (n - 1) * truck_count * 10
            no_improve = 0
            for iteration in range(max_iter):
                improved = False
                dists = [route_dist(r) for r in routes]
                sorted_indices = sorted(range(truck_count), key=lambda i: -dists[i])
                longest = sorted_indices[0]

                # intra 2-opt on longest
                r = routes[longest]
                if len(r) > 3:
                    for i in range(1, len(r)-2):
                        for j in range(i+1, len(r)-1):
                            new_r = r[:i] + r[i:j+1][::-1] + r[j+1:]
                            new_dist = route_dist(new_r)
                            if new_dist < route_dist(r) - 1e-12:
                                other_max = max(route_dist(routes[x]) for x in range(truck_count) if x != longest)
                                new_max = max(new_dist, other_max)
                                if new_max < best_max - 1e-12:
                                    routes[longest] = new_r
                                    best_routes = [r[:] for r in routes]
                                    best_max = new_max
                                    improved = True
                                    report_best_vrp(best_routes)
                                    break
                        if improved:
                            break
                if improved:
                    no_improve = 0
                    continue

                # inter 2-opt*
                for r2 in range(truck_count):
                    if r2 == longest:
                        continue
                    r1 = routes[longest]
                    if len(r1) <= 2 or len(routes[r2]) <= 2:
                        continue
                    for i in range(1, len(r1)-1):
                        for j in range(1, len(routes[r2])-1):
                            new_r1 = r1[:i] + routes[r2][j:]
                            new_r2 = routes[r2][:j] + r1[i:]
                            dist1 = route_dist(new_r1)
                            dist2 = route_dist(new_r2)
                            other_max = max(route_dist(routes[x]) for x in range(truck_count) if x != longest and x != r2)
                            new_max = max(dist1, dist2, other_max)
                            if new_max < best_max - 1e-12:
                                routes[longest] = new_r1
                                routes[r2] = new_r2
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
                    no_improve = 0
                    continue

                # relocate from longest
                r1 = routes[longest]
                if len(r1) > 2:
                    for pos in range(1, len(r1)-1):
                        cust = r1[pos]
                        temp_r1 = r1[:pos] + r1[pos+1:]
                        dist_r1 = route_dist(temp_r1)
                        for dst in range(truck_count):
                            if dst == longest:
                                continue
                            for pos2 in range(1, len(routes[dst])):
                                new_dst = routes[dst][:pos2] + [cust] + routes[dst][pos2:]
                                dist_dst = route_dist(new_dst)
                                other_max = max(route_dist(routes[x]) for x in range(truck_count) if x != longest and x != dst)
                                new_max = max(dist_r1, dist_dst, other_max)
                                if new_max < best_max - 1e-12:
                                    routes[longest] = temp_r1
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
                    no_improve = 0
                    continue

                # swap between longest and any other
                for r2 in range(truck_count):
                    if r2 == longest:
                        continue
                    r1 = routes[longest]
                    if len(r1) <= 2 or len(routes[r2]) <= 2:
                        continue
                    for i in range(1, len(r1)-1):
                        for j in range(1, len(routes[r2])-1):
                            cust1 = r1[i]
                            cust2 = routes[r2][j]
                            new_r1 = r1[:i] + [cust2] + r1[i+1:]
                            new_r2 = routes[r2][:j] + [cust1] + routes[r2][j+1:]
                            dist1 = route_dist(new_r1)
                            dist2 = route_dist(new_r2)
                            other_max = max(route_dist(routes[x]) for x in range(truck_count) if x != longest and x != r2)
                            new_max = max(dist1, dist2, other_max)
                            if new_max < best_max - 1e-12:
                                routes[longest] = new_r1
                                routes[r2] = new_r2
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
                    no_improve = 0
                    continue

                # cross exchange
                for r2 in range(truck_count):
                    if r2 == longest:
                        continue
                    r1 = routes[longest]
                    if len(r1) <= 2 or len(routes[r2]) <= 2:
                        continue
                    for i in range(1, len(r1)-1):
                        for j in range(1, len(routes[r2])-1):
                            new_r1 = r1[:i] + routes[r2][j:]
                            new_r2 = routes[r2][:j] + r1[i:]
                            dist1 = route_dist(new_r1)
                            dist2 = route_dist(new_r2)
                            other_max = max(route_dist(routes[x]) for x in range(truck_count) if x != longest and x != r2)
                            new_max = max(dist1, dist2, other_max)
                            if new_max < best_max - 1e-12:
                                routes[longest] = new_r1
                                routes[r2] = new_r2
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
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= 10:
                        break

            # Shake
            dists = [route_dist(r) for r in routes]
            sorted_indices = sorted(range(truck_count), key=lambda i: -dists[i])
            longest_idx = sorted_indices[0]
            second_idx = sorted_indices[1] if truck_count > 1 else None

            def shake_route(route_idx, fraction):
                route = routes[route_idx]
                if len(route) <= 4:
                    return False
                edges = [(route[i], route[i+1], distance_matrix[route[i], route[i+1]]) for i in range(len(route)-1)]
                internal_edges = [e for e in edges if e[0] != 0 and e[1] != 0]
                if not internal_edges:
                    return False
                internal_edges.sort(key=lambda e: -e[2])
                num_remove = max(1, int(len(internal_edges) * fraction))
                remove_edges = internal_edges[:num_remove]
                remove_set = set()
                for u, v, _ in remove_edges:
                    if u != 0:
                        remove_set.add(u)
                    if v != 0:
                        remove_set.add(v)
                new_route = [0]
                for node in route[1:-1]:
                    if node not in remove_set:
                        new_route.append(node)
                new_route.append(0)
                if len(new_route) < 2:
                    new_route = [0, 0]
                routes[route_idx] = new_route
                removed = list(remove_set)
                random.shuffle(removed)
                for cust in removed:
                    best_route = -1
                    best_pos = -1
                    best_increase = float('inf')
                    current_max = max_dist(routes)
                    for r_idx, r in enumerate(routes):
                        for pos in range(1, len(r)):
                            prev = r[pos-1]
                            nxt = r[pos]
                            added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                            new_dist = route_dist(r) + added
                            other_max = max(route_dist(routes[i]) for i in range(truck_count) if i != r_idx)
                            new_max = max(new_dist, other_max)
                            increase = new_max - current_max
                            if increase < best_increase - 1e-12 or (abs(increase - best_increase) < 1e-12 and (r_idx < best_route or (r_idx == best_route and pos < best_pos))):
                                best_increase = increase
                                best_route = r_idx
                                best_pos = pos
                    routes[best_route].insert(best_pos, cust)
                return True

            shake_route(longest_idx, 0.2)
            cur_max = max_dist(routes)
            if cur_max >= best_max - 1e-12 and second_idx is not None:
                shake_route(second_idx, 0.15)
                cur_max = max_dist(routes)
            if cur_max < best_max - 1e-12:
                best_routes = [r[:] for r in routes]
                best_max = cur_max
                report_best_vrp(best_routes)

    if best_routes is None:
        best_routes = [r[:] for r in routes]
    return best_routes