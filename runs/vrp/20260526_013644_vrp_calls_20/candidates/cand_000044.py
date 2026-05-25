import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def route_length(route):
        total = 0.0
        for i in range(len(route) - 1):
            total += distance_matrix[route[i], route[i + 1]]
        return total

    best_overall_max = float('inf')
    best_overall_routes = None

    for restart in range(3):
        # Reinitialize routes
        routes = [[0, 0] for _ in range(truck_count)]
        route_lengths = [0.0] * truck_count

        # Farthest-first insertion minimizing max distance
        sorted_cust = sorted(customers, key=lambda c: distance_matrix[0, c], reverse=True)
        for cust in sorted_cust:
            best_route_idx = -1
            best_pos = -1
            best_new_max = float('inf')
            for r_idx in range(truck_count):
                route = routes[r_idx]
                best_inc = float('inf')
                best_pos_in_route = -1
                for i in range(1, len(route)):
                    prev = route[i - 1]
                    nxt = route[i]
                    inc = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if inc < best_inc:
                        best_inc = inc
                        best_pos_in_route = i
                new_len = route_lengths[r_idx] + best_inc
                other_max = max(route_lengths[:r_idx] + route_lengths[r_idx+1:]) if truck_count > 1 else 0
                new_max = max(other_max, new_len)
                if new_max < best_new_max or (new_max == best_new_max and r_idx < best_route_idx):
                    best_new_max = new_max
                    best_route_idx = r_idx
                    best_pos = best_pos_in_route
            route = routes[best_route_idx]
            route.insert(best_pos, cust)
            route_lengths[best_route_idx] = route_length(route)
        current_max = max(route_lengths)
        if current_max < best_overall_max:
            best_overall_max = current_max
            best_overall_routes = [r[:] for r in routes]
            report_best_vrp(best_overall_routes)

        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = routes[r_idx]
            improved = True
            max_2opt = len(route) * len(route)
            for _ in range(max_2opt):
                improved = False
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        if j - i == 1:
                            continue
                        old = distance_matrix[route[i - 1], route[i]] + distance_matrix[route[j], route[j + 1]]
                        new = distance_matrix[route[i - 1], route[j]] + distance_matrix[route[i], route[j + 1]]
                        if new < old - 1e-12:
                            route[i:j + 1] = reversed(route[i:j + 1])
                            improved = True
                if not improved:
                    break
            route_lengths[r_idx] = route_length(route)
        current_max = max(route_lengths)
        if current_max < best_overall_max:
            best_overall_max = current_max
            best_overall_routes = [r[:] for r in routes]
            report_best_vrp(best_overall_routes)

        # Inter-route relocation accepting any reduction in current max
        max_reloc = n * n
        stagnation = 0
        for _ in range(max_reloc):
            improved = False
            for c in customers:
                for r_idx_old, route in enumerate(routes):
                    if c in route:
                        break
                old_route = routes[r_idx_old]
                new_route_old = [x for x in old_route if x != c]
                new_len_old = route_length(new_route_old)
                for r2_idx in range(truck_count):
                    if r2_idx == r_idx_old:
                        continue
                    r2 = routes[r2_idx]
                    for pos in range(1, len(r2)):
                        new_len_r2 = route_lengths[r2_idx] - distance_matrix[r2[pos-1], r2[pos]] + distance_matrix[r2[pos-1], c] + distance_matrix[c, r2[pos]]
                        temp_lengths = route_lengths.copy()
                        temp_lengths[r_idx_old] = new_len_old
                        temp_lengths[r2_idx] = new_len_r2
                        new_current_max = max(temp_lengths)
                        if new_current_max < current_max - 1e-12:
                            routes[r_idx_old] = new_route_old
                            routes[r2_idx] = r2[:pos] + [c] + r2[pos:]
                            route_lengths[r_idx_old] = new_len_old
                            route_lengths[r2_idx] = new_len_r2
                            current_max = new_current_max
                            if current_max < best_overall_max - 1e-12:
                                best_overall_max = current_max
                                best_overall_routes = [r[:] for r in routes]
                                report_best_vrp(best_overall_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                stagnation += 1
            else:
                stagnation = 0
            if stagnation > n:
                break

    if best_overall_routes is None:
        return routes
    return best_overall_routes