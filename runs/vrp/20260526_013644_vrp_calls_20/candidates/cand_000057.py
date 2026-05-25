import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    customers.sort(key=lambda c: distance_matrix[0, c], reverse=True)

    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count

    best_max = float('inf')
    best_routes = [r[:] for r in routes]

    def update_best():
        nonlocal best_max, best_routes
        max_len = max(route_lengths)
        if max_len < best_max - 1e-12:
            best_max = max_len
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

    def route_len(route):
        length = 0.0
        for i in range(len(route)-1):
            length += distance_matrix[route[i], route[i+1]]
        return length

    # Construction: farthest insertion with min-max
    for cust in customers:
        best_route_idx = -1
        best_pos = -1
        best_new_max = float('inf')
        for r_idx in range(truck_count):
            route = routes[r_idx]
            best_increase = float('inf')
            best_pos_in_route = -1
            for i in range(1, len(route)):
                inc = distance_matrix[route[i-1], cust] + distance_matrix[cust, route[i]] - distance_matrix[route[i-1], route[i]]
                if inc < best_increase - 1e-12:
                    best_increase = inc
                    best_pos_in_route = i
            new_len = route_lengths[r_idx] + best_increase
            other_max = 0.0
            for t in range(truck_count):
                if t == r_idx:
                    continue
                if route_lengths[t] > other_max:
                    other_max = route_lengths[t]
            new_max = max(other_max, new_len)
            if new_max < best_new_max - 1e-12 or (abs(new_max - best_new_max) < 1e-12 and r_idx < best_route_idx):
                best_new_max = new_max
                best_route_idx = r_idx
                best_pos = best_pos_in_route
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
        route_lengths[best_route_idx] = route_len(route)

    update_best()

    # 2-opt improvement
    for r_idx in range(truck_count):
        route = routes[r_idx]
        improved = True
        max_iter = max(1, len(route) - 3)
        for _ in range(max_iter):
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i+1, len(route) - 1):
                    if j - i == 1:
                        continue
                    old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    if new < old - 1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
            if not improved:
                break
        route_lengths[r_idx] = route_len(route)

    update_best()

    # Inter-route relocation from longest route
    max_iter_inter = n
    for _ in range(max_iter_inter):
        max_len = max(route_lengths)
        if max_len <= 0:
            break
        longest_idx = route_lengths.index(max_len)
        longest_route = routes[longest_idx]
        best_improvement = None
        best_new_max = max_len
        for cust_idx in range(1, len(longest_route)-1):
            cust = longest_route[cust_idx]
            for other_idx in range(truck_count):
                if other_idx == longest_idx:
                    continue
                other_route = routes[other_idx]
                best_increase = float('inf')
                best_pos = -1
                for i in range(1, len(other_route)):
                    inc = distance_matrix[other_route[i-1], cust] + distance_matrix[cust, other_route[i]] - distance_matrix[other_route[i-1], other_route[i]]
                    if inc < best_increase - 1e-12:
                        best_increase = inc
                        best_pos = i
                new_other_len = route_lengths[other_idx] + best_increase
                new_longest_len = route_len(longest_route[:cust_idx] + longest_route[cust_idx+1:])
                candidate_max = max(new_longest_len, new_other_len)
                for t in range(truck_count):
                    if t != longest_idx and t != other_idx:
                        candidate_max = max(candidate_max, route_lengths[t])
                if candidate_max < best_new_max - 1e-12:
                    best_new_max = candidate_max
                    best_improvement = (cust, longest_idx, cust_idx, other_idx, best_pos, new_longest_len, new_other_len)
        if best_improvement is not None:
            cust, from_idx, cust_idx, to_idx, pos, new_from_len, new_to_len = best_improvement
            routes[from_idx].pop(cust_idx)
            routes[to_idx].insert(pos, cust)
            route_lengths[from_idx] = new_from_len
            route_lengths[to_idx] = new_to_len
            update_best()
        else:
            break

    update_best()
    return best_routes