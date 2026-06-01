import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = len(distance_matrix)
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)] + [[0, 0]] * (truck_count - (n - 1))
        return routes

    # ---------- New initialization: greedy insertion ----------
    routes = [[0, 0] for _ in range(truck_count)]
    for cust in range(1, n):
        best_route = 0
        best_pos = 1
        best_increase = float('inf')
        for r in range(truck_count):
            route = routes[r]
            # Consider all possible insertion positions (between depot and depot)
            for pos in range(1, len(route)):
                a = route[pos-1]
                b = route[pos]
                inc = distance_matrix[a][cust] + distance_matrix[cust][b] - distance_matrix[a][b]
                if inc < best_increase or (inc == best_increase and r < best_route):
                    best_increase = inc
                    best_route = r
                    best_pos = pos
        routes[best_route].insert(best_pos, cust)

    # ---------- Same 2-opt improvement per route as parent ----------
    def route_dist(route):
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

    for idx in range(len(routes)):
        route = routes[idx]
        for _ in range(1000):
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    a, b, c, d = route[i-1], route[i], route[j], route[j+1]
                    old = distance_matrix[a][b] + distance_matrix[c][d]
                    new = distance_matrix[a][c] + distance_matrix[b][d]
                    if new < old:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        routes[idx] = route

    # ---------- Same relocate-from-longest improvement as parent ----------
    max_dist = max(route_dist(r) for r in routes)
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)

    for _ in range(n * 2):
        longest_idx = max(range(len(routes)), key=lambda i: (route_dist(routes[i]), i))
        longest = routes[longest_idx]
        candidates = sorted([c for c in longest if c != 0])
        best_new_max = float('inf')
        best_move = None
        for cust in candidates:
            new_longest = [x for x in longest if x != cust]
            if len(new_longest) < 2:
                continue
            new_longest_dist = route_dist(new_longest)
            for other_idx in range(len(routes)):
                if other_idx == longest_idx:
                    continue
                other = routes[other_idx]
                for pos in range(1, len(other)):
                    new_other = other[:pos] + [cust] + other[pos:]
                    new_other_dist = route_dist(new_other)
                    other_dists = [route_dist(routes[i]) for i in range(len(routes)) if i not in (longest_idx, other_idx)]
                    cand_max = max(new_longest_dist, new_other_dist, *other_dists)
                    if cand_max < best_new_max:
                        best_new_max = cand_max
                        best_move = (cust, other_idx, pos, new_longest, new_other)
        if best_move is not None and best_new_max < max_dist:
            cust, other_idx, pos, new_longest, new_other = best_move
            routes[longest_idx] = new_longest
            routes[other_idx] = new_other
            max_dist = best_new_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        else:
            break

    return best_routes