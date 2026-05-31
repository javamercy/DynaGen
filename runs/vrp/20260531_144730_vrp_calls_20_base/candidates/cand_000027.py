import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    assert n > 0
    if truck_count <= 0:
        return []
    # initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = list(range(1, n))

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def best_insertion(c, routes, route_dists):
        """Return (best_new_max, best_route_idx, best_pos, second_best_new_max)."""
        best = (float('inf'), -1, -1)
        second = (float('inf'), -1, -1)
        for r_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            other_max = 0.0
            for j, d in enumerate(route_dists):
                if j != r_idx and d > other_max:
                    other_max = d
            for pos in range(1, len(route)):
                pred = route[pos-1]
                succ = route[pos]
                new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                new_max = max(other_max, new_dist)
                if new_max < best[0]:
                    best, second = (new_max, r_idx, pos), best
                elif new_max < second[0]:
                    second = (new_max, r_idx, pos)
        return best[0], best[1], best[2], second[0]

    # regret construction
    while unassigned:
        bests = []
        for c in unassigned:
            best_new_max, best_route, best_pos, second_new_max = best_insertion(c, routes, route_dists)
            if best_route == -1:
                continue
            regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
            bests.append((-regret, c, best_route, best_pos, best_new_max))
        bests.sort(key=lambda x: (x[0], x[1]))
        _, c, best_route, best_pos, new_max = bests[0]
        route = routes[best_route]
        route.insert(best_pos, c)
        route_dists[best_route] = route_dist(route)
        unassigned.remove(c)
        report_best_vrp(routes)

    # intra-route 2-opt improvement
    for r_idx in range(truck_count):
        route = routes[r_idx]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old_edges = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new_edges = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    delta = new_edges - old_edges
                    if delta < -1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        route_dists[r_idx] = route_dist(route)
                        break
                if improved:
                    break
    report_best_vrp(routes)

    # inter-route tail swap improvement
    max_iter = n * truck_count
    for _ in range(max_iter):
        moved = False
        max_dist = max(route_dists)
        for i in range(truck_count):
            for j in range(i+1, truck_count):
                r1 = routes[i]
                r2 = routes[j]
                d1 = route_dists[i]
                d2 = route_dists[j]
                # try all cut positions (excluding depot at start and end? cut after a customer, so position index from 1 to len(route)-2? We'll allow cut after any index except the last? Actually we need to keep depot at end, so we cut before the last 0? Better: cut after some customer index, and then the tail includes the remaining customers and the depot. We'll iterate cut1 from 1 to len(r1)-2 (inclusive) and cut2 from 1 to len(r2)-2.
                for cut1 in range(1, len(r1)-1):  # cut after r1[cut1], tail is r1[cut1+1:]
                    for cut2 in range(1, len(r2)-1):
                        # new routes: r1[:cut1+1] + r2[cut2+1:], r2[:cut2+1] + r1[cut1+1:]
                        new_r1 = r1[:cut1+1] + r2[cut2+1:]
                        new_r2 = r2[:cut2+1] + r1[cut1+1:]
                        new_d1 = route_dist(new_r1)
                        new_d2 = route_dist(new_r2)
                        other_max = 0.0
                        for k in range(truck_count):
                            if k != i and k != j:
                                other_max = max(other_max, route_dists[k])
                        new_overall_max = max(other_max, new_d1, new_d2)
                        if new_overall_max < max_dist - 1e-12:
                            routes[i] = new_r1
                            routes[j] = new_r2
                            route_dists[i] = new_d1
                            route_dists[j] = new_d2
                            moved = True
                            report_best_vrp(routes)
                            break
                    if moved:
                        break
                if moved:
                    break
            if moved:
                break
        if not moved:
            break
    return routes