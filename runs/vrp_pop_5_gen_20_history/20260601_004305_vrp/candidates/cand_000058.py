import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # --- Initial solution: sequential insertion heuristic ---
    routes = [[0, 0] for _ in range(truck_count)]
    cust_list = list(range(1, n))
    for cust in cust_list:
        best_ri = -1
        best_pos = -1
        best_inc = math.inf
        for ri, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_dist = sum(distance_matrix[new_route[k]][new_route[k+1]] for k in range(len(new_route)-1))
                old_dist = sum(distance_matrix[route[k]][route[k+1]] for k in range(len(route)-1))
                increase = new_dist - old_dist
                if increase < best_inc or (increase == best_inc and (ri < best_ri or (ri == best_ri and pos < best_pos))):
                    best_inc = increase
                    best_ri = ri
                    best_pos = pos
        routes[best_ri].insert(best_pos, cust)

    # Helper functions
    def route_dist(route):
        return sum(distance_matrix[route[k]][route[k+1]] for k in range(len(route)-1))

    def compute_max(routes):
        return max(route_dist(r) for r in routes)

    def copy_routes(routes):
        return [list(r) for r in routes]

    current = copy_routes(routes)
    current_max = compute_max(current)
    best = copy_routes(current)
    best_max = current_max
    report_best_vrp(best)

    # --- ALNS parameters ---
    max_iter = 2000
    initial_temp = 0.1 * current_max
    final_temp = 0.001
    cooling_rate = (final_temp / initial_temp) ** (1.0 / max_iter)
    temp = initial_temp

    # Operator weights for adaptive selection
    destroy_ops = ["random", "worst"]
    repair_ops = ["greedy", "regret2"]
    weights_d = [1.0, 1.0]
    weights_r = [1.0, 1.0]
    scores = [0.0, 0.0, 0.0, 0.0]  # random-greedy, random-regret2, worst-greedy, worst-regret2
    usage = [0, 0, 0, 0]
    n_cust = m

    for it in range(max_iter):
        # Select destroy and repair
        d_idx = random.choices(range(2), weights=weights_d)[0]
        r_idx = random.choices(range(2), weights=weights_r)[0]
        op_idx = d_idx * 2 + r_idx

        # Determine destroy size
        q = max(1, n_cust // 10)  # remove ~10% of customers

        # Destroy
        removed = []
        new_routes = copy_routes(current)
        if destroy_ops[d_idx] == "random":
            # random removal
            all_cust = [c for route in new_routes for c in route if c != 0]
            random.shuffle(all_cust)
            for c in all_cust[:q]:
                for route in new_routes:
                    if c in route:
                        route.remove(c)
                        removed.append(c)
                        break
        else:  # worst removal based on detour cost
            detour = {}
            for route in new_routes:
                for p in range(1, len(route)-1):
                    c = route[p]
                    prev = route[p-1]
                    nxt = route[p+1]
                    det = distance_matrix[prev][c] + distance_matrix[c][nxt] - distance_matrix[prev][nxt]
                    detour[c] = det
            sorted_cust = sorted(detour.items(), key=lambda x: -x[1])
            for c, _ in sorted_cust[:q]:
                for route in new_routes:
                    if c in route:
                        route.remove(c)
                        removed.append(c)
                        break

        # Repair
        random.shuffle(removed)
        if repair_ops[r_idx] == "greedy":
            for c in removed:
                best_inc = math.inf
                best_ri = -1
                best_pos = -1
                for ri, route in enumerate(new_routes):
                    old_dist = route_dist(route)
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [c] + route[pos:]
                        new_dist = route_dist(new_route)
                        other_dists = [route_dist(r) for ri2, r in enumerate(new_routes) if ri2 != ri]
                        new_max = max(new_dist, *other_dists)
                        if new_max < best_inc or (new_max == best_inc and (ri < best_ri or (ri == best_ri and pos < best_pos))):
                            best_inc = new_max
                            best_ri = ri
                            best_pos = pos
                new_routes[best_ri].insert(best_pos, c)
        else:  # regret-2 insertion
            for _ in range(len(removed)):
                best_c = -1
                best_regret = -1
                best_ri = -1
                best_pos = -1
                best_max_val = math.inf
                for c in removed:
                    first = (math.inf, -1, -1)
                    second = (math.inf, -1, -1)
                    for ri, route in enumerate(new_routes):
                        for pos in range(1, len(route)):
                            new_route = route[:pos] + [c] + route[pos:]
                            new_dist = route_dist(new_route)
                            other_dists = [route_dist(r) for ri2, r in enumerate(new_routes) if ri2 != ri]
                            new_max = max(new_dist, *other_dists)
                            if new_max < first[0]:
                                second = first
                                first = (new_max, ri, pos)
                            elif new_max < second[0]:
                                second = (new_max, ri, pos)
                    if first[0] == math.inf:
                        continue
                    regret = second[0] - first[0]
                    if regret > best_regret or (regret == best_regret and c < best_c):
                        best_regret = regret
                        best_c = c
                        best_ri = first[1]
                        best_pos = first[2]
                        best_max_val = first[0]
                if best_c != -1:
                    new_routes[best_ri].insert(best_pos, best_c)
                    removed.remove(best_c)

        # Evaluate new solution
        new_max = compute_max(new_routes)
        delta = new_max - current_max

        # Accept using simulated annealing
        accepted = False
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = new_routes
            current_max = new_max
            accepted = True
            if new_max < best_max:
                best = copy_routes(new_routes)
                best_max = new_max
                report_best_vrp(best)

        # Update scores and weights
        if accepted:
            if new_max < best_max:
                scores[op_idx] += 1.0
            else:
                scores[op_idx] += 0.5
        usage[op_idx] += 1

        # Update weights every 100 iterations
        if (it + 1) % 100 == 0:
            for ii in range(4):
                if usage[ii] > 0:
                    scores[ii] /= usage[ii]
            for d_idx2 in range(2):
                avg = (scores[d_idx2*2] + scores[d_idx2*2+1]) / 2.0 if (usage[d_idx2*2] + usage[d_idx2*2+1]) > 0 else 0
                weights_d[d_idx2] = max(0.1, weights_d[d_idx2] * 0.9 + avg * 0.1)
            for r_idx2 in range(2):
                avg = (scores[r_idx2] + scores[2+r_idx2]) / 2.0 if (usage[r_idx2] + usage[2+r_idx2]) > 0 else 0
                weights_r[r_idx2] = max(0.1, weights_r[r_idx2] * 0.9 + avg * 0.1)
            for ii in range(4):
                scores[ii] = 0.0
                usage[ii] = 0

        # Cool temperature
        temp *= cooling_rate

    return best