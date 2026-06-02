import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_distance(route):
        total = 0.0
        for a in range(len(route)-1):
            total += distance_matrix[route[a]][route[a+1]]
        return total

    def compute_max(routes):
        return max(route_distance(r) for r in routes)

    def copy_routes(routes):
        return [list(r) for r in routes]

    def local_search(routes):
        current_routes = copy_routes(routes)
        current_max = compute_max(current_routes)
        improved = True
        max_iter = n * 2
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = current_routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        if j - i == 1:
                            continue
                        old = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                        new = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                        if new < old - 1e-10:
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            new_dist = route_distance(new_route)
                            other_max = max(route_distance(current_routes[k]) for k in range(truck_count) if k != r_idx)
                            new_max = max(new_dist, other_max)
                            if new_max < current_max - 1e-10:
                                current_routes[r_idx] = new_route
                                current_max = new_max
                                improved = True
                                try:
                                    report_best_vrp(current_routes)
                                except:
                                    pass
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Inter-route relocate from longest route
            max_dist = 0
            max_idx = 0
            for i, r in enumerate(current_routes):
                d = route_distance(r)
                if d > max_dist:
                    max_dist = d
                    max_idx = i
            route_from = current_routes[max_idx]
            for cust in route_from[1:-1]:
                for r_to_idx in range(truck_count):
                    if r_to_idx == max_idx:
                        continue
                    route_to = current_routes[r_to_idx]
                    for pos in range(1, len(route_to)):
                        new_route_from = [0] + [x for x in route_from[1:-1] if x != cust] + [0]
                        if len(new_route_from) == 1:
                            new_route_from = [0,0]
                        new_route_to = route_to[:pos] + [cust] + route_to[pos:]
                        new_dist_from = route_distance(new_route_from)
                        new_dist_to = route_distance(new_route_to)
                        other_max = 0.0
                        for k in range(truck_count):
                            if k != max_idx and k != r_to_idx:
                                dk = route_distance(current_routes[k])
                                if dk > other_max:
                                    other_max = dk
                        new_max = max(new_dist_from, new_dist_to, other_max)
                        if new_max < current_max - 1e-10:
                            current_routes[max_idx] = new_route_from
                            current_routes[r_to_idx] = new_route_to
                            current_max = new_max
                            improved = True
                            try:
                                report_best_vrp(current_routes)
                            except:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Inter-route exchange
            for i in range(truck_count):
                for j in range(i+1, truck_count):
                    route_i = current_routes[i]
                    route_j = current_routes[j]
                    for pos_i in range(1, len(route_i)-1):
                        for pos_j in range(1, len(route_j)-1):
                            cust_i = route_i[pos_i]
                            cust_j = route_j[pos_j]
                            old_i = distance_matrix[route_i[pos_i-1]][cust_i] + distance_matrix[cust_i][route_i[pos_i+1]]
                            old_j = distance_matrix[route_j[pos_j-1]][cust_j] + distance_matrix[cust_j][route_j[pos_j+1]]
                            new_i = distance_matrix[route_i[pos_i-1]][cust_j] + distance_matrix[cust_j][route_i[pos_i+1]]
                            new_j = distance_matrix[route_j[pos_j-1]][cust_i] + distance_matrix[cust_i][route_j[pos_j+1]]
                            gain = (old_i + old_j) - (new_i + new_j)
                            if gain > 1e-10:
                                len_i = route_distance(route_i)
                                len_j = route_distance(route_j)
                                new_len_i = len_i - old_i + new_i
                                new_len_j = len_j - old_j + new_j
                                other_max = 0.0
                                for k in range(truck_count):
                                    if k != i and k != j:
                                        dk = route_distance(current_routes[k])
                                        if dk > other_max:
                                            other_max = dk
                                new_max = max(new_len_i, new_len_j, other_max)
                                if new_max < current_max - 1e-10:
                                    new_route_i = list(route_i)
                                    new_route_j = list(route_j)
                                    new_route_i[pos_i], new_route_j[pos_j] = cust_j, cust_i
                                    current_routes[i] = new_route_i
                                    current_routes[j] = new_route_j
                                    current_max = new_max
                                    improved = True
                                    try:
                                        report_best_vrp(current_routes)
                                    except:
                                        pass
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Cross-route 2-opt*
            for i in range(truck_count):
                for j in range(i+1, truck_count):
                    route_i = current_routes[i]
                    route_j = current_routes[j]
                    if len(route_i) <= 2 or len(route_j) <= 2:
                        continue
                    for p in range(1, len(route_i)-1):
                        for q in range(1, len(route_j)-1):
                            new_route_i = route_i[:p+1] + route_j[q+1:]
                            new_route_j = route_j[:q+1] + route_i[p+1:]
                            if new_route_i[-1] != 0:
                                new_route_i.append(0)
                            if new_route_j[-1] != 0:
                                new_route_j.append(0)
                            if new_route_i[-1] != 0 or new_route_j[-1] != 0:
                                continue
                            len_i = route_distance(route_i)
                            len_j = route_distance(route_j)
                            new_len_i = route_distance(new_route_i)
                            new_len_j = route_distance(new_route_j)
                            other_max = 0.0
                            for k in range(truck_count):
                                if k != i and k != j:
                                    dk = route_distance(current_routes[k])
                                    if dk > other_max:
                                        other_max = dk
                            new_max = max(new_len_i, new_len_j, other_max)
                            if new_max < current_max - 1e-10:
                                current_routes[i] = new_route_i
                                current_routes[j] = new_route_j
                                current_max = new_max
                                improved = True
                                try:
                                    report_best_vrp(current_routes)
                                except:
                                    pass
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return current_routes, current_max

    def regret_construction():
        # Initialize routes
        routes = [[0,0] for _ in range(truck_count)]
        assigned = [False] * (n)
        assigned[0] = True
        unassigned = set(customers)
        # Insert first customers to each route to avoid empty routes
        for k in range(min(truck_count, len(unassigned))):
            # pick customer farthest from depot?
            best_cust = None
            best_dist = -1.0
            for c in unassigned:
                d = distance_matrix[0][c]
                if d > best_dist:
                    best_dist = d
                    best_cust = c
            # insert into route k
            routes[k] = [0, best_cust, 0]
            assigned[best_cust] = True
            unassigned.remove(best_cust)
        # Regret insertion
        while unassigned:
            best_cust = None
            best_regret = -1.0
            best_cost = float('inf')
            best_route_idx = -1
            best_pos = -1
            for c in sorted(unassigned):
                costs = []
                positions = []
                for r_idx, route in enumerate(routes):
                    if len(route) <= 2:
                        # only one position
                        pos = 1
                        new_len = distance_matrix[0][c] + distance_matrix[c][0]
                        max_other = compute_max([routes[k] for k in range(truck_count) if k != r_idx])
                        cost = max(new_len, max_other)
                        costs.append(cost)
                        positions.append((r_idx, pos))
                    else:
                        for pos in range(1, len(route)):
                            # compute new route distance if c inserted at pos
                            prev = route[pos-1]
                            nxt = route[pos]
                            old = distance_matrix[prev][nxt]
                            new = distance_matrix[prev][c] + distance_matrix[c][nxt]
                            new_len = route_distance(route) - old + new
                            max_other = compute_max([routes[k] for k in range(truck_count) if k != r_idx])
                            cost = max(new_len, max_other)
                            costs.append((cost, r_idx, pos))
                # sort costs
                costs.sort()
                # regret = second best - best
                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = 0
                if regret > best_regret or (regret == best_regret and costs[0][0] < best_cost):
                    best_regret = regret
                    best_cost = costs[0][0]
                    best_cust = c
                    best_route_idx = costs[0][1]
                    best_pos = costs[0][2]
            # insert best_cust
            route = routes[best_route_idx]
            if len(route) <= 2:
                routes[best_route_idx] = [0, best_cust, 0]
            else:
                routes[best_route_idx] = route[:best_pos] + [best_cust] + route[best_pos:]
            assigned[best_cust] = True
            unassigned.remove(best_cust)
        return routes

    def perturbation(routes):
        # Remove two customers with highest removal saving and reinsert via regret
        current_routes = copy_routes(routes)
        savings = []
        for r_idx, route in enumerate(current_routes):
            if len(route) <= 2:
                continue
            for pos in range(1, len(route)-1):
                cust = route[pos]
                prev = route[pos-1]
                nxt = route[pos+1]
                old = distance_matrix[prev][cust] + distance_matrix[cust][nxt]
                new = distance_matrix[prev][nxt]
                saving = old - new
                savings.append((saving, r_idx, pos, cust))
        if not savings:
            return current_routes
        savings.sort(reverse=True)
        num_remove = min(2, len(savings))
        removed_customers = []
        for _, r_idx, pos, cust in savings[:num_remove]:
            route = current_routes[r_idx]
            del route[pos]
            if len(route) == 1:
                route = [0,0]
            current_routes[r_idx] = route
            removed_customers.append(cust)
        # Reinsert removed customers via regret
        unassigned = set(removed_customers)
        while unassigned:
            best_cust = None
            best_regret = -1.0
            best_cost = float('inf')
            best_route_idx = -1
            best_pos = -1
            for c in sorted(unassigned):
                costs = []
                for r_idx, route in enumerate(current_routes):
                    if len(route) <= 2:
                        pos = 1
                        new_len = distance_matrix[0][c] + distance_matrix[c][0]
                        max_other = compute_max([current_routes[k] for k in range(truck_count) if k != r_idx])
                        cost = max(new_len, max_other)
                        costs.append((cost, r_idx, pos))
                    else:
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            old = distance_matrix[prev][nxt]
                            new = distance_matrix[prev][c] + distance_matrix[c][nxt]
                            new_len = route_distance(route) - old + new
                            max_other = compute_max([current_routes[k] for k in range(truck_count) if k != r_idx])
                            cost = max(new_len, max_other)
                            costs.append((cost, r_idx, pos))
                costs.sort()
                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = 0
                if regret > best_regret or (regret == best_regret and costs[0][0] < best_cost):
                    best_regret = regret
                    best_cost = costs[0][0]
                    best_cust = c
                    best_route_idx = costs[0][1]
                    best_pos = costs[0][2]
            route = current_routes[best_route_idx]
            if len(route) <= 2:
                current_routes[best_route_idx] = [0, best_cust, 0]
            else:
                current_routes[best_route_idx] = route[:best_pos] + [best_cust] + route[best_pos:]
            unassigned.remove(best_cust)
        return current_routes

    best_routes = None
    best_max = float('inf')

    # Multi-start with regret construction
    for start in range(min(3, n-1)):  # try a few different orders
        # For diversity, vary order by starting with different customer? Actually regret is deterministic.
        # We'll just run once and then perturb. So just one construction.
        pass
    routes = regret_construction()
    routes, cur_max = local_search(routes)
    if cur_max < best_max:
        best_max = cur_max
        best_routes = copy_routes(routes)
        try:
            report_best_vrp(best_routes)
        except:
            pass

    # Additional restarts with perturbation
    for restart in range(2):  # limited restarts
        routes = perturbation(best_routes if best_routes else routes)
        routes, cur_max = local_search(routes)
        if cur_max < best_max:
            best_max = cur_max
            best_routes = copy_routes(routes)
            try:
                report_best_vrp(best_routes)
            except:
                pass

    if best_routes is None:
        # fallback: trivial solution
        best_routes = [[0,0] for _ in range(truck_count)]
    # Ensure all customers covered
    used = set()
    for r in best_routes:
        for c in r:
            if c != 0:
                used.add(c)
    missing = [c for c in customers if c not in used]
    if missing:
        # assign missing to first non-empty route
        for r in best_routes:
            if len(r) > 2 or (len(r)==2 and r[0]==0 and r[1]==0):
                continue
            # insert missing
            for c in missing:
                best_pos = 1
                best_inc = float('inf')
                for pos in range(1, len(r)):
                    prev = r[pos-1]
                    nxt = r[pos]
                    old = distance_matrix[prev][nxt]
                    new = distance_matrix[prev][c] + distance_matrix[c][nxt]
                    inc = new - old
                    if inc < best_inc:
                        best_inc = inc
                        best_pos = pos
                r = r[:best_pos] + [c] + r[best_pos:]
                best_routes[0] = r  # update
            break
    # Ensure empty trucks
    while len(best_routes) < truck_count:
        best_routes.append([0,0])
    return best_routes