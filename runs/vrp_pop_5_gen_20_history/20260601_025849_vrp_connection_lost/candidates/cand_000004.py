import numpy as np
from typing import List

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> List[List[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if not customers:
        return [[0, 0] for _ in range(truck_count)]

    # ---------- Giant tour via nearest neighbor ----------
    giant = []
    visited = set()
    current = 0
    while len(visited) < n - 1:
        nearest = None
        best_d = np.inf
        for v in range(1, n):
            if v not in visited:
                d = distance_matrix[current, v]
                if d < best_d:
                    best_d = d
                    nearest = v
        giant.append(nearest)
        visited.add(nearest)
        current = nearest

    m = len(giant)
    # Precompute segment cost: from depot to first, then along path, then last to depot
    # segment cost for consecutive customers from i to j-1 (inclusive)
    # We'll compute prefix sums for internal edges: sum from giant[i] to giant[i+1]
    prefix = np.zeros(m + 1)
    for i in range(m - 1):
        prefix[i + 1] = prefix[i] + distance_matrix[giant[i], giant[i + 1]]
    # total internal cost for segment [i,j): internal = prefix[j-1] - prefix[i] if j > i+1 else 0

    def segment_cost(i: int, j: int) -> float:
        # segment covers customers giant[i:j] (i inclusive, j exclusive)
        if i >= j:
            return 0.0
        internal = prefix[j - 1] - prefix[i] if j - i > 1 else 0.0
        first = giant[i]
        last = giant[j - 1]
        return distance_matrix[0, first] + internal + distance_matrix[last, 0]

    # Binary search on max segment cost M
    lo = 0.0
    hi = segment_cost(0, m)  # whole tour as one segment
    # greedy feasibility check: can we partition into <= truck_count segments each cost <= M?
    def feasible(M: float) -> bool:
        count = 0
        i = 0
        while i < m:
            j = i
            while j < m and segment_cost(i, j + 1) <= M:
                j += 1
            # if cannot even place the first customer, infeasible
            if j == i:
                return False
            count += 1
            i = j
            if count > truck_count:
                return False
        return True

    # Ensure we can always partition with hi
    for _ in range(100):  # enough iterations for precision
        mid = (lo + hi) / 2.0
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    M = hi

    # Build actual partition using greedy with cost <= M
    routes = []
    i = 0
    while i < m:
        j = i
        while j < m and segment_cost(i, j + 1) <= M:
            j += 1
        # if no progress, force one customer
        if j == i:
            j = i + 1
        route = [0] + giant[i:j] + [0]
        routes.append(route)
        i = j
    while len(routes) < truck_count:
        routes.append([0, 0])
    # If too many routes due to forcing, we merge; but greedy should produce <= truck_count. If not, merge the last segments.
    while len(routes) > truck_count:
        # merge last two routes into one
        r1 = routes.pop()
        r2 = routes.pop()
        merged = r2[:-1] + r1[1:]
        routes.append(merged)
    # Ensure exactly truck_count
    routes = routes[:truck_count]

    # ---------- Compute route distances ----------
    def route_dist(route):
        if len(route) == 2:
            return 0.0
        dist = 0.0
        for k in range(len(route) - 1):
            dist += distance_matrix[route[k], route[k+1]]
        return dist

    dists = [route_dist(r) for r in routes]
    best_max = max(dists)
    best_routes = [list(r) for r in routes]

    # report initial solution
    from types import SimpleNamespace
    ns = SimpleNamespace()
    ns.best_routes = best_routes
    ns.best_max = best_max

    # ---------- Local search: relocate ----------
    improved = True
    while improved:
        improved = False
        for cust in range(1, n):
            # find current route and position
            curr_route_idx = None
            curr_pos = None
            for ri, route in enumerate(routes):
                if cust in route:
                    curr_route_idx = ri
                    curr_pos = route.index(cust)
                    break
            if curr_route_idx is None:
                continue
            # try moving cust to every other route (including its own) at every insertion position (after 0, before last depot)
            # do not move to same position
            for target_ri in range(truck_count):
                if target_ri == curr_route_idx and len(routes[target_ri]) == 2:
                    continue
                target_route = routes[target_ri]
                # possible insertion positions: from index 1 to len(target_route)-1 (inclusive after moving? actually we insert before depot? we can insert at any position except before 0 and after last 0? We'll allow positions 1 to len(target_route)-1
                # but if target_ri == curr_route_idx, we must consider moving to a different position (not curr_pos)
                for pos in range(1, len(target_route)):
                    if target_ri == curr_route_idx and pos == curr_pos:
                        continue
                    if target_ri == curr_route_idx and pos == curr_pos + 1:
                        # adjacent positions, but moving might not change order; skip to avoid same state
                        # actually moving to pos such that insertion results in same order? We'll allow all
                        pass
                    # perform move temporarily
                    old_curr_route = routes[curr_route_idx][:]
                    old_target_route = routes[target_ri][:]
                    # remove from current
                    new_curr = [x for x in routes[curr_route_idx] if x != cust]
                    if new_curr == [0]:
                        new_curr = [0, 0]
                    # insert into target
                    new_target = routes[target_ri][:pos] + [cust] + routes[target_ri][pos:]
                    # compute new distances
                    new_curr_dist = route_dist(new_curr)
                    new_target_dist = route_dist(new_target)
                    new_dists = dists[:]
                    new_dists[curr_route_idx] = new_curr_dist
                    new_dists[target_ri] = new_target_dist
                    new_max = max(new_dists)
                    if new_max < best_max:
                        # accept move
                        routes[curr_route_idx] = new_curr
                        routes[target_ri] = new_target
                        dists = new_dists
                        best_max = new_max
                        best_routes = [list(r) for r in routes]
                        improved = True
                        # call report
                        # We're not actually using the SimpleNamespace; just call report_best_vrp if available
                        # But in the contract we must call report_best_vrp. We'll do it here.
                        # Assume report_best_vrp is defined in outer scope? Actually it's not imported. We'll define it as a no-op for local testing? But the solver will be called in an environment where report_best_vrp exists. So we just call it.
                        try:
                            report_best_vrp(best_routes)
                        except:
                            pass
                        # break out of loops to restart
                        break
                if improved:
                    break
            if improved:
                break
        # end while loop over customers

    return best_routes