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
        report_best_vrp(routes)
        return routes

    # --- Initial solution using Clarke-Wright savings ---
    # Initialize each customer as a separate route
    routes = [[0, c, 0] for c in customers]
    route_of = {c: i for i, c in enumerate(customers)}  # customer -> route index
    # Precompute savings
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            savings.append((s, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))  # descending savings, tie by smaller i,j
    # Merge until exactly truck_count routes
    while len(routes) > truck_count:
        merged = False
        for s, i, j in savings:
            if i not in route_of or j not in route_of:
                continue
            ri = route_of[i]
            rj = route_of[j]
            if ri == rj:
                continue
            # Check that i and j are at ends of their routes
            route_i = routes[ri]
            route_j = routes[rj]
            # i must be at an end of route_i (first or last customer)
            pos_i = route_i.index(i)
            if pos_i != 1 and pos_i != len(route_i)-2:
                continue
            pos_j = route_j.index(j)
            if pos_j != 1 and pos_j != len(route_j)-2:
                continue
            # Determine merge orientation
            # If i is first customer (pos_i==1) and j is last customer (pos_j==len(route_j)-2), connect i after j?
            # We can always connect by having i at end of route_i and j at start of route_j (or vice versa)
            # To keep routes oriented from depot, we need to reverse one route if necessary.
            # Typical merging: if i is at end of its route and j at start of its route, we can append route_j after i.
            # If i is at start and j at end, we can prepend route_j before i (but careful with direction).
            # We'll handle all cases by creating a new route that goes from depot, covers both routes preserving order.
            # Given that each route starts and ends at depot, we need to connect them at the endpoints.
            # The standard savings merge: if i is last customer of route_i and j is first customer of route_j, merge route_i + route_j.
            # If i is first of route_i and j is last of route_j, merge route_j + route_i (reverse order).
            # If i is last of route_i and j is last of route_j, we need to reverse route_j and then merge.
            # If i is first of route_i and j is first of route_j, reverse route_i and then merge.
            # To avoid complexity, we'll always represent routes as [0, ... , 0] and we'll allow reversing.
            # Let's define the end of route as the customer adjacent to depot at the end.
            # We'll just pick the orientation that keeps the order: we want to combine the two routes.
            # Since we only care about feasibility, we can create a new route that is the concatenation of the two routes without the intermediate depot.
            # But we must keep depot at ends. So we can combine route_i and route_j by removing the depot from one of them.
            # For simplicity, we'll always treat i as the last customer (before final depot) and j as the first customer (after initial depot).
            # We'll adjust the roles by possibly reversing one route.
            # We'll ensure i is the last customer of route_i and j is the first customer of route_j.
            # If not, swap the roles (i becomes j? Actually we can just try both orientations).
            # Deterministic: always try to make i the last customer of its route and j the first customer of its route.
            # If that's not possible, try the other orientation.
            # We'll implement a simple check: if i is first customer of route_i and j is first customer of route_j, we can reverse route_i then merge.
            # But easier: just create a new route string without depots and then add depot at ends.
            # We'll extract the customer sequences:
            seq_i = route_i[1:-1]
            seq_j = route_j[1:-1]
            # Depending on the positions of i and j, we can combine.
            # If i is last of seq_i and j is first of seq_j, new_seq = seq_i + seq_j
            # If i is first of seq_i and j is last of seq_j, new_seq = seq_j + seq_i
            # If i is last and j is last, new_seq = seq_i + seq_j[::-1]
            # If i is first and j is first, new_seq = seq_i[::-1] + seq_j
            # We'll find a valid combination:
            possible = []
            # determine if i is at ends of seq_i
            if seq_i[0] == i:
                i_at_start = True
            elif seq_i[-1] == i:
                i_at_start = False
            else:
                continue
            if seq_j[0] == j:
                j_at_start = True
            elif seq_j[-1] == j:
                j_at_start = False
            else:
                continue
            # Now construct new_seq
            if i_at_start == False and j_at_start == True:
                # i last, j first: seq_i + seq_j
                new_seq = seq_i + seq_j
                ok = True
            elif i_at_start == True and j_at_start == False:
                # i first, j last: seq_j + seq_i
                new_seq = seq_j + seq_i
                ok = True
            elif i_at_start == False and j_at_start == False:
                # both last: need to reverse one
                new_seq = seq_i + seq_j[::-1]
                ok = True
            elif i_at_start == True and j_at_start == True:
                # both first: need to reverse one
                new_seq = seq_i[::-1] + seq_j
                ok = True
            else:
                ok = False
            if not ok:
                continue
            new_route = [0] + new_seq + [0]
            # Update route_of for customers in new route
            for c in new_seq:
                route_of[c] = ri
            # Remove route_j
            # Remove route_j from list (we'll replace route_i and delete later)
            # We'll simply mark route_j for deletion by setting it to None, then filter later.
            routes[ri] = new_route
            routes[rj] = None
            merged = True
            break
        if not merged:
            # If no merge possible, break to avoid infinite loop (should not happen)
            break
        # Remove None routes
        routes = [r for r in routes if r is not None]
    # If still too many routes (should not), fill with empty until truck_count
    while len(routes) < truck_count:
        routes.append([0, 0])
    # Ensure exactly truck_count routes
    routes = routes[:truck_count]

    # --- Helper functions (same as original) ---
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

    destroy_ops = ["random", "worst"]
    repair_ops = ["greedy", "regret2"]
    weights_d = [1.0, 1.0]
    weights_r = [1.0, 1.0]
    scores = [0.0, 0.0, 0.0, 0.0]
    usage = [0, 0, 0, 0]
    n_cust = m

    for it in range(max_iter):
        d_idx = random.choices(range(2), weights=weights_d)[0]
        r_idx = random.choices(range(2), weights=weights_r)[0]
        op_idx = d_idx * 2 + r_idx

        q = max(1, n_cust // 10)

        removed = []
        new_routes = copy_routes(current)
        if destroy_ops[d_idx] == "random":
            all_cust = [c for route in new_routes for c in route if c != 0]
            random.shuffle(all_cust)
            for c in all_cust[:q]:
                for route in new_routes:
                    if c in route:
                        route.remove(c)
                        removed.append(c)
                        break
        else:
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

        random.shuffle(removed)
        if repair_ops[r_idx] == "greedy":
            for c in removed:
                best_inc = math.inf
                best_ri = -1
                best_pos = -1
                for ri, route in enumerate(new_routes):
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
        else:  # regret-2
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

        new_max = compute_max(new_routes)
        delta = new_max - current_max

        accepted = False
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = new_routes
            current_max = new_max
            accepted = True
            if new_max < best_max:
                best = copy_routes(new_routes)
                best_max = new_max
                report_best_vrp(best)

        if accepted:
            if new_max < best_max:
                scores[op_idx] += 1.0
            else:
                scores[op_idx] += 0.5
        usage[op_idx] += 1

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

        temp *= cooling_rate

    return best