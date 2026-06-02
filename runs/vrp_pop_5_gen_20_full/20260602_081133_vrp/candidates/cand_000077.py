import numpy as np
import random
import itertools
from collections import Counter

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    n_cust = len(customers)
    
    # --- Regret-3 insertion ---
    def regret_insertion():
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(customers)
        while unassigned:
            best_val = -1.0
            best_cust = None
            best_route = None
            best_pos = None
            best_ins_cost = float('inf')
            for cust in unassigned:
                # compute insertion costs for all routes and positions
                options = []
                for r_idx, route in enumerate(routes):
                    if len(route) == 2:
                        cost = distance_matrix[0][cust] + distance_matrix[cust][0]
                        options.append((cost, r_idx, 1))
                    else:
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            cost = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                            options.append((cost, r_idx, pos))
                if not options:
                    continue
                options.sort(key=lambda x: x[0])
                costs_sorted = [opt[0] for opt in options]
                best = costs_sorted[0]
                second = costs_sorted[1] if len(costs_sorted) > 1 else best
                third = costs_sorted[2] if len(costs_sorted) > 2 else best
                regret = (second - best) + (third - best)
                if regret > best_val or (regret == best_val and best < best_ins_cost):
                    best_val = regret
                    best_cust = cust
                    best_ins_cost = best
                    best_route = options[0][1]
                    best_pos = options[0][2]
            if best_cust is None:
                break
            routes[best_route].insert(best_pos, best_cust)
            unassigned.remove(best_cust)
        return routes
    
    best_solution = None
    best_max = float('inf')
    
    # Initial construction
    routes = regret_insertion()
    
    def route_dist(route):
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    
    def max_dist(routes):
        return max(route_dist(r) for r in routes)
    
    best_solution = [list(r) for r in routes]
    best_max = max_dist(routes)
    report_best_vrp(best_solution)
    
    # --- Iterated Local Search ---
    n_iter = min(10, n_cust // 5 if n_cust > 0 else 1)
    for iteration in range(n_iter):
        # Shake: remove some customers from the longest route
        lengths = [route_dist(r) for r in routes]
        longest_idx = np.argmax(lengths)
        longest_route = routes[longest_idx]
        if len(longest_route) > 3:
            # Remove a fraction of customers (at least 1) from longest route
            remove_count = max(1, len(longest_route) // 4)
            # Remove customers in the middle (skip first and last depot)
            indices = list(range(1, len(longest_route)-1))
            random.shuffle(indices)
            removed_custs = []
            for idx in sorted(indices[:remove_count], reverse=True):
                removed_custs.append(longest_route.pop(idx))
            routes[longest_idx] = longest_route
            # Reinsert the removed customers using regret-3
            unassigned = set(removed_custs)
            while unassigned:
                best_val = -1.0
                best_cust = None
                best_route = None
                best_pos = None
                best_ins_cost = float('inf')
                for cust in unassigned:
                    options = []
                    for r_idx, route in enumerate(routes):
                        if len(route) == 2:
                            cost = distance_matrix[0][cust] + distance_matrix[cust][0]
                            options.append((cost, r_idx, 1))
                        else:
                            for pos in range(1, len(route)):
                                prev = route[pos-1]
                                nxt = route[pos]
                                cost = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                                options.append((cost, r_idx, pos))
                    if not options:
                        continue
                    options.sort(key=lambda x: x[0])
                    costs_sorted = [opt[0] for opt in options]
                    best = costs_sorted[0]
                    second = costs_sorted[1] if len(costs_sorted) > 1 else best
                    third = costs_sorted[2] if len(costs_sorted) > 2 else best
                    regret = (second - best) + (third - best)
                    if regret > best_val or (regret == best_val and best < best_ins_cost):
                        best_val = regret
                        best_cust = cust
                        best_ins_cost = best
                        best_route = options[0][1]
                        best_pos = options[0][2]
                if best_cust is None:
                    break
                routes[best_route].insert(best_pos, best_cust)
                unassigned.remove(best_cust)
            # Update max
            current_max = max_dist(routes)
            if current_max < best_max - 1e-9:
                best_solution = [list(r) for r in routes]
                best_max = current_max
                report_best_vrp(best_solution)
        
        # Local search: improve max route
        improved = True
        max_local_iters = n_cust * 2
        local_count = 0
        while improved and local_count < max_local_iters:
            improved = False
            local_count += 1
            # Compute lengths
            lengths = [route_dist(r) for r in routes]
            max_len = max(lengths)
            # Try to reduce max by moving customers from longest route
            longest_idx = lengths.index(max_len)
            longest_route = routes[longest_idx]
            # Relocate move: move a customer from longest route to another route
            for pos in range(1, len(longest_route)-1):
                cust = longest_route[pos]
                # compute removal cost
                prev = longest_route[pos-1]
                nxt = longest_route[pos+1]
                removal_gain = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                new_len_longest = lengths[longest_idx] - removal_gain
                # try insert into other routes
                for r_idx in range(truck_count):
                    if r_idx == longest_idx:
                        continue
                    route = routes[r_idx]
                    for ins_pos in range(1, len(route)):
                        prev2 = route[ins_pos-1]
                        nxt2 = route[ins_pos]
                        insert_cost = distance_matrix[prev2][cust] + distance_matrix[cust][nxt2] - distance_matrix[prev2][nxt2]
                        new_len_other = lengths[r_idx] + insert_cost
                        new_max = max(new_len_longest, new_len_other, max([lengths[i] for i in range(truck_count) if i != longest_idx and i != r_idx]))
                        if new_max < max_len - 1e-9:
                            # Apply move
                            routes[longest_idx].pop(pos)
                            routes[r_idx].insert(ins_pos, cust)
                            improved = True
                            current_max = max_dist(routes)
                            if current_max < best_max - 1e-9:
                                best_solution = [list(r) for r in routes]
                                best_max = current_max
                                report_best_vrp(best_solution)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Exchange move: swap customers between longest route and another
            for r_idx in range(truck_count):
                if r_idx == longest_idx:
                    continue
                other_route = routes[r_idx]
                for pos1 in range(1, len(longest_route)-1):
                    for pos2 in range(1, len(other_route)-1):
                        cust1 = longest_route[pos1]
                        cust2 = other_route[pos2]
                        # compute old costs
                        prev1 = longest_route[pos1-1]
                        nxt1 = longest_route[pos1+1]
                        prev2 = other_route[pos2-1]
                        nxt2 = other_route[pos2+1]
                        old1 = distance_matrix[prev1][cust1] + distance_matrix[cust1][nxt1]
                        old2 = distance_matrix[prev2][cust2] + distance_matrix[cust2][nxt2]
                        new1 = distance_matrix[prev1][cust2] + distance_matrix[cust2][nxt1]
                        new2 = distance_matrix[prev2][cust1] + distance_matrix[cust1][nxt2]
                        new_len1 = lengths[longest_idx] - old1 + new1
                        new_len2 = lengths[r_idx] - old2 + new2
                        other_lengths = [lengths[i] for i in range(truck_count) if i != longest_idx and i != r_idx]
                        new_max = max(new_len1, new_len2, max(other_lengths))
                        if new_max < max_len - 1e-9:
                            # swap
                            longest_route[pos1], other_route[pos2] = other_route[pos2], longest_route[pos1]
                            routes[longest_idx] = longest_route
                            routes[r_idx] = other_route
                            improved = True
                            current_max = max_dist(routes)
                            if current_max < best_max - 1e-9:
                                best_solution = [list(r) for r in routes]
                                best_max = current_max
                                report_best_vrp(best_solution)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # 2-opt on longest route
            if len(longest_route) > 3:
                best_gain = 0
                best_ij = None
                for i in range(1, len(longest_route)-2):
                    for j in range(i+1, len(longest_route)-1):
                        if j == i+1:
                            continue
                        old = distance_matrix[longest_route[i-1]][longest_route[i]] + distance_matrix[longest_route[j]][longest_route[j+1]]
                        new = distance_matrix[longest_route[i-1]][longest_route[j]] + distance_matrix[longest_route[i]][longest_route[j+1]]
                        gain = old - new
                        if gain > best_gain:
                            best_gain = gain
                            best_ij = (i, j)
                if best_gain > 1e-9:
                    i, j = best_ij
                    longest_route[i:j+1] = reversed(longest_route[i:j+1])
                    routes[longest_idx] = longest_route
                    improved = True
                    current_max = max_dist(routes)
                    if current_max < best_max - 1e-9:
                        best_solution = [list(r) for r in routes]
                        best_max = current_max
                        report_best_vrp(best_solution)
    
    # Return best found solution
    # Ensure exactly truck_count routes, all starting and ending at 0
    result = [list(r) for r in best_solution]
    # Ensure all customers are assigned (should be)
    assigned = set([c for route in result for c in route[1:-1]])
    if len(assigned) != n_cust:
        # fallback: should not happen
        result = [list(r) for r in regret_insertion()]
    return result