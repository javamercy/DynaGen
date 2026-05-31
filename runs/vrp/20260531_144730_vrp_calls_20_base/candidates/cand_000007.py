import numpy as np
import math
from typing import List, Tuple

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> List[List[int]]:
    n = distance_matrix.shape[0]
    if n == 0:
        return []
    if truck_count <= 0:
        return []
    # Initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = list(range(1, n))
    
    def route_dist(route: List[int]) -> float:
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def best_insertion_max(customer: int, routes: List[List[int]], route_dists: List[float]) -> Tuple[float, int, int, float]:
        """Return (best_new_max, best_route_idx, best_pos, second_best_new_max)."""
        best_max = float('inf')
        best_route = -1
        best_pos = -1
        second_best = float('inf')
        for r_idx, route in enumerate(routes):
            # compute current max of other routes
            other_max = 0.0
            for j, d in enumerate(route_dists):
                if j != r_idx and d > other_max:
                    other_max = d
            for pos in range(1, len(route)):
                pred = route[pos-1]
                succ = route[pos]
                new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, customer] + distance_matrix[customer, succ]
                new_max = max(other_max, new_dist)
                if new_max < best_max:
                    second_best = best_max
                    best_max = new_max
                    best_route = r_idx
                    best_pos = pos
                elif new_max < second_best:
                    second_best = new_max
        return best_max, best_route, best_pos, second_best
    
    # Construction: regret-based insertion minimizing max distance
    while unassigned:
        best_regret = -float('inf')
        best_customer = -1
        best_route = -1
        best_pos = -1
        best_new_max = float('inf')
        for c in unassigned:
            bmax, br, bp, second = best_insertion_max(c, routes, route_dists)
            if br == -1:
                continue
            regret = second - bmax if second != float('inf') else float('inf')
            # maximize regret; tie-breaking by customer index
            if regret > best_regret or (regret == best_regret and c < best_customer):
                best_regret = regret
                best_customer = c
                best_route = br
                best_pos = bp
                best_new_max = bmax
        if best_customer == -1:
            break  # should not happen
        # insert
        routes[best_route].insert(best_pos, best_customer)
        route_dists[best_route] = route_dist(routes[best_route])
        unassigned.remove(best_customer)
    
    # report initial solution
    report_best_vrp([r[:] for r in routes])
    
    # Local search: relocate, swap, 2-opt
    n_customers = n - 1
    max_iter = (n_customers + truck_count) * 10  # bounded
    for _ in range(max_iter):
        improved = False
        # find longest route
        max_dist = max(route_dists)
        max_idx = route_dists.index(max_dist)
        
        # relocate from longest route to any other
        route_long = routes[max_idx]
        for i in range(1, len(route_long)-1):
            customer = route_long[i]
            # compute new dist for long route after removal
            pred = route_long[i-1]
            succ = route_long[i+1]
            new_long_dist = route_dists[max_idx] - distance_matrix[pred, customer] - distance_matrix[customer, succ] + distance_matrix[pred, succ]
            # try inserting into other routes
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                # find best insertion in other route to minimize max
                bmax_other, best_other_pos, _ = _best_insertion_in_route(customer, other_route, route_dists[other_idx], distance_matrix)
                # compute resulting max
                other_maxes = []
                for j, d in enumerate(route_dists):
                    if j == max_idx:
                        other_maxes.append(new_long_dist)
                    elif j == other_idx:
                        other_maxes.append(bmax_other)
                    else:
                        other_maxes.append(d)
                new_max = max(other_maxes)
                if new_max < max_dist - 1e-12:
                    # perform move
                    route_long.pop(i)
                    other_route.insert(best_other_pos, customer)
                    route_dists[max_idx] = new_long_dist
                    route_dists[other_idx] = bmax_other
                    improved = True
                    report_best_vrp([r[:] for r in routes])
                    break
            if improved:
                break
        if improved:
            continue
        
        # inter-route swap
        for ri in range(truck_count):
            routei = routes[ri]
            for i in range(1, len(routei)-1):
                custi = routei[i]
                for rj in range(truck_count):
                    if rj < ri:
                        continue
                    routej = routes[rj]
                    for j in range(1, len(routej)-1):
                        if ri == rj and i >= j:
                            continue
                        custj = routej[j]
                        # compute new distances after swap
                        # remove custi from routei, custj from routej
                        # then insert custj into routei, custi into routej
                        # simplification: compute route distances directly
                        new_routei = routei[:i] + routei[i+1:]
                        new_routei = new_routei[:i] + [custj] + new_routei[i:]  # if ri != rj, need to adjust
                        if ri == rj:
                            # swap within same route
                            new_routei = routei[:]
                            new_routei[i], new_routei[j] = routei[j], routei[i]
                            new_dist_i = route_dist(new_routei)
                            other_dists = route_dists.copy()
                            other_dists[ri] = new_dist_i
                            new_max = max(other_dists)
                            if new_max < max_dist - 1e-12:
                                routes[ri] = new_routei
                                route_dists[ri] = new_dist_i
                                improved = True
                                report_best_vrp([r[:] for r in routes])
                                break
                            continue
                        # ri != rj
                        new_routej = routej[:j] + routej[j+1:]
                        # insert custj into routei at position i (original index, but after removal maybe shift)
                        new_routei = routei[:]
                        new_routei.pop(i)
                        # find best position to insert custj in new_routei (minimizing max? but we just test one insertion: at i?)
                        # For simplicity, we test insertion at each pos, but to keep bounded, we can test the original i position or use best
                        # Here we'll try all positions in new_routei for insertion of custj
                        best_i_dist = float('inf')
                        best_i_pos = -1
                        for p in range(1, len(new_routei)+1):
                            temp = new_routei[:p] + [custj] + new_routei[p:]
                            d = route_dist(temp)
                            if d < best_i_dist:
                                best_i_dist = d
                                best_i_pos = p
                        # similarly insert custi into new_routej
                        new_routej = routej[:]
                        new_routej.pop(j)
                        best_j_dist = float('inf')
                        best_j_pos = -1
                        for p in range(1, len(new_routej)+1):
                            temp = new_routej[:p] + [custi] + new_routej[p:]
                            d = route_dist(temp)
                            if d < best_j_dist:
                                best_j_dist = d
                                best_j_pos = p
                        # apply
                        new_routei_final = new_routei[:best_i_pos] + [custj] + new_routei[best_i_pos:]
                        new_routej_final = new_routej[:best_j_pos] + [custi] + new_routej[best_j_pos:]
                        new_dists = route_dists.copy()
                        new_dists[ri] = route_dist(new_routei_final)
                        new_dists[rj] = route_dist(new_routej_final)
                        new_max = max(new_dists)
                        if new_max < max_dist - 1e-12:
                            routes[ri] = new_routei_final
                            routes[rj] = new_routej_final
                            route_dists[ri] = new_dists[ri]
                            route_dists[rj] = new_dists[rj]
                            improved = True
                            report_best_vrp([r[:] for r in routes])
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        
        # intra-route 2-opt
        for ri in range(truck_count):
            route = routes[ri]
            best_delta = 0.0
            best_i, best_j = -1, -1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old_edges = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new_edges = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    delta = new_edges - old_edges
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            if best_delta < -1e-12:
                # apply 2-opt
                route[best_i:best_j+1] = reversed(route[best_i:best_j+1])
                route_dists[ri] = route_dist(route)
                improved = True
                # no need to report if max didn't improve? But report anyway
                report_best_vrp([r[:] for r in routes])
        if not improved:
            break
    return routes

def _best_insertion_in_route(customer: int, route: List[int], current_dist: float, dist_matrix: np.ndarray) -> Tuple[float, int, float]:
    """Return (new_dist, best_pos, delta) for inserting customer into route at the best position."""
    best_new = float('inf')
    best_pos = -1
    best_delta = 0.0
    for pos in range(1, len(route)+1):
        pred = route[pos-1] if pos > 0 else route[0]
        succ = route[pos] if pos < len(route) else route[-1]
        # Actually for position at end, we need to consider pred and succ
        # But route always has at least two 0's, so we can iterate from 1 to len(route)
        # Better: for pos in range(1, len(route)): which is before index pos
        pass
    # simplified: use the same logic as before
    for pos in range(1, len(route)):
        pred = route[pos-1]
        succ = route[pos]
        delta = dist_matrix[pred, customer] + dist_matrix[customer, succ] - dist_matrix[pred, succ]
        new = current_dist + delta
        if new < best_new:
            best_new = new
            best_pos = pos
            best_delta = delta
    # also consider inserting at end? But route ends with 0, so inserting before last 0 is covered by pos = len(route)-1
    return best_new, best_pos, best_delta