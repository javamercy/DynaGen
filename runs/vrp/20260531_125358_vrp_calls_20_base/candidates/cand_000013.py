import numpy as np
import math
from heapq import heappush, heappop
import itertools

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    
    routes = [[0, 0] for _ in range(truck_count)]
    dist = distance_matrix
    
    def route_distance(route):
        d = 0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d
    
    def best_insertion(customer, route):
        best_pos = -1
        best_inc = float('inf')
        for i in range(1, len(route)):
            prev = route[i-1]
            next_ = route[i]
            inc = dist[prev, customer] + dist[customer, next_] - dist[prev, next_]
            if inc < best_inc:
                best_inc = inc
                best_pos = i
        return best_pos, best_inc
    
    def compute_regret_info(customer):
        incs = []
        for r_idx, route in enumerate(routes):
            pos, inc = best_insertion(customer, route)
            incs.append((inc, pos, r_idx))
        incs.sort(key=lambda x: x[0])
        if len(incs) >= 2:
            best_inc = incs[0][0]
            second_best_inc = incs[1][0]
            regret = second_best_inc - best_inc
        else:
            best_inc = incs[0][0]
            regret = 0.0
        best_pos = incs[0][1]
        best_route = incs[0][2]
        return regret, best_inc, best_pos, best_route
    
    remaining_customers = set(customers)
    while remaining_customers:
        regret_list = []
        for c in remaining_customers:
            regret, best_inc, best_pos, best_route = compute_regret_info(c)
            regret_list.append((regret, best_inc, -c, c, best_pos, best_route))
        regret_list.sort(key=lambda x: (-x[0], -x[1], x[2]))
        _, _, _, customer, best_pos, best_route = regret_list[0]
        route = routes[best_route]
        route.insert(best_pos, customer)
        remaining_customers.remove(customer)
    
    route_dists = [route_distance(r) for r in routes]
    max_dist = max(route_dists)
    best_routes = [list(r) for r in routes]
    best_max = max_dist
    report_best_vrp([list(r) for r in routes])
    
    # Intensified local search
    def evaluate_move(new_route_i, new_route_j, i, j):
        # compute new max assuming only routes i and j change
        new_dists = route_dists[:]
        new_dists[i] = route_distance(new_route_i)
        if j != i:
            new_dists[j] = route_distance(new_route_j)
        new_max = max(new_dists)
        return new_max
    
    improved = True
    iteration = 0
    max_iter = n * truck_count * 2
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        best_new_max = best_max
        best_move = None
        
        # Intra-route 2-opt
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-1):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < route_dists[r_idx]:
                        new_max = max(new_dist, max(route_dists[:r_idx] + route_dists[r_idx+1:]))
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('2opt', r_idx, i, j)
        
        # Inter-route relocate
        for src_idx, src_route in enumerate(routes):
            if len(src_route) <= 2:
                continue
            for pos in range(1, len(src_route)-1):
                customer = src_route[pos]
                new_src_route = src_route[:pos] + src_route[pos+1:]
                for dst_idx, dst_route in enumerate(routes):
                    if dst_idx == src_idx:
                        continue
                    for ins_pos in range(1, len(dst_route)):
                        new_dst_route = dst_route[:ins_pos] + [customer] + dst_route[ins_pos:]
                        new_max = evaluate_move(new_src_route, new_dst_route, src_idx, dst_idx)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('relocate', src_idx, pos, dst_idx, ins_pos)
        
        # Inter-route swap
        for i_idx, i_route in enumerate(routes):
            if len(i_route) <= 2:
                continue
            for i_pos in range(1, len(i_route)-1):
                cust_i = i_route[i_pos]
                for j_idx in range(i_idx+1, len(routes)):
                    j_route = routes[j_idx]
                    if len(j_route) <= 2:
                        continue
                    for j_pos in range(1, len(j_route)-1):
                        cust_j = j_route[j_pos]
                        new_i_route = i_route[:i_pos] + [cust_j] + i_route[i_pos+1:]
                        new_j_route = j_route[:j_pos] + [cust_i] + j_route[j_pos+1:]
                        new_max = evaluate_move(new_i_route, new_j_route, i_idx, j_idx)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('swap', i_idx, i_pos, j_idx, j_pos)
        
        # Inter-route or-opt: move a segment of up to 2 customers
        for src_idx, src_route in enumerate(routes):
            if len(src_route) <= 3:
                continue
            for start in range(1, len(src_route)-2):
                for length in [1, 2]:
                    end = start + length - 1
                    if end >= len(src_route)-1:
                        continue
                    segment = src_route[start:end+1]
                    new_src_route = src_route[:start] + src_route[end+1:]
                    for dst_idx, dst_route in enumerate(routes):
                        if dst_idx == src_idx:
                            continue
                        for ins_pos in range(1, len(dst_route)):
                            new_dst_route = dst_route[:ins_pos] + segment + dst_route[ins_pos:]
                            new_max = evaluate_move(new_src_route, new_dst_route, src_idx, dst_idx)
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = ('oropt', src_idx, start, end, dst_idx, ins_pos)
        
        if best_move is not None and best_new_max < best_max:
            # Apply best move
            move_type = best_move[0]
            if move_type == '2opt':
                r_idx, i, j = best_move[1], best_move[2], best_move[3]
                routes[r_idx] = routes[r_idx][:i] + routes[r_idx][i:j+1][::-1] + routes[r_idx][j+1:]
            elif move_type == 'relocate':
                src_idx, pos, dst_idx, ins_pos = best_move[1], best_move[2], best_move[3], best_move[4]
                customer = routes[src_idx].pop(pos)
                routes[dst_idx].insert(ins_pos, customer)
            elif move_type == 'swap':
                i_idx, i_pos, j_idx, j_pos = best_move[1], best_move[2], best_move[3], best_move[4]
                routes[i_idx][i_pos], routes[j_idx][j_pos] = routes[j_idx][j_pos], routes[i_idx][i_pos]
            elif move_type == 'oropt':
                src_idx, start, end, dst_idx, ins_pos = best_move[1], best_move[2], best_move[3], best_move[4], best_move[5]
                segment = routes[src_idx][start:end+1]
                del routes[src_idx][start:end+1]
                routes[dst_idx][ins_pos:ins_pos] = segment
            # Update route distances
            route_dists = [route_distance(r) for r in routes]
            best_max = max(route_dists)
            best_routes = [list(r) for r in routes]
            report_best_vrp([list(r) for r in routes])
            improved = True
    
    return best_routes