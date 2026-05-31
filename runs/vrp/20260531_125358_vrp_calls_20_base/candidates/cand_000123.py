import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def balanced_greedy_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0] * truck_count
        current_max = 0
        cust_list = list(customers)
        random.shuffle(cust_list)
        for cust in cust_list:
            best_truck = None
            best_pos = None
            best_new_max = float('inf')
            best_inc = float('inf')
            for t in range(truck_count):
                route = routes[t]
                for pos in range(1, len(route)):
                    # insertion cost
                    inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                    new_len = lengths[t] + inc
                    # new max among trucks
                    new_max = max(new_len, current_max if t != current_max_owner else max([lengths[i] for i in range(truck_count) if i != t]))
                    # Actually we need to know which truck currently has max. Simpler: keep array of lengths and compute new overall max.
                    # But we can compute new_max = max(new_len, max(lengths[:t] + lengths[t+1:])). That's O(truck_count).
                    other_max = max(lengths[:t] + lengths[t+1:]) if truck_count > 1 else 0
                    new_max = max(new_len, other_max)
                    if new_max < best_new_max or (new_max == best_new_max and t < best_truck):
                        best_new_max = new_max
                        best_truck = t
                        best_pos = pos
                        best_inc = inc
            # insert
            route = routes[best_truck]
            route.insert(best_pos, cust)
            lengths[best_truck] += best_inc
            current_max = max(lengths)
            # update current_max_owner if needed (not used further)
        return routes, lengths

    # construction
    routes, lengths = balanced_greedy_construction()
    best_routes = [r[:] for r in routes]
    best_max = max(lengths)
    report_best_vrp(best_routes)

    # Simulated Annealing
    current_routes = [r[:] for r in routes]
    current_lengths = lengths[:]
    current_max = best_max

    # SA parameters
    max_iters = n * truck_count * 10
    temp_update_interval = n * truck_count
    T = best_max * 0.1  # initial temperature
    alpha = 0.95

    for iteration in range(max_iters):
        # generate random neighbor
        move_type = random.choice(['relocate', 'swap', '2opt'])
        new_routes = [r[:] for r in current_routes]
        new_lengths = current_lengths[:]

        if move_type == 'relocate':
            # pick a random customer
            cust = random.choice(customers)
            # find source truck and position
            src_truck = None
            src_pos = None
            for t in range(truck_count):
                route = new_routes[t]
                if cust in route:
                    src_truck = t
                    src_pos = route.index(cust)
                    break
            # choose destination truck different from source
            possible_dst = [t for t in range(truck_count) if t != src_truck]
            if not possible_dst:
                continue
            dst_truck = random.choice(possible_dst)
            dst_route = new_routes[dst_truck]
            if len(dst_route) <= 2:
                # cannot insert because only depot? Actually can insert between depots, route length is 2 (depots only)
                # insertion positions between 0 and 0: only pos=1
                pos = 1
            else:
                pos = random.randint(1, len(dst_route)-1)
            # remove from source
            new_routes[src_truck] = new_routes[src_truck][:src_pos] + new_routes[src_truck][src_pos+1:]
            # insert into destination
            new_routes[dst_truck] = dst_route[:pos] + [cust] + dst_route[pos:]
            # recalc lengths
            new_lengths[src_truck] = route_distance(new_routes[src_truck])
            new_lengths[dst_truck] = route_distance(new_routes[dst_truck])

        elif move_type == 'swap':
            # pick two customers from different trucks
            trucks_with_customers = [t for t in range(truck_count) if len(new_routes[t]) > 2]
            if len(trucks_with_customers) < 2:
                continue
            t1, t2 = random.sample(trucks_with_customers, 2)
            route1 = new_routes[t1]
            route2 = new_routes[t2]
            # pick positions (1 to len-2)
            pos1 = random.randint(1, len(route1)-2)
            pos2 = random.randint(1, len(route2)-2)
            cust1 = route1[pos1]
            cust2 = route2[pos2]
            # swap
            route1[pos1] = cust2
            route2[pos2] = cust1
            new_lengths[t1] = route_distance(route1)
            new_lengths[t2] = route_distance(route2)

        else:  # 2opt
            # pick a random truck with at least 4 nodes (including depots)
            eligible = [t for t in range(truck_count) if len(new_routes[t]) >= 4]
            if not eligible:
                continue
            t = random.choice(eligible)
            route = new_routes[t]
            # pick two indices i, j such that 1 <= i < j <= len(route)-2
            i = random.randint(1, len(route)-3)
            j = random.randint(i+1, len(route)-2)
            # reverse segment i to j
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            new_routes[t] = new_route
            new_lengths[t] = route_distance(new_route)

        # compute new max
        new_max = max(new_lengths)
        delta = new_max - current_max
        if delta < 0 or random.random() < math.exp(-delta / T):
            # accept
            current_routes = new_routes
            current_lengths = new_lengths
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                report_best_vrp(best_routes)
        # update temperature
        if (iteration + 1) % temp_update_interval == 0:
            T *= alpha

    return best_routes