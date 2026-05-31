import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    dist = distance_matrix

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    def total_max(routes):
        m = max(route_distance(r) for r in routes)
        t = sum(route_distance(r) for r in routes)
        return m, t

    # generate initial solution using parent's greedy insertion
    def greedy_initial(seed):
        random.seed(seed)
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in customers:
            best_truck = None
            best_pos = None
            best_max = float('inf')
            best_total = float('inf')
            for t, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_routes = routes[:t] + [new_route] + routes[t+1:]
                    nm, nt = total_max(new_routes)
                    if nm < best_max or (nm == best_max and nt < best_total):
                        best_max = nm
                        best_total = nt
                        best_truck = t
                        best_pos = pos
            routes[best_truck].insert(best_pos, cust)
        return routes

    # helper: best insertion of a single customer into current routes
    def best_insertion(routes, cust):
        best_max = float('inf')
        best_total = float('inf')
        best_truck = None
        best_pos = None
        for t, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_routes = routes[:t] + [new_route] + routes[t+1:]
                nm, nt = total_max(new_routes)
                if nm < best_max or (nm == best_max and nt < best_total):
                    best_max = nm
                    best_total = nt
                    best_truck = t
                    best_pos = pos
        routes[best_truck].insert(best_pos, cust)

    # crossover: route-based combine two parents
    def crossover(routes1, routes2):
        # randomly select a subset of routes from parent1 and parent2, then repair
        # ensure at least one route from each parent? we can pick random split
        # will pick each route from p1 with prob 0.5, but must cover all customers
        # simple: take all routes from p1, then for each customer that is duplicated (appears in both) remove from p2? but easier: take routes from p1, then add routes from p2 that don't have customers already present, then repair missing.
        # to keep it bounded, we'll do: pick a random number of routes from p1 (k), then take those routes, and then from p2 take routes that don't conflict (no overlapping customers with already selected), then greedily insert remaining customers.
        # simpler: take all customers from p1 routes; then for p2 routes, try to add routes that have new customers not in p1; finally, missing customers are inserted greedily.
        used = set()
        child_routes = []
        # add all p1 routes
        for r in routes1:
            custs = [c for c in r if c != 0]
            used.update(custs)
            child_routes.append(list(r))
        # add p2 routes containing only new customers
        for r in routes2:
            custs = [c for c in r if c != 0]
            if used.isdisjoint(custs):
                used.update(custs)
                child_routes.append(list(r))
        # ensure we have exactly truck_count routes; if too many, remove empty routes? but we start with truck_count exactly, so we need to maintain truck_count routes.
        # The problem: we must output exactly truck_count routes. So before crossover, we assume both parents have truck_count routes. After adding routes from both, we may have more than truck_count routes. We need to merge or delete some. Alternatively, we can do crossover differently: select a subset of routes from each parent such that total routes = truck_count and customers covered (possibly with duplicates resolved). But that's complex.
        # For simplicity, we'll use a different crossover: order crossover on the permutation of customers, then decode into routes via greedy insertion? That might change route structure drastically but is feasible.
        # Let's implement ordered crossover (OX) on a permutation representation.
        # Represent solution as a permutation of customers (excluding depots). Then to get routes, we decode greedily: start first route, add customers until adding next would exceed some bound? But without capacity, we need a method to split into truck_count routes. That's challenging.
        # Alternative: Perform crossover by exchanging a segment of the permutation, then reassign routes via best insertion? That would be very similar to destroy-repair, which we want to avoid.
        # Given time, I'll stick with a simple but effective crossover: select a random truck index, exchange all customers of that truck between two parents, then repair duplicates/missing.
        # This is similar to a route-exchange crossover.
        # We'll pick a random truck idx, and swap its route between parents. Then each child will have that route from one parent, and the rest from the other parent. Then there may be duplicate customers (if the swapped route and original routes share customers) and missing customers. We'll remove duplicates from the other routes (keep only first occurrence) and then greedily insert missing customers.
        # That's bounded and straightforward.

    # I'll implement the route-exchange crossover.
    def crossover(p1, p2):
        # pick a random truck index
        t = random.randint(0, truck_count-1)
        # child1 inherits p1's t-th route, and p2's other routes
        child1_routes = [list(p2[i]) for i in range(truck_count)]
        child1_routes[t] = list(p1[t])
        # child2: opposite
        child2_routes = [list(p1[i]) for i in range(truck_count)]
        child2_routes[t] = list(p2[t])
        # repair duplicates and missing for child1
        for routes in [child1_routes, child2_routes]:
            # count occurrences of each customer
            seen = {}
            for r in routes:
                for c in r[1:-1]:
                    seen[c] = seen.get(c, 0) + 1
            # remove duplicates from routes other than the swapped one (or all? better: keep first occurrence, remove later)
            for i, r in enumerate(routes):
                if i == t:
                    continue
                new_route = [0]
                for c in r[1:-1]:
                    if seen[c] > 1 and c not in [cc for cc in new_route if cc != 0]:
                        # skip duplicate
                        seen[c] -= 1
                    else:
                        new_route.append(c)
                new_route.append(0)
                routes[i] = new_route
            # now some customers may be missing (if duplicates were removed and not present elsewhere)
            present = set()
            for r in routes:
                for c in r[1:-1]:
                    present.add(c)
            missing = [c for c in range(1,n) if c not in present]
            # greedily insert missing
            for cust in missing:
                best_insertion(routes, cust)
        return child1_routes, child2_routes

    # mutation: relocate a random customer to a random different position
    def mutate(routes):
        # select a random customer
        all_custs = [(t, i, c) for t, r in enumerate(routes) for i, c in enumerate(r[1:-1], 1)]
        if not all_custs:
            return
        t_old, pos, cust = random.choice(all_custs)
        # remove
        route_old = routes[t_old]
        route_old.pop(pos)
        # choose another truck (possibly same) but different position
        t_new = random.randint(0, truck_count-1)
        route_new = routes[t_new]
        # if same route, ensure different position
        if t_new == t_old:
            # may insert at any position except the original (but already removed)
            pos_options = list(range(1, len(route_new)+1))
            if pos-1 in pos_options:
                pos_options.remove(pos-1)
            if not pos_options:
                # revert
                route_old.insert(pos, cust)
                return
            pos_new = random.choice(pos_options)
            route_new.insert(pos_new, cust)
        else:
            # find best insertion in t_new to maintain quality? use greedy
            # for simplicity, insert at random position
            pos_new = random.randint(1, len(route_new)-1)
            route_new.insert(pos_new, cust)
        # note: this can worsen solution, but mutation is for diversity

    # local search: 2-opt on each route, then relocate inter-route improvement
    def local_search(routes):
        improved = True
        max_iter = 5 * n
        iters = 0
        while improved and iters < max_iter:
            improved = False
            # intra-route 2-opt
            for t, route in enumerate(routes):
                if len(route) <= 3:
                    continue
                best_d = route_distance(route)
                best_route = list(route)
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        # reverse segment i..j
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_d = route_distance(new_route)
                        if new_d < best_d - 1e-12:
                            best_d = new_d
                            best_route = new_route
                            improved = True
                routes[t] = best_route
            # inter-route relocate: try moving each customer to a different route at best position
            for t_old, route in enumerate(routes):
                for i, cust in enumerate(route[1:-1], 1):
                    cur_routes = [list(r) for r in routes]
                    # remove cust
                    cur_routes[t_old].pop(i)
                    best_max = float('inf')
                    best_total = float('inf')
                    best_truck = None
                    best_pos = None
                    for t_new, r in enumerate(cur_routes):
                        if t_new == t_old:
                            continue
                        for pos in range(1, len(r)):
                            new_route = r[:pos] + [cust] + r[pos:]
                            tmp_routes = cur_routes[:]
                            tmp_routes[t_new] = new_route
                            nm, nt = total_max(tmp_routes)
                            if nm < best_max or (nm == best_max and nt < best_total):
                                best_max = nm
                                best_total = nt
                                best_truck = t_new
                                best_pos = pos
                    if best_truck is not None:
                        # check if moving improves overall max (or total if max equal)
                        old_max, old_total = total_max(routes)
                        if best_max < old_max - 1e-12 or (abs(best_max - old_max) < 1e-12 and best_total < old_total):
                            # perform move
                            routes = [list(r) for r in cur_routes]
                            routes[best_truck].insert(best_pos, cust)
                            improved = True
                            break  # restart after one improvement
                if improved:
                    break
            iters += 1
        return routes

    # initialize population
    pop_size = 10
    pop = []
    for i in range(pop_size):
        ind = greedy_initial(i+1)  # different seeds
        pop.append((ind, total_max(ind)))
    # evaluate
    best_ind = min(pop, key=lambda x: (x[1][0], x[1][1]))
    best_routes = [list(r) for r in best_ind[0]]
    report_best_vrp(best_routes)

    # GA parameters
    gen_max = min(50, max(10, n//2))
    for gen in range(gen_max):
        new_pop = []
        # elitism
        elite = min(pop, key=lambda x: (x[1][0], x[1][1]))
        new_pop.append((elite[0], elite[1]))
        while len(new_pop) < pop_size:
            # tournament selection
            candidates = random.sample(pop, 3)
            parent1 = min(candidates, key=lambda x: (x[1][0], x[1][1]))[0]
            candidates = random.sample(pop, 3)
            parent2 = min(candidates, key=lambda x: (x[1][0], x[1][1]))[0]
            if random.random() < 0.8:
                child1, child2 = crossover(parent1, parent2)
                for child in [child1, child2]:
                    # mutation
                    if random.random() < 0.2:
                        mutate(child)
                    # local search
                    child = local_search(child)
                    new_pop.append((child, total_max(child)))
                    if len(new_pop) >= pop_size:
                        break
            else:
                # direct copy with mutation
                child = [list(r) for r in parent1]
                if random.random() < 0.2:
                    mutate(child)
                child = local_search(child)
                new_pop.append((child, total_max(child)))
        pop = new_pop[:pop_size]
        # update best
        candidate_best = min(pop, key=lambda x: (x[1][0], x[1][1]))
        if candidate_best[1][0] < best_ind[1][0] - 1e-12 or (abs(candidate_best[1][0] - best_ind[1][0]) < 1e-12 and candidate_best[1][1] < best_ind[1][1]):
            best_ind = candidate_best
            best_routes = [list(r) for r in best_ind[0]]
            report_best_vrp(best_routes)
    return best_routes