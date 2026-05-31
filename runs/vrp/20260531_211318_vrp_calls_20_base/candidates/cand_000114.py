import numpy as np
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

    def insertion_delta(route, pos, cust):
        prev = route[pos-1]
        nxt = route[pos]
        return dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt]

    def removal_delta(route, pos):
        prev = route[pos-1]
        nxt = route[pos+1]
        return dist[prev, route[pos]] + dist[route[pos], nxt] - dist[prev, nxt]

    # --- Initial solution generator (random insertion) ---
    def generate_initial_solution(seed):
        random.seed(seed)
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        for cust in customers:
            best_truck = None
            best_pos = None
            best_metric = (float('inf'), float('inf'))  # (max_dist, total_dist)
            for t, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_dist = route_dists[t] + insertion_delta(route, pos, cust)
                    new_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                    new_total = sum(route_dists) + insertion_delta(route, pos, cust)
                    metric = (new_max, new_total)
                    if metric < best_metric:
                        best_metric = metric
                        best_truck = t
                        best_pos = pos
            route = routes[best_truck]
            routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
            route_dists[best_truck] += insertion_delta(route, best_pos, cust)
        return routes, [route_distance(r) for r in routes]

    # --- Local search: 2-opt ---
    def local_search(routes, dists):
        improved = True
        best_max = max(dists)
        best_total = sum(dists)
        while improved:
            improved = False
            for t in range(truck_count):
                route = routes[t]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < dists[t] - 1e-9:
                            new_max = max(dists[:t] + [new_dist] + dists[t+1:])
                            new_total = sum(dists[:t]) + new_dist + sum(dists[t+1:])
                            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                                routes[t] = new_route
                                dists[t] = new_dist
                                best_max = new_max
                                best_total = new_total
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return routes, dists, best_max, best_total

    # --- Mutation: destroy-repair (one iteration) ---
    def mutate(solution, dists):
        routes = [list(r) for r in solution]
        dists = list(dists)
        removal_fraction = 0.1
        num_removals = max(1, int(removal_fraction * (n-1)))
        # Select destroy operator randomly
        if random.random() < 0.5:  # worst removal
            all_contribs = []
            for t, route in enumerate(routes):
                if len(route) <= 2:
                    continue
                for pos in range(1, len(route)-1):
                    contrib = removal_delta(route, pos)
                    all_contribs.append((contrib, t, pos, route[pos]))
            all_contribs.sort(key=lambda x: (-x[0], x[3]))
            to_remove = set()
            for _, t, pos, cust in all_contribs[:num_removals]:
                to_remove.add(cust)
        else:  # random removal
            all_customers = [c for r in routes for c in r[1:-1]]
            random.shuffle(all_customers)
            to_remove = set(all_customers[:num_removals])
        new_routes = []
        new_dists = []
        for route in routes:
            new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
            new_routes.append(new_route)
            new_dists.append(route_distance(new_route))
        removed = list(to_remove)
        # Repair: greedy
        routes_repair = [list(r) for r in new_routes]
        dists_repair = list(new_dists)
        unassigned = list(removed)
        current_max_repair = max(dists_repair)
        for cust in unassigned:
            best_truck = None
            best_pos = None
            best_new_max = float('inf')
            best_new_total = float('inf')
            best_delta = None
            for t, route in enumerate(routes_repair):
                old_dist = dists_repair[t]
                for pos in range(1, len(route)):
                    delta = insertion_delta(route, pos, cust)
                    new_dist = old_dist + delta
                    new_max = max(current_max_repair, new_dist)
                    new_total = sum(dists_repair) + delta
                    if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                        best_new_max = new_max
                        best_new_total = new_total
                        best_truck = t
                        best_pos = pos
                        best_delta = delta
            route = routes_repair[best_truck]
            routes_repair[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
            dists_repair[best_truck] += best_delta
            if dists_repair[best_truck] > current_max_repair:
                current_max_repair = dists_repair[best_truck]
        return routes_repair, dists_repair

    # --- Crossover: route inheritance ---
    def crossover(parent1, parent2, dists1, dists2):
        # Choose better parent as base
        if max(dists1) < max(dists2) or (max(dists1) == max(dists2) and sum(dists1) <= sum(dists2)):
            base_routes = [list(r) for r in parent1]
        else:
            base_routes = [list(r) for r in parent2]
        # For each truck, randomly decide to take route from the other parent if feasible
        other = parent2 if base_routes != parent1 else parent1
        child_routes = [list(r) for r in base_routes]
        # Collect all customers in child
        existing = set()
        for route in child_routes:
            for cust in route[1:-1]:
                existing.add(cust)
        # For each truck, with 0.5 probability, try to replace with other parent's route
        for t in range(truck_count):
            if random.random() < 0.5:
                other_route = other[t]
                new_custs = [c for c in other_route[1:-1] if c not in existing]
                if new_custs:
                    # Replace current route with a route that includes these customers
                    # But we need to keep all customers. Simpler: just add new customers to current route
                    # Actually, we want to inherit route structure. Instead, we'll use a different approach:
                    # Randomly select a subset of routes from one parent and combine with complement from other.
                    pass
        # Simplified: use a random split of customers and rebuild
        # Since route inheritance is complex, we do a simple merging: combine all customers from both parents, then rebuild using cheapest insertion
        # This is more like a mutation, but we keep it as crossover.
        all_customers = list(existing)  # customers from base
        for route in other:
            for cust in route[1:-1]:
                if cust not in existing:
                    all_customers.append(cust)
        # Rebuild from scratch
        random.shuffle(all_customers)
        new_routes = [[0, 0] for _ in range(truck_count)]
        new_dists = [0.0] * truck_count
        for cust in all_customers:
            best_truck = None
            best_pos = None
            best_metric = (float('inf'), float('inf'))
            for t, route in enumerate(new_routes):
                for pos in range(1, len(route)):
                    delta = insertion_delta(route, pos, cust)
                    new_dist = new_dists[t] + delta
                    new_max = max(new_dists[:t] + [new_dist] + new_dists[t+1:])
                    new_total = sum(new_dists) + delta
                    metric = (new_max, new_total)
                    if metric < best_metric:
                        best_metric = metric
                        best_truck = t
                        best_pos = pos
            route = new_routes[best_truck]
            new_routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
            new_dists[best_truck] += insertion_delta(route, best_pos, cust)
        return new_routes, new_dists

    # --- Initialize population ---
    pop_size = 10
    population = []
    for i in range(pop_size):
        routes, dists = generate_initial_solution(i)
        maxd = max(dists)
        totald = sum(dists)
        population.append((routes, dists, maxd, totald))
    # Sort by (max_dist, total_dist)
    population.sort(key=lambda x: (x[2], x[3]))
    best_solution = population[0]
    report_best_vrp(best_solution[0])

    # --- GA parameters ---
    max_gen = min(100, 10 * n)
    mutation_prob = 0.3
    crossover_prob = 0.8
    tournament_size = 3

    for gen in range(max_gen):
        # Tournament selection
        parents = []
        for _ in range(2):
            indices = random.sample(range(pop_size), tournament_size)
            best_idx = min(indices, key=lambda i: (population[i][2], population[i][3]))
            parents.append(population[best_idx])
        p1_routes, p1_dists, p1_max, p1_total = parents[0]
        p2_routes, p2_dists, p2_max, p2_total = parents[1]

        # Crossover
        if random.random() < crossover_prob:
            child_routes, child_dists = crossover(p1_routes, p2_routes, p1_dists, p2_dists)
        else:
            child_routes = [list(r) for r in p1_routes]
            child_dists = list(p1_dists)

        # Mutation
        if random.random() < mutation_prob:
            child_routes, child_dists = mutate(child_routes, child_dists)

        # Local search
        child_routes, child_dists, child_max, child_total = local_search(child_routes, child_dists)

        # Update best global
        if child_max < best_solution[2] - 1e-9 or (abs(child_max - best_solution[2]) < 1e-9 and child_total < best_solution[3]):
            best_solution = (child_routes, child_dists, child_max, child_total)
            report_best_vrp(child_routes)

        # Replace worst in population if child is better
        pop_worst = max(range(pop_size), key=lambda i: (population[i][2], population[i][3]))
        if (child_max, child_total) < (population[pop_worst][2], population[pop_worst][3]):
            population[pop_worst] = (child_routes, child_dists, child_max, child_total)
            population.sort(key=lambda x: (x[2], x[3]))

    return best_solution[0]