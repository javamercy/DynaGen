import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def max_route_len(routes):
        return max(route_length(r) for r in routes) if routes else float('inf')

    best_routes = None
    best_max = float('inf')
    max_attempts = max(2, n // 5)

    for _ in range(max_attempts):
        # Probabilistic regret construction
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            regrets = []
            for cust in unassigned:
                insert_options = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_len = route_length(route) + cost
                        other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                        new_max = max(new_len, *other_lens)
                        insert_options.append((new_max, cost, r_idx, pos))
                if not insert_options:
                    continue
                insert_options.sort(key=lambda x: (x[0], x[1]))
                best = insert_options[0]
                second = insert_options[1] if len(insert_options) > 1 else (best[0]+1e9, best[1]+1e9, -1, -1)
                regret = max(second[0] - best[0], 0)
                regrets.append((regret, cust, best))
            if not regrets:
                break
            total_regret = sum(r for r,_,_ in regrets) + 1e-9
            probs = [(r+1e-9)/total_regret for r,_,_ in regrets]
            rnd = random.random()
            cumulative = 0.0
            chosen = None
            for prob, (_, cust, best) in zip(probs, regrets):
                cumulative += prob
                if rnd <= cumulative:
                    chosen = (cust, best)
                    break
            if chosen is None:
                cust, best = max(regrets, key=lambda x: x[0])[1:]
            else:
                cust, best = chosen
            _, _, r_idx, pos = best
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)

        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

        # Simulated annealing local search
        initial_temp = current_max * 0.1
        temp = initial_temp
        cooling_rate = 0.99
        max_iter = n * truck_count
        for iteration in range(max_iter):
            improved = False
            lengths = [route_length(r) for r in routes]
            current_max_local = max(lengths)
            max_idx = int(np.argmax(lengths))
            max_route = routes[max_idx]
            if len(max_route) > 2:
                moves = []
                for i, cust in enumerate(max_route[1:-1]):
                    new_max_route = [x for x in max_route if x != cust]
                    new_max_len = route_length(new_max_route)
                    for r_idx in range(truck_count):
                        if r_idx == max_idx:
                            continue
                        other_route = routes[r_idx]
                        for pos in range(1, len(other_route)):
                            new_other = other_route[:pos] + [cust] + other_route[pos:]
                            new_other_len = route_length(new_other)
                            other_lens = [lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)]
                            new_max_candidate = max(new_max_len, new_other_len, *other_lens)
                            delta = current_max_local - new_max_candidate
                            moves.append((delta, new_max_candidate, cust, max_idx, r_idx, pos))
                if moves:
                    moves.sort(key=lambda x: -x[0])
                    best_delta = moves[0][0]
                    if best_delta > 0:
                        # accept improvement
                        delta, new_max_val, cust, from_idx, to_idx, pos = moves[0]
                        routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                        routes[to_idx].insert(pos, cust)
                        current_max_local = new_max_val
                        improved = True
                    else:
                        # accept worsening with SA probability
                        chosen_move = random.choice(moves)
                        delta, new_max_val, cust, from_idx, to_idx, pos = chosen_move
                        if delta < 0:
                            prob_accept = math.exp(delta / temp)
                            if random.random() < prob_accept:
                                routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                                routes[to_idx].insert(pos, cust)
                                current_max_local = new_max_val
                                improved = True
            # Intra-route 2-opt with SA
            if not improved:
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 3:
                        continue
                    for i in range(1, len(route)-2):
                        for k in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                            new_len = route_length(new_route)
                            old_len = route_length(route)
                            delta_len = old_len - new_len  # positive if improvement
                            if delta_len > 0:
                                route[:] = new_route
                                improved = True
                                current_max_local = max(route_length(r) for r in routes)
                            else:
                                # SA acceptance for worsening
                                prob_accept = math.exp(delta_len / temp)  # delta_len negative
                                if random.random() < prob_accept:
                                    route[:] = new_route
                                    improved = True
                                    current_max_local = max(route_length(r) for r in routes)
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            if current_max_local < best_max:
                best_max = current_max_local
                best_routes = [r[:] for r in routes]
                report_best_vrp(routes)
            temp *= cooling_rate
            if temp < 1e-6:
                break
            if not improved:
                break

        # Diversified ruin-and-recreate
        total_customers = n - 1
        remove_ratio = 0.3 + random.random() * 0.2
        num_remove = max(1, int(total_customers * remove_ratio))
        all_customers = list(range(1, n))
        random.shuffle(all_customers)
        customers_to_remove = all_customers[:num_remove]
        for r_idx in range(truck_count):
            route = routes[r_idx]
            routes[r_idx] = [c for c in route if c not in customers_to_remove]
        unassigned = customers_to_remove
        # Reinsert using same probabilistic regret
        while unassigned:
            regrets = []
            for cust in unassigned:
                insert_options = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_len = route_length(route) + cost
                        other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                        new_max = max(new_len, *other_lens)
                        insert_options.append((new_max, cost, r_idx, pos))
                if not insert_options:
                    continue
                insert_options.sort(key=lambda x: (x[0], x[1]))
                best = insert_options[0]
                second = insert_options[1] if len(insert_options) > 1 else (best[0]+1e9, best[1]+1e9, -1, -1)
                regret = max(second[0] - best[0], 0)
                regrets.append((regret, cust, best))
            if not regrets:
                break
            total_regret = sum(r for r,_,_ in regrets) + 1e-9
            probs = [(r+1e-9)/total_regret for r,_,_ in regrets]
            rnd = random.random()
            cumulative = 0.0
            chosen = None
            for prob, (_, cust, best) in zip(probs, regrets):
                cumulative += prob
                if rnd <= cumulative:
                    chosen = (cust, best)
                    break
            if chosen is None:
                cust, best = max(regrets, key=lambda x: x[0])[1:]
            else:
                cust, best = chosen
            _, _, r_idx, pos = best
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)

        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

    return best_routes if best_routes else routes