import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    best_tour = None
    best_dist = float('inf')
    restarts = min(15, n)

    for _ in range(restarts):
        # Regret insertion with random triangle
        start = random.sample(range(n), 3)
        tour = list(start)
        remaining = set(range(n)) - set(start)
        while remaining:
            best_city = -1
            best_regret = -1
            best_pos = -1
            for city in remaining:
                deltas = []
                for pos in range(len(tour)):
                    before = tour[pos-1]
                    after = tour[pos] if pos < len(tour) else tour[0]
                    delta = distance_matrix[before, city] + distance_matrix[city, after] - distance_matrix[before, after]
                    deltas.append((delta, pos))
                deltas.sort(key=lambda x: x[0])
                first = deltas[0][0]
                second = deltas[1][0] if len(deltas) > 1 else first
                regret = second - first
                if regret > best_regret or (regret == best_regret and city < best_city):
                    best_regret = regret
                    best_city = city
                    best_pos = deltas[0][1]
            tour.insert(best_pos, best_city)
            remaining.remove(best_city)

        tour = np.array(tour, dtype=int)
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour[i], tour[(i+1)%n]]
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)

        # VND and ILS
        for _ in range(5):  # ILS iterations per restart
            improved = True
            while improved:
                improved = False
                # 2-opt
                for i in range(n-1):
                    for j in range(i+2, n):
                        a = tour[i]
                        b = tour[(i+1)%n]
                        c = tour[j]
                        d = tour[(j+1)%n]
                        delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                        if delta < -1e-10:
                            tour[i+1:j+1] = tour[j:i:-1]
                            dist += delta
                            if dist < best_dist - 1e-10:
                                best_dist = dist
                                best_tour = tour.copy()
                                report_best_tour(best_tour)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue
                # Or-opt
                for L in [1, 2, 3]:
                    if L >= n:
                        continue
                    for i in range(n):
                        if i + L <= n:
                            seg = list(tour[i:i+L])
                            new_tour_list = list(tour[:i]) + list(tour[i+L:])
                        else:
                            seg = list(tour[i:]) + list(tour[:i+L-n])
                            new_tour_list = list(tour[i+L-n:i])
                        m = len(new_tour_list)
                        if m == 0:
                            continue
                        for pos in range(m + 1):
                            cand = new_tour_list[:pos] + seg + new_tour_list[pos:]
                            cand_dist = 0.0
                            for k in range(n):
                                cand_dist += distance_matrix[cand[k], cand[(k+1)%n]]
                            if cand_dist < dist - 1e-10:
                                tour = np.array(cand, dtype=int)
                                dist = cand_dist
                                if dist < best_dist - 1e-10:
                                    best_dist = dist
                                    best_tour = tour.copy()
                                    report_best_tour(best_tour)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue
                # Node swap
                for i in range(n):
                    for j in range(i+1, n):
                        new_tour = tour.copy()
                        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
                        new_dist = 0.0
                        for k in range(n):
                            new_dist += distance_matrix[new_tour[k], new_tour[(k+1)%n]]
                        if new_dist < dist - 1e-10:
                            tour = new_tour
                            dist = new_dist
                            if dist < best_dist - 1e-10:
                                best_dist = dist
                                best_tour = tour.copy()
                                report_best_tour(best_tour)
                            improved = True
                            break
                    if improved:
                        break
            # Perturbation: double-bridge move
            if random.random() < 0.2:  # perturbation probability
                # Choose 4 distinct indices
                idx = sorted(random.sample(range(n), 4))
                a, b, c, d = idx
                # Break into segments: [0:a], [a:b], [b:c], [c:d], [d:n]
                # Reorder: [0:a], [c:d], [b:c], [a:b], [d:n]
                new_tour = np.concatenate([tour[:a], tour[c:d], tour[b:c], tour[a:b], tour[d:]])
                # Ensure correct wrap-around? No, this works for a<b<c<d.
                dist = 0.0
                for k in range(n):
                    dist += distance_matrix[new_tour[k], new_tour[(k+1)%n]]
                tour = new_tour
            else:
                break  # stop ILS if no perturbation
    return best_tour