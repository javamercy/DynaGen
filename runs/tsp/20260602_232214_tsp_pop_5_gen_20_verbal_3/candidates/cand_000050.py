import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 3:
        return np.arange(n)

    def tour_dist(tour):
        total = 0.0
        for i in range(n):
            total += distance_matrix[tour[i]][tour[(i+1)%n]]
        return total

    def two_opt(tour, max_passes=10):
        best = tour.copy()
        best_dist = tour_dist(best)
        for _ in range(max_passes):
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    a = best[i]
                    b = best[(i+1)%n]
                    c = best[j]
                    d = best[(j+1)%n]
                    old = distance_matrix[a][b] + distance_matrix[c][d]
                    new = distance_matrix[a][c] + distance_matrix[b][d]
                    if new < old:
                        new_tour = np.concatenate([best[:i+1], best[i+1:j+1][::-1], best[j+1:]])
                        new_dist = tour_dist(new_tour)
                        if new_dist < best_dist:
                            best = new_tour
                            best_dist = new_dist
                            improved = True
            if not improved:
                break
        return best, best_dist

    def regret_insertion():
        tour = [0]
        first = np.argmin(distance_matrix[0][1:]) + 1
        tour.append(first)
        unvisited = set(range(n)) - set(tour)
        while unvisited:
            best_insert = {}
            second_best = {}
            for city in unvisited:
                best = float('inf')
                sec = float('inf')
                best_idx = None
                for pos in range(len(tour)):
                    i = tour[pos]
                    j = tour[(pos+1)%len(tour)]
                    cost = distance_matrix[i][city] + distance_matrix[city][j] - distance_matrix[i][j]
                    if cost < best:
                        sec = best
                        best = cost
                        best_idx = pos
                    elif cost < sec:
                        sec = cost
                best_insert[city] = (best_idx, best)
                second_best[city] = sec if sec != float('inf') else best
            regret = {c: second_best[c] - best_insert[c][1] for c in unvisited}
            chosen = max(unvisited, key=lambda c: (regret[c], -best_insert[c][1]))
            idx, _ = best_insert[chosen]
            tour.insert(idx+1, chosen)
            unvisited.remove(chosen)
        return np.array(tour)

    def random_insertion():
        tour = [0]
        unvisited = list(range(1, n))
        np.random.shuffle(unvisited)
        for city in unvisited:
            best_pos = 0
            best_cost = float('inf')
            for pos in range(len(tour)):
                i = tour[pos]
                j = tour[(pos+1)%len(tour)]
                cost = distance_matrix[i][city] + distance_matrix[city][j] - distance_matrix[i][j]
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
            tour.insert(best_pos+1, city)
        return np.array(tour)

    best_tour = regret_insertion()
    best_dist = tour_dist(best_tour)
    report_best_tour(best_tour)
    best_tour, best_dist = two_opt(best_tour)
    report_best_tour(best_tour)

    for _ in range(10):
        tour = random_insertion()
        tour, dist = two_opt(tour)
        if dist < best_dist:
            best_tour = tour
            best_dist = dist
            report_best_tour(best_tour)

    return best_tour