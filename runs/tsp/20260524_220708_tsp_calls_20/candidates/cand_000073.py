import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        np.random.default_rng(seed).shuffle(tour)
        return tour
    rng = np.random.default_rng(seed)
    dist = distance_matrix
    
    def nn_tour(start):
        tour = [start]
        unvisited = set(range(n)) - {start}
        while unvisited:
            last = tour[-1]
            next_city = min(unvisited, key=lambda c: dist[last, c])
            tour.append(next_city)
            unvisited.remove(next_city)
        return np.array(tour, dtype=np.int64)
    
    def tour_dist(t):
        return sum(dist[t[i], t[(i+1)%n]] for i in range(n))
    
    # Initial tour
    start = rng.integers(n)
    cur_tour = nn_tour(start)
    cur_dist = tour_dist(cur_tour)
    best_tour = cur_tour.copy()
    best_dist = cur_dist
    report_best_tour(best_tour)
    
    ops = 0
    while ops < budget:
        improved = False
        for i in range(n-1):
            if ops >= budget:
                break
            for j in range(i+2, n):
                if ops >= budget:
                    break
                ops += 1
                a = cur_tour[i]
                b = cur_tour[(i+1)%n]
                c = cur_tour[j]
                d = cur_tour[(j+1)%n]
                old = dist[a,b] + dist[c,d]
                new = dist[a,c] + dist[b,d]
                if new < old - 1e-12:
                    cur_tour[i+1:j+1] = cur_tour[i+1:j+1][::-1]
                    cur_dist += new - old
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = cur_tour.copy()
                        report_best_tour(best_tour)
                    improved = True
                    break
            if improved or ops >= budget:
                break
        if ops >= budget:
            break
        if not improved:
            # Restart with a new random start
            if ops >= budget:
                break
            start = rng.integers(n)
            cur_tour = nn_tour(start)
            cur_dist = tour_dist(cur_tour)
            if cur_dist < best_dist:
                best_dist = cur_dist
                best_tour = cur_tour.copy()
                report_best_tour(best_tour)
    return best_tour