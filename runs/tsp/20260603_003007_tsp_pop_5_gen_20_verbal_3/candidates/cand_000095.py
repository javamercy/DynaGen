import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    
    def tsp_dist(tour):
        return sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    
    def nearest_neighbor(start):
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        cur = start
        while unvisited:
            next_city = min(unvisited, key=lambda x: distance_matrix[cur, x])
            tour.append(next_city)
            unvisited.remove(next_city)
            cur = next_city
        return np.array(tour)
    
    def two_opt(tour):
        tour = tour.copy()
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    if j - i == 1:
                        continue
                    a, b = tour[i], tour[i+1]
                    c, d = tour[j], tour[(j+1)%n]
                    if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
        return tour
    
    def double_bridge(tour):
        n = len(tour)
        cuts = sorted(np.random.choice(range(1, n), 3, replace=False))
        a, b, c = cuts
        seg1 = tour[:a]
        seg2 = tour[a:b]
        seg3 = tour[b:c]
        seg4 = tour[c:]
        new_tour = np.concatenate([seg1, seg3, seg2, seg4])
        return new_tour
    
    best_tour = nearest_neighbor(0)
    best_dist = tsp_dist(best_tour)
    report_best_tour(best_tour)
    
    max_no_improve = 50
    no_improve = 0
    for iteration in range(200):
        new_tour = double_bridge(best_tour)
        new_tour = two_opt(new_tour)
        new_dist = tsp_dist(new_tour)
        if new_dist < best_dist:
            best_dist = new_dist
            best_tour = new_tour
            report_best_tour(best_tour)
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= max_no_improve:
            start = np.random.randint(n)
            best_tour = nearest_neighbor(start)
            best_dist = tsp_dist(best_tour)
            report_best_tour(best_tour)
            no_improve = 0
    return best_tour