import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n == 1:
        tour = np.array([0])
        report_best_tour(tour)
        return tour
    if n == 2:
        tour = np.array([0, 1])
        report_best_tour(tour)
        return tour

    def compute_cost(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))

    best_tour = None
    best_cost = float('inf')
    num_rounds = 10

    for _ in range(num_rounds):
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n)) - {start}
        while unvisited:
            far_node = max(unvisited, key=lambda v: min(distance_matrix[v, t] for t in tour))
            best_inc = np.inf
            best_pos = 0
            for i in range(len(tour)):
                prev = tour[i]
                nxt = tour[(i+1) % len(tour)]
                inc = distance_matrix[prev][far_node] + distance_matrix[far_node][nxt] - distance_matrix[prev][nxt]
                if inc < best_inc:
                    best_inc = inc
                    best_pos = i+1
            tour.insert(best_pos, far_node)
            unvisited.remove(far_node)
        tour_arr = np.array(tour, dtype=np.int32)
        cost = compute_cost(tour_arr)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour_arr.copy()
            report_best_tour(best_tour)

        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    a, b = tour_arr[i], tour_arr[(i+1)%n]
                    c, d = tour_arr[j], tour_arr[(j+1)%n]
                    if distance_matrix[a][c] + distance_matrix[b][d] < distance_matrix[a][b] + distance_matrix[c][d]:
                        tour_arr[i+1:j+1] = tour_arr[i+1:j+1][::-1]
                        cost = compute_cost(tour_arr)
                        if cost < best_cost:
                            best_cost = cost
                            best_tour = tour_arr.copy()
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
    return best_tour