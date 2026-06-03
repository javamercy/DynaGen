import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    best_dist = np.inf
    best_tour = None
    num_restarts = min(10, max(3, n // 10))

    def tour_length(tour):
        return sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))

    def two_opt(tour, dist):
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    if j - i == 1:
                        continue
                    a, b = tour[i], tour[(i+1)%n]
                    c, d = tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
                        dist += delta
        return tour, dist

    def regret_construction():
        start = np.random.randint(n)
        end = np.argmax(distance_matrix[start])
        tour_list = [start, end]
        unvisited = set(range(n))
        unvisited.difference_update(tour_list)
        while unvisited:
            best_city = None
            best_regret = -np.inf
            best_pos = None
            for city in unvisited:
                costs = []
                for i in range(len(tour_list)):
                    j = (i+1) % len(tour_list)
                    cost = distance_matrix[tour_list[i], city] + distance_matrix[city, tour_list[j]] - distance_matrix[tour_list[i], tour_list[j]]
                    costs.append(cost)
                sorted_costs = sorted(costs)
                if len(sorted_costs) > 1:
                    regret = sorted_costs[1] - sorted_costs[0]
                else:
                    regret = sorted_costs[0]
                if regret > best_regret:
                    best_regret = regret
                    best_city = city
                    best_pos = np.argmin(costs)
            tour_list.insert(best_pos + 1, best_city)
            unvisited.remove(best_city)
        return np.array(tour_list, dtype=int)

    for _ in range(num_restarts):
        tour = regret_construction()
        dist = tour_length(tour)
        tour, dist = two_opt(tour, dist)
        if dist < best_dist:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        for _ in range(max(5, n // 4)):
            i, j = np.random.choice(n, 2, replace=False)
            new_tour = tour.copy()
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            new_dist = tour_length(new_tour)
            new_tour, new_dist = two_opt(new_tour, new_dist)
            if new_dist < best_dist:
                best_dist = new_dist
                best_tour = new_tour.copy()
                report_best_tour(best_tour)
                tour = new_tour
                dist = new_dist
            elif new_dist < dist * 1.01:
                tour = new_tour
                dist = new_dist
    return best_tour