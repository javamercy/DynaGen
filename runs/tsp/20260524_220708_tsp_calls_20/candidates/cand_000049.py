import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        np.random.default_rng(seed).shuffle(tour)
        return tour
    rng = np.random.default_rng(seed)

    def regret_construction(start):
        tour = [start]
        unvisited = set(range(n)) - {start}
        while unvisited:
            best_data = {}
            for city in unvisited:
                best_cost = np.inf
                second_best = np.inf
                best_pos = -1
                m = len(tour)
                for i in range(m):
                    prev = tour[i]
                    nxt = tour[(i+1) % m]
                    inc = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                    if inc < best_cost:
                        second_best = best_cost
                        best_cost = inc
                        best_pos = i+1
                    elif inc < second_best:
                        second_best = inc
                regret = second_best - best_cost if second_best != np.inf else best_cost
                best_data[city] = (regret, best_pos, best_cost)
            chosen = max(unvisited, key=lambda c: best_data[c][0])
            pos = best_data[chosen][1]
            tour.insert(pos, chosen)
            unvisited.remove(chosen)
        return np.array(tour, dtype=np.int64)

    start = rng.integers(n)
    tour = regret_construction(start)
    tour_list = tour.tolist()  # work with list for efficient insertion
    best_tour = tour.copy()
    best_dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)
    cur_dist = best_dist
    ops = 0
    improved = True
    no_improve_count = 0
    MAX_NO_IMPROVE = 5

    while ops < budget:
        if not improved:
            no_improve_count += 1
            if no_improve_count >= MAX_NO_IMPROVE:
                # Restart
                start = rng.integers(n)
                tour = regret_construction(start)
                tour_list = tour.tolist()
                cur_dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                no_improve_count = 0
                improved = True
                continue
            else:
                improved = True
                continue
        else:
            no_improve_count = 0
        improved = False
        # First-improvement relocate: try every (i, j) pair
        for i in range(n):
            if ops >= budget:
                break
            city = tour_list[i]
            # consider removal: neighbors are (i-1) and (i+1) mod n
            left = tour_list[(i-1)%n]
            right = tour_list[(i+1)%n]
            removal_delta = distance_matrix[left, city] + distance_matrix[city, right] - distance_matrix[left, right]
            for j in range(n):
                if ops >= budget:
                    break
                if j == i or j == (i-1)%n:
                    continue
                # insertion between j and j+1
                after_j = tour_list[(j+1)%n]
                insertion_delta = distance_matrix[tour_list[j], city] + distance_matrix[city, after_j] - distance_matrix[tour_list[j], after_j]
                delta = insertion_delta - removal_delta
                if delta < -1e-12:
                    # perform move
                    ops += 1
                    # remove city from i
                    if i < j:
                        moved = tour_list.pop(i)
                        tour_list.insert(j, moved)
                    else:
                        moved = tour_list.pop(i)
                        tour_list.insert(j+1, moved)
                    cur_dist += delta
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = np.array(tour_list, dtype=np.int64)
                        report_best_tour(best_tour)
                    improved = True
                    break
            if improved:
                break
        # if full pass without improvement, will set improved=False and restart logic will trigger next iteration
    return best_tour