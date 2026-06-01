import numpy as np
import math
from itertools import combinations

class VRPSolver:
    def __init__(self, dist, truck_count):
        self.dist = dist
        self.n = dist.shape[0]
        self.truck_count = truck_count
        self.routes = None
        self.route_dists = None
        self.max_dist = None

    def compute_route_dist(self, route):
        total = 0.0
        for i in range(len(route)-1):
            total += self.dist[route[i]][route[i+1]]
        return total

    def evaluate(self, routes):
        dists = [self.compute_route_dist(r) for r in routes]
        return max(dists), dists

    def build_initial(self):
        routes = [[0, 0] for _ in range(self.truck_count)]
        route_dists = [0.0 for _ in range(self.truck_count)]
        remaining = list(range(1, self.n))
        while remaining:
            c = remaining.pop(0)
            best_truck = None
            best_pos = None
            best_increase = float('inf')
            for t in range(self.truck_count):
                route = routes[t]
                cur_len = route_dists[t]
                for pos in range(1, len(route)):
                    i = route[pos-1]
                    j = route[pos]
                    increase = self.dist[i][c] + self.dist[c][j] - self.dist[i][j]
                    if increase < best_increase:
                        best_increase = increase
                        best_truck = t
                        best_pos = pos
            route = routes[best_truck]
            route.insert(best_pos, c)
            route_dists[best_truck] += best_increase
        self.routes = routes
        self.route_dists = route_dists
        self.max_dist = max(route_dists)
        from vrp_solver_interface import report_best_vrp
        report_best_vrp(self.routes)

    def try_relocate(self):
        improved = False
        for t_from in range(self.truck_count):
            route_from = self.routes[t_from]
            if len(route_from) <= 2:
                continue
            for idx in range(1, len(route_from)-1):
                c = route_from[idx]
                # remove c from route_from
                new_route_from = route_from[:idx] + route_from[idx+1:]
                new_dist_from = self.compute_route_dist(new_route_from)
                for t_to in range(self.truck_count):
                    if t_to == t_from:
                        continue
                    route_to = self.routes[t_to]
                    for pos in range(1, len(route_to)):
                        new_route_to = route_to[:pos] + [c] + route_to[pos:]
                        new_dist_to = self.compute_route_dist(new_route_to)
                        new_max = max(self.max_dist if self.max_dist >= new_dist_from and self.max_dist >= new_dist_to else 
                                       new_dist_from, new_dist_to, 
                                       max(d for i,d in enumerate(self.route_dists) if i not in (t_from, t_to)))
                        if new_max < self.max_dist:
                            self.routes[t_from] = new_route_from
                            self.routes[t_to] = new_route_to
                            self.route_dists[t_from] = new_dist_from
                            self.route_dists[t_to] = new_dist_to
                            self.max_dist = new_max
                            from vrp_solver_interface import report_best_vrp
                            report_best_vrp(self.routes)
                            improved = True
                            return True
        return False

    def try_swap(self):
        improved = False
        for t1, t2 in combinations(range(self.truck_count), 2):
            route1 = self.routes[t1]
            route2 = self.routes[t2]
            if len(route1) <= 2 or len(route2) <= 2:
                continue
            for i in range(1, len(route1)-1):
                c1 = route1[i]
                for j in range(1, len(route2)-1):
                    c2 = route2[j]
                    # swap c1 and c2
                    new_route1 = route1[:i] + [c2] + route1[i+1:]
                    new_route2 = route2[:j] + [c1] + route2[j+1:]
                    new_dist1 = self.compute_route_dist(new_route1)
                    new_dist2 = self.compute_route_dist(new_route2)
                    new_max = max(new_dist1, new_dist2, 
                                  max(d for k,d in enumerate(self.route_dists) if k not in (t1, t2)))
                    if new_max < self.max_dist:
                        self.routes[t1] = new_route1
                        self.routes[t2] = new_route2
                        self.route_dists[t1] = new_dist1
                        self.route_dists[t2] = new_dist2
                        self.max_dist = new_max
                        from vrp_solver_interface import report_best_vrp
                        report_best_vrp(self.routes)
                        improved = True
                        return True
        return False

    def try_intra_two_opt(self):
        improved = False
        for t in range(self.truck_count):
            route = self.routes[t]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = self.compute_route_dist(new_route)
                    if new_dist < self.route_dists[t]:
                        self.routes[t] = new_route
                        self.route_dists[t] = new_dist
                        self.max_dist = max(self.route_dists)
                        from vrp_solver_interface import report_best_vrp
                        report_best_vrp(self.routes)
                        improved = True
                        return True
        return False

    def try_inter_two_opt_star(self):
        improved = False
        for t1, t2 in combinations(range(self.truck_count), 2):
            route1 = self.routes[t1]
            route2 = self.routes[t2]
            if len(route1) < 2 or len(route2) < 2:
                continue
            for i in range(1, len(route1)-1):
                for j in range(1, len(route2)-1):
                    new_route1 = route1[:i] + route2[j:]
                    new_route2 = route2[:j] + route1[i:]
                    new_dist1 = self.compute_route_dist(new_route1)
                    new_dist2 = self.compute_route_dist(new_route2)
                    new_max = max(new_dist1, new_dist2, 
                                  max(d for k,d in enumerate(self.route_dists) if k not in (t1, t2)))
                    if new_max < self.max_dist:
                        self.routes[t1] = new_route1
                        self.routes[t2] = new_route2
                        self.route_dists[t1] = new_dist1
                        self.route_dists[t2] = new_dist2
                        self.max_dist = new_max
                        from vrp_solver_interface import report_best_vrp
                        report_best_vrp(self.routes)
                        improved = True
                        return True
        return False

    def improve(self):
        max_iter = 100 * self.n
        for _ in range(max_iter):
            improved = False
            improved = improved or self.try_relocate()
            if improved:
                continue
            improved = improved or self.try_swap()
            if improved:
                continue
            improved = improved or self.try_intra_two_opt()
            if improved:
                continue
            improved = improved or self.try_inter_two_opt_star()
            if not improved:
                break

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    solver = VRPSolver(distance_matrix, truck_count)
    solver.build_initial()
    solver.improve()
    return solver.routes