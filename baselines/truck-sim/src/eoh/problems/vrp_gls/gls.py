import time
import random

import numpy as np
from numba import jit
from itertools import pairwise

from copy import deepcopy
from collections import defaultdict

from src.eoh.problems.interface import truck_num_scaling


def route_dist(distance_mtx: np.ndarray, route: list[int]) -> float:
    """
    Calculate the cost of a single route, including return to depot.
    route: list of customer indices
    """
    # assert route is not None
    # assert len(route) > 0
    #
    # cost: float = 0
    # for c, n in pairwise(route):
    #     cost += distance_mtx[c, n]
    #
    # return cost
    return np.sum(distance_mtx[route[:-1], route[1:]])


class Solution:
    def __init__(self, routes):
        """
        routes: list of routes, each route is a list of customer indices (excluding depot)
        """
        self.routes = routes
        self.costs = [0 for _ in routes]
        # self.cost = 0  # Will be calculated based on routes

    def __deepcopy__(self, memodict={}):
        copy_obj = Solution([r[:] for r in self.routes])
        copy_obj.costs = self.costs[:]
        # copy_obj = copy(self)
        # copy_obj.routes = [r[:] for r in self.routes]
        # copy_obj.costs = self.costs[:]
        return copy_obj

    def calculate_cost(self, distance_mtx: np.ndarray):
        """
        Calculate the total cost based on the current routes and the distance matrix
        """
        self.costs = [route_dist(distance_mtx, route) for route in self.routes]

    def get_cost(self) -> float:
        return max(self.costs)


def initial_solution(size: int, distance_mtx: np.ndarray) -> Solution:
    """
    Create an initial solution using the greedy insertion heuristic
    """
    # start at depot
    routes = [[0] for _ in range(truck_num_scaling(size))]
    customer_indices = list(range(1, size))  # Customers are indexed from 1
    random.shuffle(customer_indices)  # Shuffle to introduce randomness

    for customer in customer_indices:
        best_route = min(routes, key=lambda r: route_dist(distance_mtx, r + [customer]))
        best_route.append(customer)

    # end at depot
    for route in routes:
        route.append(0)

    solution = Solution(routes)
    solution.calculate_cost(distance_mtx)
    return solution


class GuidedLocalSearch:
    def __init__(self,
                 eva,
                 size: int,
                 instance: np.ndarray,
                 distance_mtx: np.ndarray,
                 max_iterations: int = 1000,
                 running_time: float = 10):
        # , alpha = 1.0, penalty_multiplier = 10):
        """
        eva: the heuristic to evaluate
        size: problem size
        instance: the problem instance (list of coordinates)
        distance_mtx: distance matrix
        max_iterations: maximum number of iterations per run
        """
        # alpha: penalty update rate
        # penalty_multiplier: multiplier for adjusting penalties

        self.eva = eva
        self.size: int = size
        self.instance: np.ndarray = instance
        self.distance_mtx: np.ndarray = distance_mtx
        self.max_iterations: int = max_iterations
        self.running_time: float = running_time
        # self.alpha = alpha
        # self.penalty_multiplier = penalty_multiplier
        # self.features = defaultdict(int)
        self.edge_distance_guided: np.ndarray = np.zeros(np.shape(self.distance_mtx))
        self.edge_distance_gap: np.ndarray = np.zeros(np.shape(self.distance_mtx))
        self.edge_penalty: np.ndarray = np.zeros(np.shape(self.distance_mtx))

    def solution_cost(self, solution: Solution) -> float:
        """
        Calculate the cost of a single route, including return to depot, modified with the feature penalties
        """
        # costs: list[float] = []
        # for route in solution.routes:
        #     assert route is not None
        #     assert len(route) > 0
        #
        #     cost: float = 0
        #     for c, n in pairwise(route):
        #         cost += self.edge_distance_guided[c, n]
        #     costs.append(cost)

        return sum(
            np.sum(self.edge_distance_guided[route[:-1], route[1:]])
            for route in solution.routes)

    def solution_costs(self, solution: Solution) -> list[float]:
        """
        Calculate the cost of a single route, including return to depot, modified with the feature penalties
        """
        # costs: list[float] = []
        # for route in solution.routes:
        #     assert route is not None
        #     assert len(route) > 0
        #
        #     cost: float = 0
        #     for c, n in pairwise(route):
        #         cost += self.edge_distance_guided[c, n]
        #     costs.append(cost)

        return [np.sum(self.edge_distance_guided[route[:-1], route[1:]])
                for route in solution.routes]

    def update_penalties(self, solution: Solution):
        """
        Update penalties based on ??? (feature frequencies?)
        """
        for route in solution.routes:
            assert route is not None
            assert len(route) > 0

        self.edge_distance_guided = np.asmatrix(
            self.eva.update_edge_distance(self.distance_mtx.copy(),
                                          [r[:] for r in solution.routes],
                                          self.edge_penalty.copy()))

        self.edge_distance_gap = self.edge_distance_guided - self.distance_mtx

        max_indices = np.argmax(self.edge_distance_gap, axis=None)
        rows, cols = np.unravel_index(max_indices, self.edge_distance_gap.shape)

        self.edge_penalty[rows, cols] += 1
        self.edge_penalty[cols, rows] += 1

        # # ???
        # self.edge_distance_gap[rows, cols] = 0
        # self.edge_distance_gap[cols, rows] = 0

    def generate_neighbors(self, solution):
        """
        Generate neighbors by applying 2-opt or relocating a customer to another route.
        """
        # cur_costs = self.solution_costs(solution)
        cur_cost = self.solution_cost(solution)
        neighbors = []
        neighbor_costs = []
        best_delta: float = 0

        # 2-opt within the same route
        for route_idx, route in enumerate(solution.routes):
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    # previously: [0 ... i-1] [i ... j] [j+1 ... n]
                    # new result: [0 ... i-1] [j ... i] [j+1 ... n]

                    i_l = route[i-1]
                    i_r = route[i]
                    j_l = route[j]
                    j_r = route[j+1]

                    # delta = \
                    #     (- self.edge_distance_guided[i_l, i_r]
                    #      - self.edge_distance_guided[j_l, j_r]
                    #      + self.edge_distance_guided[i_l, j_l]
                    #      + self.edge_distance_guided[i_r, j_r])
                    vals = np.ravel(self.edge_distance_guided[[i_l, j_l, i_l, i_r],
                                                              [i_r, j_r, j_l, j_r]])
                    delta = \
                        (- vals[0]
                         - vals[1]
                         + vals[2]
                         + vals[3])

                    if delta < best_delta:
                        best_delta = delta
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]

                        neighbor = deepcopy(solution)
                        neighbor.routes[route_idx] = new_route

                        # neighbor.calculate_cost(self.distance_mtx)
                        # delta_cost = \
                        #     (- self.distance_mtx[i_l, i_r]
                        #      - self.distance_mtx[j_l, j_r]
                        #      + self.distance_mtx[i_l, j_l]
                        #      + self.distance_mtx[i_r, j_r])
                        vals = np.ravel(self.distance_mtx[[i_l, j_l, i_l, i_r],
                                                          [i_r, j_r, j_l, j_r]])
                        delta_cost = \
                            (- vals[0]
                             - vals[1]
                             + vals[2]
                             + vals[3])
                        neighbor.costs[route_idx] += delta_cost

                        neighbors.append(neighbor)
                        neighbor_costs.append(cur_cost + delta_cost)
                        # if cur_cost + delta != self.solution_cost(neighbor):
                        #     print(f"2-opt: {cur_cost + delta} vs {self.solution_cost(neighbor)}")

        # Relocate a node from one route to another
        for from_route_idx, from_route in enumerate(solution.routes):
            for to_route_idx, to_route in enumerate(solution.routes):
                if to_route_idx == from_route_idx:
                    continue

                for from_node_idx, from_node in enumerate(from_route):
                    if from_node == 0:
                        continue

                    # start: [0 ... f-1] [f] [f+1 ... n]
                    #        [0 ... l] [r ... n]
                    # end: [0 ... f-1] [f+1 ... n]
                    #      [0 ... l] [f] [r ... n]
                    f_l = from_route[from_node_idx-1]
                    f = from_node
                    f_r = from_route[from_node_idx+1]

                    # too expensive to check for all locations that the node can go to
                    to_node_idx = random.randint(0, len(to_route)-2)
                    t_l = to_route[to_node_idx]
                    t_r = to_route[to_node_idx+1]

                    # delta_cost = \
                    #     (- self.edge_distance_guided[f_l, f]
                    #      - self.edge_distance_guided[f, f_r]
                    #      + self.edge_distance_guided[f_l, f_r]
                    #      - self.edge_distance_guided[t_l, t_r]
                    #      + self.edge_distance_guided[t_l, f]
                    #      + self.edge_distance_guided[f, t_r])
                    vals = np.ravel(self.edge_distance_guided[[f_l, f, f_l, t_l, t_l, f],
                                                              [f, f_r, f_r, t_r, f, t_r]])
                    delta = \
                        (- vals[0]
                         - vals[1]
                         + vals[2]
                         - vals[3]
                         + vals[4]
                         + vals[5])

                    # if delta_cost < 0:
                    if delta < best_delta:
                        best_delta = delta

                        new_from_route = from_route[:from_node_idx] + from_route[from_node_idx+1:]
                        new_to_route = to_route[:to_node_idx+1] + [from_node] + to_route[to_node_idx+1:]

                        neighbor = deepcopy(solution)
                        neighbor.routes[from_route_idx] = new_from_route
                        neighbor.routes[to_route_idx] = new_to_route

                        # neighbor.calculate_cost(self.distance_mtx)
                        # delta_cost_from = \
                        #     (- self.distance_mtx[f_l, f]
                        #      - self.distance_mtx[f, f_r]
                        #      + self.distance_mtx[f_l, f_r])
                        # delta_cost_to = \
                        #     (- self.distance_mtx[t_l, t_r]
                        #      + self.distance_mtx[t_l, f]
                        #      + self.distance_mtx[f, t_r])
                        vals = np.ravel(self.distance_mtx[[f_l, f, f_l, t_l, t_l, f],
                                                          [f, f_r, f_r, t_r, f, t_r]])
                        delta_cost_from = \
                            (- vals[0]
                             - vals[1]
                             + vals[2])
                        delta_cost_to = \
                            (- vals[3]
                             + vals[4]
                             + vals[5])

                        neighbor.costs[from_route_idx] += delta_cost_from
                        neighbor.costs[to_route_idx] += delta_cost_to

                        neighbors.append(neighbor)
                        neighbor_costs.append(cur_cost + delta)
                        # if cur_cost + delta != self.solution_cost(neighbor):
                        #     print(f"relocate: {cur_cost + delta} vs {self.solution_cost(neighbor)}")

                        # TODO: not performing that well (metric is not taking into account the span)
                        #   instead we should probably only operate on the longest route
                        #   or maybe have two modes (minimize length, minimize span)

        # print(best_delta)
        return neighbors, neighbor_costs

    def search(self) -> Solution:
        # create the initial solution
        end_time = time.time() + self.running_time

        # - current solution is the current state of our solution (best value using guided matrix)
        # - best solution is the best actual solution (best value using distance matrix)
        curr_solution: Solution = initial_solution(self.size, self.distance_mtx)
        best_solution: Solution = deepcopy(curr_solution)

        # iterations of GLS
        for iteration in range(1, self.max_iterations + 1):
            if time.time() > end_time:
                break

            # local search start #####
            # cost: float = self.solution_cost(curr_solution)
            #
            # Generate neighbors
            # neighbors: list[Solution] = self.generate_neighbors(curr_solution)
            # if not neighbors:
            #     break
            #
            # # Select the best neighbor based on modified cost
            # # neighbors.sort(key=lambda s: self.solution_cost(s))
            # # best_neighbor = neighbors[0]
            # # best_neighbor = min(neighbors, key=lambda s: self.solution_cost(s))
            # neighbor_costs = [self.solution_cost(s) for s in neighbors]
            # best_neighbor = neighbors[np.argmin(neighbor_costs)]
            # best_neighbor_cost = self.solution_cost(best_neighbor)
            # # local search end #####

            neighbors, neighbor_costs = self.generate_neighbors(curr_solution)
            if len(neighbors) == 0:
                # Update penalties to encourage exploration
                # print("No neighbors found")
                self.update_penalties(curr_solution)

            else:
                # Every neighbor returned will have a better score than curr_solution
                curr_solution = neighbors[np.argmin(neighbor_costs)]

                # # don't need??? (need to update how cost updates)
                # curr_solution.calculate_cost(self.distance_mtx)
                # best_solution.calculate_cost(self.distance_mtx)
                # prev_cur_cost = curr_solution.get_cost()
                # curr_solution.calculate_cost(self.distance_mtx)
                # if prev_cur_cost != curr_solution.get_cost():
                #     print(f"Cost: {prev_cur_cost} vs {curr_solution.get_cost()}")

                # Update best if actual cost is better
                if curr_solution.get_cost() < best_solution.get_cost():
                    best_solution = deepcopy(curr_solution)

            # best_neighbor = neighbors[np.argmin(neighbor_costs)]
            # best_neighbor_cost = min(neighbor_costs)
            #
            # # If the best neighbor is better under modified costs, accept it
            # if best_neighbor_cost < cost:
            #     curr_solution = best_neighbor
            #
            #     # TODO: ???
            #     curr_solution.calculate_cost(self.distance_mtx)
            #     best_solution.calculate_cost(self.distance_mtx)
            #
            #     # Update best if actual cost is better
            #     if curr_solution.get_cost() < best_solution.get_cost():
            #         best_solution = deepcopy(curr_solution)
            #
            # else:
            #     # Update penalties to encourage exploration
            #     self.update_penalties(curr_solution)

            # # Optional: print progress
            # if iteration % 100 == 0 or iteration == 1:
            #     # print(best_solution.routes)
            #     # print(best_solution.costs)
            #     # print(curr_solution.routes)
            #     # print(curr_solution.costs)
            #     # if len(neighbor_costs) > 0:
            #     #     print(min(neighbor_costs))
            #     # print(f"Iteration {iteration}, Best Cost: {best_solution.cost:.2f}")
            #     print("Iteration {}, Best Cost: {}".format(iteration, best_solution.get_cost()))
            #     print()

        return best_solution

"""
The role of constraints on features is to guide local search on the basis of information not being
incorporated in the cost function because it is either ambiguous or unknown at the time.
- given some application we often have some idea of what makes a good solution
- cost function tries to format that understanding in a mathematical way
- however it is not easy to include all information about the problem into the cost function

e.g. for example in TSP long edges are undesirable, but we cannot exclude all long edges from the beginning
- extra information for what we can keep and what we shouldn't will become known once we search
- if a solution is visited we can then exclude it and possibility other solution (e.g. with higher cost) from being searched in the future (branch and bound uses this)

with GLS information is converted into constraints using penalty terms to confine local search to promising solutions
- exploits two pieces of information which are the cost of features and also the local minima already visited
- each time local search is trapped in a local minimum, GLS can increment the penalty parameter of one or more of the solution features
- when penalty parameter of a feature is incremented (features is penalized) then we avoid this feature by local search

Initially, all penalty parameters are set to 0 (no features are constrained)
- call local search to find a local minimum of the original cost function
- the algorithm then loops modifying the cost function on one or more of the local minimum's features then performing local search again
    - modification action is incrementing by one the penalty parameter of one or more the local minimum's features
- information is gradually inserted in the augmented cost function by selecting which penalty parameter to increment

- every feature fi is given a constant cost ci
- we care about the difference between the previous penalty and the current penalty

- greedy local search is good but can be time-consuming for large scale problems
- a particular local minimum solution s* with n features fi is denoted Ii(s*) = n
- the vector of the indicator function

procedure GLS(S, g, λ, [I1, ...,IN], [c1,...,cN], M)
begin
    k ← 0;
    s0 ← arbitrary solution in S;
    for i ← 1 until M do
        pi ← 0;
    h ← g + λ * ∑ (pi*Ii) ;

    while StoppingCriterion do
    begin
        sk+1 ← LocalSearch(sk, h);
        for i ← 1 until M do
            utili ← Ii(sk+1) * ci / (1+pi);
        for each i such that utili is maximum do
            pi ← pi + 1;
        k ← k+1;
    end

    s* ← best solution found with respect to cost function g;
    return s*;
end


what things should I penalize?
- that is decided by the heuristic!
- tsp_gls asks the heuristic what weights to assign to change the edge_weights to
- then finds the edges that got the greatest amount of weight added to them and increases the edge_penalty for them
- uses the
"""

