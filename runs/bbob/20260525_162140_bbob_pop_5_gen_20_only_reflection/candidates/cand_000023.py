import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        cur_x = self.rng.uniform(lb, ub)
        cur_f = func(cur_x)
        best_x = cur_x.copy()
        best_f = cur_f
        report_best(best_f, best_x)
        evals = 1
        step = 0.2 * (ub - lb)
        min_step = 1e-3 * (ub - lb)
        p_reset = 0.2
        while evals < self.budget:
            improved_in_cycle = False
            if self.rng.uniform() < p_reset and evals < self.budget:
                coord = self.rng.randint(dim)
                candidate = cur_x.copy()
                candidate[coord] = self.rng.uniform(lb[coord], ub[coord])
                f_candidate = func(candidate)
                evals += 1
                if f_candidate < cur_f:
                    cur_f = f_candidate
                    cur_x = candidate.copy()
                    if f_candidate < best_f:
                        best_f = f_candidate
                        best_x = cur_x.copy()
                        report_best(best_f, best_x)
                    improved_in_cycle = True
            for coord in range(dim):
                if evals >= self.budget:
                    break
                candidate = cur_x.copy()
                candidate[coord] += step[coord]
                candidate = np.clip(candidate, lb, ub)
                f_candidate = func(candidate)
                evals += 1
                if f_candidate < cur_f:
                    cur_f = f_candidate
                    cur_x = candidate.copy()
                    if f_candidate < best_f:
                        best_f = f_candidate
                        best_x = cur_x.copy()
                        report_best(best_f, best_x)
                    improved_in_cycle = True
                    step[coord] = min(step[coord] * 1.2, 0.5 * (ub[coord] - lb[coord]))
                    continue
                if evals >= self.budget:
                    step[coord] = max(step[coord] * 0.5, min_step[coord])
                    break
                candidate2 = cur_x.copy()
                candidate2[coord] -= step[coord]
                candidate2 = np.clip(candidate2, lb, ub)
                f_candidate2 = func(candidate2)
                evals += 1
                if f_candidate2 < cur_f:
                    cur_f = f_candidate2
                    cur_x = candidate2.copy()
                    if f_candidate2 < best_f:
                        best_f = f_candidate2
                        best_x = cur_x.copy()
                        report_best(best_f, best_x)
                    improved_in_cycle = True
                    step[coord] = min(step[coord] * 1.2, 0.5 * (ub[coord] - lb[coord]))
                else:
                    step[coord] = max(step[coord] * 0.5, min_step[coord])
            if not improved_in_cycle and evals < self.budget:
                perturbation = self.rng.standard_cauchy(dim) * step
                candidate = best_x + perturbation
                candidate = np.clip(candidate, lb, ub)
                f_candidate = func(candidate)
                evals += 1
                if f_candidate < best_f:
                    best_f = f_candidate
                    best_x = candidate.copy()
                    report_best(best_f, best_x)
                cur_x = candidate.copy()
                cur_f = f_candidate
        return best_f, best_x