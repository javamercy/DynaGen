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
        # initial point
        cur_x = self.rng.uniform(lb, ub)
        cur_f = func(cur_x)
        best_x = cur_x.copy()
        best_f = cur_f
        report_best(best_f, best_x)
        evals = 1
        step = 0.2 * (ub - lb)
        min_step = 1e-3 * (ub - lb)
        cycles_no_improve = 0
        while evals < self.budget:
            improved_in_cycle = False
            for coord in range(dim):
                if evals >= self.budget:
                    break
                # diversification: random coordinate reset with low probability
                if self.rng.rand() < 0.2 / dim:
                    new_x = self.rng.uniform(lb, ub)
                    new_f = func(new_x)
                    evals += 1
                    if new_f < cur_f:
                        cur_f = new_f
                        cur_x = new_x.copy()
                        if new_f < best_f:
                            best_f = new_f
                            best_x = cur_x.copy()
                            report_best(best_f, best_x)
                        improved_in_cycle = True
                    continue
                # try positive direction
                candidate = cur_x.copy()
                candidate[coord] += step[coord]
                candidate = np.clip(candidate, lb, ub)
                f_cand = func(candidate)
                evals += 1
                if f_cand < cur_f:
                    cur_f = f_cand
                    cur_x = candidate.copy()
                    if f_cand < best_f:
                        best_f = f_cand
                        best_x = cur_x.copy()
                        report_best(best_f, best_x)
                    improved_in_cycle = True
                    step[coord] *= 1.2
                    continue
                # try negative direction
                if evals >= self.budget:
                    step[coord] *= 0.5
                    break
                candidate2 = cur_x.copy()
                candidate2[coord] -= step[coord]
                candidate2 = np.clip(candidate2, lb, ub)
                f_cand2 = func(candidate2)
                evals += 1
                if f_cand2 < cur_f:
                    cur_f = f_cand2
                    cur_x = candidate2.copy()
                    if f_cand2 < best_f:
                        best_f = f_cand2
                        best_x = cur_x.copy()
                        report_best(best_f, best_x)
                    improved_in_cycle = True
                    step[coord] *= 1.2
                else:
                    step[coord] *= 0.5
                step[coord] = max(step[coord], min_step[coord])
            if not improved_in_cycle and evals < self.budget:
                cycles_no_improve += 1
                step = 0.2 * (ub - lb)  # reset step sizes
                if cycles_no_improve % 2 == 0:  # local restart around best
                    perturbation = self.rng.uniform(-1, 1, size=dim) * step * 0.5
                    new_x = np.clip(best_x + perturbation, lb, ub)
                else:  # global restart
                    new_x = self.rng.uniform(lb, ub)
                new_f = func(new_x)
                evals += 1
                if new_f < best_f:
                    best_f = new_f
                    best_x = new_x.copy()
                    report_best(best_f, best_x)
                cur_x = new_x.copy()
                cur_f = new_f
            else:
                cycles_no_improve = 0
        return best_f, best_x