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
        rng = self.rng
        # initial point
        cur_x = rng.uniform(lb, ub)
        cur_f = func(cur_x)
        best_x = cur_x.copy()
        best_f = cur_f
        report_best(best_f, best_x)
        evals = 1
        step = 0.2 * (ub - lb)
        min_step = 1e-3 * (ub - lb)
        while evals < self.budget:
            improved_in_cycle = False
            for coord in range(dim):
                if evals >= self.budget:
                    break
                # try positive direction
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
                    step[coord] *= 1.2
                    continue
                # try negative direction
                if evals >= self.budget:
                    step[coord] *= 0.5
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
                    step[coord] *= 1.2
                else:
                    step[coord] *= 0.5
                step[coord] = max(step[coord], min_step[coord])
            if evals >= self.budget:
                break
            if not improved_in_cycle:
                # diversification: try random perturbation around current point
                if evals < self.budget:
                    trial = cur_x + step * rng.randn(dim)
                    trial = np.clip(trial, lb, ub)
                    f_trial = func(trial)
                    evals += 1
                    if f_trial < cur_f:
                        cur_f = f_trial
                        cur_x = trial.copy()
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = cur_x.copy()
                            report_best(best_f, best_x)
                        improved_in_cycle = True
                        step = 0.2 * (ub - lb)  # reset step to initial
                        continue  # go back to coordinate loop (since we improved)
                    # else, full restart around best point
                    if evals < self.budget:
                        cur_x = best_x + 0.1 * (ub - lb) * rng.randn(dim)
                        cur_x = np.clip(cur_x, lb, ub)
                        cur_f = func(cur_x)
                        evals += 1
                        if cur_f < best_f:
                            best_f = cur_f
                            best_x = cur_x.copy()
                            report_best(best_f, best_x)
                        step = 0.2 * (ub - lb)  # reset step sizes
        return best_f, best_x