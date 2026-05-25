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
        best_f = np.inf
        best_x = None
        evals = 0
        step = 0.1 * (ub - lb)
        while evals < self.budget:
            x = lb + (ub - lb) * self.rng.rand(dim)
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
            local_step = step.copy()
            no_improve_cycles = 0
            max_no_improve = 10 + dim
            while evals < self.budget:
                improved = False
                for i in range(dim):
                    if evals >= self.budget:
                        break
                    candidate = x.copy()
                    candidate[i] = x[i] + local_step[i]
                    candidate[i] = np.clip(candidate[i], lb[i], ub[i])
                    f_candidate = func(candidate)
                    evals += 1
                    if f_candidate < f:
                        f = f_candidate
                        x = candidate.copy()
                        local_step[i] *= 1.2
                        improved = True
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                            report_best(best_f, best_x)
                        break
                    candidate = x.copy()
                    candidate[i] = x[i] - local_step[i]
                    candidate[i] = np.clip(candidate[i], lb[i], ub[i])
                    f_candidate = func(candidate)
                    evals += 1
                    if f_candidate < f:
                        f = f_candidate
                        x = candidate.copy()
                        local_step[i] *= 1.2
                        improved = True
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                            report_best(best_f, best_x)
                        break
                    else:
                        local_step[i] *= 0.5
                        if local_step[i] < 1e-15:
                            local_step[i] = 1e-15
                if not improved:
                    no_improve_cycles += 1
                    if no_improve_cycles >= max_no_improve:
                        break
                else:
                    no_improve_cycles = 0
        return best_f, best_x