import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        low = func.bounds.lb
        high = func.bounds.ub
        dim = self.dim
        budget = self.budget
        x = np.random.uniform(low, high, size=dim)
        f = func(x)
        best_x = x.copy()
        best_f = f
        report_best(best_f, best_x)
        evals = 1
        if budget <= 1:
            return best_f, best_x
        step = (high - low) * 0.1
        step = np.maximum(step, 1e-6)
        stagnation = 0
        restart_step = (high - low) * 0.02
        while evals < budget:
            improved = False
            if stagnation > 2 * dim:
                x = np.clip(best_x + np.random.randn(dim) * restart_step, low, high)
                f = func(x)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    report_best(best_f, best_x)
                step = restart_step.copy()
                stagnation = 0
                continue
            for i in np.random.permutation(dim):
                if evals >= budget:
                    break
                if np.random.rand() < 0.05:
                    direction = np.random.randn(dim)
                    direction = direction / np.linalg.norm(direction)
                    step_scaled = step * direction
                    x_new = np.clip(x + step_scaled, low, high)
                    f_new = func(x_new)
                    evals += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                        report_best(best_f, best_x)
                        x = x_new.copy()
                        improved = True
                        while evals < budget:
                            x_pat = np.clip(x + step_scaled, low, high)
                            f_pat = func(x_pat)
                            evals += 1
                            if f_pat < best_f:
                                best_f = f_pat
                                best_x = x_pat.copy()
                                report_best(best_f, best_x)
                                x = x_pat.copy()
                            else:
                                break
                        break
                else:
                    # positive direction
                    x_new = x.copy()
                    x_new[i] += step[i]
                    x_new = np.clip(x_new, low, high)
                    f_new = func(x_new)
                    evals += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                        report_best(best_f, best_x)
                        x = x_new.copy()
                        improved = True
                        while evals < budget:
                            x_pat = x.copy()
                            x_pat[i] += step[i]
                            x_pat = np.clip(x_pat, low, high)
                            f_pat = func(x_pat)
                            evals += 1
                            if f_pat < best_f:
                                best_f = f_pat
                                best_x = x_pat.copy()
                                report_best(best_f, best_x)
                                x = x_pat.copy()
                            else:
                                break
                        break
                    # negative direction
                    x_new = x.copy()
                    x_new[i] -= step[i]
                    x_new = np.clip(x_new, low, high)
                    f_new = func(x_new)
                    evals += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                        report_best(best_f, best_x)
                        x = x_new.copy()
                        improved = True
                        while evals < budget:
                            x_pat = x.copy()
                            x_pat[i] -= step[i]
                            x_pat = np.clip(x_pat, low, high)
                            f_pat = func(x_pat)
                            evals += 1
                            if f_pat < best_f:
                                best_f = f_pat
                                best_x = x_pat.copy()
                                report_best(best_f, best_x)
                                x = x_pat.copy()
                            else:
                                break
                        break
            if improved:
                stagnation = 0
            else:
                step *= 0.5
                stagnation += 1
        return best_f, best_x