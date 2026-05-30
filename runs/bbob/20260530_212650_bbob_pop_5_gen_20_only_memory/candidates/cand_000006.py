import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_x = None
        self.best_value = np.inf

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initial point
        x0 = lb + self.rng.rand(self.dim) * (ub - lb)
        val0 = func(x0)
        self.best_x = x0.copy()
        self.best_value = val0
        report_best(self.best_value, self.best_x)
        evals = 1
        step_sizes = (ub - lb) * 0.1
        iteration = 0
        while evals < self.budget:
            # alternate between coordinate and pattern search
            if iteration % 2 == 0:
                # coordinate search: pick a dimension in round-robin
                d = (iteration // 2) % self.dim
                step = step_sizes[d]
                # positive direction
                x_candidate = self.best_x.copy()
                x_candidate[d] = np.clip(self.best_x[d] + step, lb[d], ub[d])
                val_candidate = func(x_candidate)
                evals += 1
                improved = False
                if val_candidate < self.best_value:
                    self.best_x = x_candidate
                    self.best_value = val_candidate
                    report_best(self.best_value, self.best_x)
                    step_sizes[d] *= 1.5
                    improved = True
                    # line search extension
                    while evals < self.budget:
                        new_x = self.best_x.copy()
                        new_x[d] = np.clip(self.best_x[d] + step_sizes[d], lb[d], ub[d])
                        new_val = func(new_x)
                        evals += 1
                        if new_val < self.best_value:
                            self.best_x = new_x
                            self.best_value = new_val
                            report_best(self.best_value, self.best_x)
                            step_sizes[d] *= 1.5
                        else:
                            break
                else:
                    # negative direction
                    x_candidate = self.best_x.copy()
                    x_candidate[d] = np.clip(self.best_x[d] - step, lb[d], ub[d])
                    val_candidate = func(x_candidate)
                    evals += 1
                    if val_candidate < self.best_value:
                        self.best_x = x_candidate
                        self.best_value = val_candidate
                        report_best(self.best_value, self.best_x)
                        step_sizes[d] *= 1.5
                        improved = True
                        # line search extension
                        while evals < self.budget:
                            new_x = self.best_x.copy()
                            new_x[d] = np.clip(self.best_x[d] - step_sizes[d], lb[d], ub[d])
                            new_val = func(new_x)
                            evals += 1
                            if new_val < self.best_value:
                                self.best_x = new_x
                                self.best_value = new_val
                                report_best(self.best_value, self.best_x)
                                step_sizes[d] *= 1.5
                            else:
                                break
                    else:
                        step_sizes[d] *= 0.7
                step_sizes[d] = max(step_sizes[d], 1e-10)
            else:
                # pattern search: random direction
                direction = self.rng.randn(self.dim)
                norm = np.linalg.norm(direction)
                if norm == 0:
                    direction = np.ones(self.dim) / np.sqrt(self.dim)
                else:
                    direction /= norm
                avg_step = np.mean(step_sizes)
                candidate = np.clip(self.best_x + avg_step * direction, lb, ub)
                val_candidate = func(candidate)
                evals += 1
                improved = False
                if val_candidate < self.best_value:
                    self.best_x = candidate
                    self.best_value = val_candidate
                    report_best(self.best_value, self.best_x)
                    step_sizes *= 1.5
                    improved = True
                    # line search extension along direction
                    while evals < self.budget:
                        new_x = np.clip(self.best_x + avg_step * direction, lb, ub)
                        new_val = func(new_x)
                        evals += 1
                        if new_val < self.best_value:
                            self.best_x = new_x
                            self.best_value = new_val
                            report_best(self.best_value, self.best_x)
                            step_sizes *= 1.5
                        else:
                            break
                else:
                    step_sizes *= 0.7
                step_sizes = np.maximum(step_sizes, 1e-10)
            iteration += 1
            if evals >= self.budget:
                break
        return self.best_value, self.best_x