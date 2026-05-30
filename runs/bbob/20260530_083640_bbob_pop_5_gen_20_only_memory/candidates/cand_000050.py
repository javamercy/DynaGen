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
        # Initial random point
        best_x = lb + self.rng.rand(dim) * (ub - lb)
        best_f = func(best_x)
        evals = 1
        report_best(best_f, best_x)
        # Current point for polling (starts as best)
        current_x = best_x.copy()
        current_f = best_f
        # Step sizes
        step = 0.2 * (ub - lb)
        # Tracking last improvement
        last_improvement_evals = 0
        # Main loop
        while evals < self.budget:
            # Coordinate polling in random order
            perm = self.rng.permutation(dim)
            success_cycle = False
            for i in perm:
                if evals >= self.budget:
                    break
                # Positive direction
                trial = current_x.copy()
                trial[i] = np.clip(current_x[i] + step[i], lb[i], ub[i])
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    current_x = trial
                    current_f = f
                    step[i] *= 1.2
                    last_improvement_evals = evals
                    success_cycle = True
                    break
                # Negative direction
                trial = current_x.copy()
                trial[i] = np.clip(current_x[i] - step[i], lb[i], ub[i])
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    current_x = trial
                    current_f = f
                    step[i] *= 1.2
                    last_improvement_evals = evals
                    success_cycle = True
                    break
                else:
                    step[i] *= 0.8
            if not success_cycle and evals < self.budget:
                # Check stagnation
                if evals - last_improvement_evals > 2 * dim:
                    # Restart with new random point
                    current_x = lb + self.rng.rand(dim) * (ub - lb)
                    current_f = func(current_x)
                    evals += 1
                    if current_f < best_f:
                        best_f = current_f
                        best_x = current_x.copy()
                        report_best(best_f, best_x)
                    step = 0.2 * (ub - lb)
                    last_improvement_evals = evals
                else:
                    # Small random perturbation
                    delta = self.rng.randn(dim) * step
                    trial = np.clip(current_x + delta, lb, ub)
                    f = func(trial)
                    evals += 1
                    if f < best_f:
                        best_f = f
                        best_x = trial
                        report_best(best_f, best_x)
                        current_x = trial
                        current_f = f
                        step = 0.2 * (ub - lb)  # reset steps on improvement
                        last_improvement_evals = evals
        return best_f, best_x