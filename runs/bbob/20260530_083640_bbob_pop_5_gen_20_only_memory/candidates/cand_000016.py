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
        budget = self.budget

        best_x = None
        best_f = np.inf
        evals = 0

        # Restart parameters
        max_stagnation = max(1, int(dim * 1.5))  # evaluations without improvement before restart

        while evals < budget:
            # Random initial point
            x = lb + rng.rand(dim) * (ub - lb)
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

            step = 0.2 * (ub - lb)  # adaptive step sizes per coordinate
            stagnation_count = 0

            while evals < budget and stagnation_count < max_stagnation:
                success = False

                # Random permutation of coordinates
                perm = rng.permutation(dim)
                for i in perm:
                    if evals >= budget:
                        break
                    # Positive direction
                    trial = x.copy()
                    trial[i] = np.clip(x[i] + step[i], lb[i], ub[i])
                    f_trial = func(trial)
                    evals += 1
                    if f_trial < f:
                        f = f_trial
                        x = trial
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                            report_best(best_f, best_x)
                        step[i] = min(step[i] * 2, ub[i] - lb[i])
                        success = True
                        break

                    # Negative direction
                    trial = x.copy()
                    trial[i] = np.clip(x[i] - step[i], lb[i], ub[i])
                    f_trial = func(trial)
                    evals += 1
                    if f_trial < f:
                        f = f_trial
                        x = trial
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                            report_best(best_f, best_x)
                        step[i] = min(step[i] * 2, ub[i] - lb[i])
                        success = True
                        break
                    else:
                        step[i] = max(step[i] * 0.5, (ub[i] - lb[i]) * 1e-10)

                if not success and evals < budget:
                    # Random direction poll
                    direction = rng.randn(dim)
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                    trial = np.clip(x + step * direction, lb, ub)
                    f_trial = func(trial)
                    evals += 1
                    if f_trial < f:
                        f = f_trial
                        x = trial
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                            report_best(best_f, best_x)
                        step = np.minimum(step * 2, ub - lb)
                        success = True

                if success:
                    stagnation_count = 0
                else:
                    stagnation_count += 1

            # If we exit the inner loop due to stagnation, restart
            if evals >= budget:
                break
            # Reset stagnation count for next restart
            # Continue while loop to restart

        return best_f, best_x