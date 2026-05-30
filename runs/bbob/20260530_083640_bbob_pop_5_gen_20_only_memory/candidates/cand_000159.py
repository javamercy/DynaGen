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
        budget = self.budget
        rng = self.rng

        best_val = np.inf
        best_x = None
        evals = 0

        # Evaluate initial point
        x = lb + rng.rand(dim) * (ub - lb)
        val = func(x)
        evals += 1
        best_val = val
        best_x = x.copy()
        report_best(best_val, best_x)

        # Simulated Annealing parameters
        T0 = 1.0  # initial temperature
        T = T0
        alpha = 0.98  # cooling factor
        sigma = 0.2 * (ub - lb)  # initial step size
        sigma_min = 1e-5 * (ub - lb)
        stagnation = 0
        max_stagnation = 20
        current_x = x.copy()
        current_val = val

        while evals < budget:
            # Generate candidate
            step = sigma * rng.randn(dim)
            candidate = np.clip(current_x + step, lb, ub)
            cand_val = func(candidate)
            evals += 1

            # Acceptance criterion
            delta = cand_val - current_val
            if delta <= 0 or rng.rand() < np.exp(-delta / max(T, 1e-10)):
                current_x = candidate
                current_val = cand_val
                if cand_val < best_val:
                    best_val = cand_val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1

            # Cool temperature
            T *= alpha

            # Adapt step size based on recent acceptance (simple: reduce if many rejections)
            if stagnation > dim * 2:
                sigma = np.maximum(sigma * 0.5, sigma_min)
                stagnation = 0

            # Restart if stuck
            if stagnation >= max_stagnation and evals < budget:
                # Random restart
                new_x = lb + rng.rand(dim) * (ub - lb)
                new_val = func(new_x)
                evals += 1
                current_x = new_x
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x.copy()
                    report_best(best_val, best_x)
                T = T0
                sigma = 0.2 * (ub - lb)
                stagnation = 0

            # Early termination if temperature very low and no improvement
            if T < 1e-10 and stagnation > dim:
                break

        # Final coordinate search refinement
        if evals < budget and best_x is not None:
            step = 0.1 * (ub - lb)
            for i in range(dim):
                if evals >= budget:
                    break
                for direction in [1, -1]:
                    if evals >= budget:
                        break
                    trial = best_x.copy()
                    trial[i] = np.clip(best_x[i] + direction * step[i], lb[i], ub[i])
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        step[i] *= 2
                        break
                    else:
                        step[i] *= 0.5

        return best_val, best_x