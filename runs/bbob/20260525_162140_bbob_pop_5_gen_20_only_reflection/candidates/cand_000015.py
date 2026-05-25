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

        # Initialization
        best_x = lb + rng.rand(dim) * (ub - lb)
        best_f = func(best_x)
        report_best(best_f, best_x)
        evals = 1

        # Step sizes for each dimension
        step_size = 0.2 * (ub - lb)
        # Success/failure counters for each dimension
        successes = np.zeros(dim, dtype=int)
        failures = np.zeros(dim, dtype=int)
        # Window size for adaptation
        window = max(1, dim * 5)
        stagnation_limit = max(10, dim * 10)
        no_improve_steps = 0

        while evals < budget:
            # Choose a random dimension
            d = rng.randint(dim)
            # Perturb that dimension
            delta = step_size[d] * rng.uniform(-1, 1)
            candidate = best_x.copy()
            candidate[d] = best_x[d] + delta
            candidate = np.clip(candidate, lb, ub)
            f = func(candidate)
            evals += 1

            if f < best_f - 1e-12:
                best_f = f
                best_x = candidate
                report_best(best_f, best_x)
                successes[d] += 1
                no_improve_steps = 0
            else:
                failures[d] += 1
                no_improve_steps += 1

            # Adaptation step (every 'window' evaluations)
            if evals % window == 0:
                for d in range(dim):
                    total = successes[d] + failures[d]
                    if total > 0:
                        success_rate = successes[d] / total
                        if success_rate > 0.2:
                            step_size[d] = min(step_size[d] * 1.2, 0.5 * (ub[d] - lb[d]))
                        else:
                            step_size[d] = max(step_size[d] * 0.8, 0.01 * (ub[d] - lb[d]))
                successes.fill(0)
                failures.fill(0)

            # Restart if stagnation
            if no_improve_steps >= stagnation_limit:
                best_x = lb + rng.rand(dim) * (ub - lb)
                best_f = func(best_x)
                evals += 1
                report_best(best_f, best_x)
                step_size = 0.2 * (ub - lb)
                successes.fill(0)
                failures.fill(0)
                no_improve_steps = 0

        return best_f, best_x