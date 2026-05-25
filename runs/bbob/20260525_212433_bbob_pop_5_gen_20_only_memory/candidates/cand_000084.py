import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng
        evals = 0
        budget = self.budget

        # Initial point
        x = lb + (ub - lb) * rng.rand(self.dim)
        val = func(x)
        evals += 1
        self.best_val = val
        self.best_x = x.copy()
        report_best(self.best_val, self.best_x)

        # Parameters
        T = 1.0
        step_size = 0.1
        stagnation_limit = max(1, budget // 10)
        no_improve_evals = 0

        # Annealing loop
        while evals < budget:
            # Generate candidate by perturbing each dimension
            noise = rng.randn(self.dim) * step_size * (ub - lb)
            x_new = x + noise
            x_new = np.clip(x_new, lb, ub)
            val_new = func(x_new)
            evals += 1

            if val_new < self.best_val:
                self.best_val = val_new
                self.best_x = x_new.copy()
                report_best(self.best_val, self.best_x)
                no_improve_evals = 0
            else:
                no_improve_evals += 1

            # Acceptance probability
            if val_new < val:
                x = x_new
                val = val_new
            else:
                delta = val_new - val
                if rng.rand() < np.exp(-delta / T):
                    x = x_new
                    val = val_new
                # else reject, keep current

            # Cool down
            T = 1.0 * (1 - evals / budget)  # linear cooling

            # Adapt step size based on recent acceptance? Simple heuristic: if many stucks, reduce
            # Use a simple rule: if no improvement for a while, increase step to escape
            if no_improve_evals >= stagnation_limit:
                # Restart from best with larger step
                x = self.best_x.copy()
                val = self.best_val
                step_size = min(0.25, step_size * 2)
                no_improve_evals = 0
            else:
                # Slight adaptation: if many rejections, reduce step; if many accepts, increase
                # Here we keep it simple: step_size decays slowly
                step_size *= 0.999
                step_size = max(0.001, min(0.5, step_size))

        return self.best_val, self.best_x