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

        def evaluate(x):
            nonlocal evals, best_val, best_x
            if evals >= budget:
                return None
            x_clipped = np.clip(x, lb, ub)
            val = func(x_clipped)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x_clipped.copy()
                report_best(best_val, best_x)
            return val

        # Initial evaluation
        x0 = lb + (ub - lb) * rng.rand(dim)
        evaluate(x0)

        # If budget very small, just random search
        if budget <= 10:
            for _ in range(budget - evals):
                if evals >= budget:
                    break
                x = lb + (ub - lb) * rng.rand(dim)
                evaluate(x)
            return best_val, best_x

        # CEM parameters
        pop_size = max(3, min(30, budget // 5))
        max_iter = budget // pop_size
        elite_frac = 0.2
        n_elite = max(1, int(pop_size * elite_frac))
        lr = 0.5  # learning rate for distribution update

        # Initialize distribution
        mean = (lb + ub) / 2.0
        std = (ub - lb) / 4.0

        for it in range(max_iter):
            if evals >= budget:
                break
            # Sample population
            pop = rng.normal(loc=mean, scale=std, size=(pop_size, dim))
            pop = np.clip(pop, lb, ub)

            # Evaluate and store values
            values = []
            evaluated_points = []
            for i in range(pop_size):
                if evals >= budget:
                    break
                val = evaluate(pop[i])
                if val is None:
                    break
                values.append(val)
                evaluated_points.append(pop[i].copy())

            if len(values) == 0:
                break

            # Select elites
            idx = np.argsort(values)[:n_elite]
            elites = np.array([evaluated_points[i] for i in idx])

            # Update distribution
            new_mean = np.mean(elites, axis=0)
            new_std = np.std(elites, axis=0) + 1e-10  # avoid zero std
            # Smooth update
            mean = (1 - lr) * mean + lr * new_mean
            std = (1 - lr) * std + lr * new_std
            # Ensure std is not too small
            std = np.maximum(std, 1e-10 * (ub - lb))

        # Final random points if budget remains
        while evals < budget:
            x = lb + (ub - lb) * rng.rand(dim)
            evaluate(x)

        return best_val, best_x