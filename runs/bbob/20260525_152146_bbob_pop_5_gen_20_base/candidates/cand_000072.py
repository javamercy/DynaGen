import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng
        budget = self.budget

        # initial point
        x = rng.uniform(lb, ub, size=dim)
        f = func(x)
        budget -= 1
        best_x = x.copy()
        best_f = f
        report_best(best_f, best_x)

        if budget <= 0:
            return best_f, best_x

        # ES parameters
        mu = 1
        lambda_ = 4
        sigma = 0.2 * (ub - lb).mean()  # initial step size
        min_sigma = 1e-12
        max_sigma = np.max(ub - lb)

        parent_x = x
        parent_f = f

        while budget > 0:
            # generate offspring
            offspring_f = []
            offspring_x = []
            for _ in range(lambda_):
                if budget <= 0:
                    break
                z = rng.randn(dim)
                off = parent_x + sigma * z
                off = np.clip(off, lb, ub)
                off_f = func(off)
                budget -= 1
                offspring_f.append(off_f)
                offspring_x.append(off)
                if off_f < best_f:
                    best_f = off_f
                    best_x = off.copy()
                    report_best(best_f, best_x)

            # (1+lambda) selection: best among parent and offspring
            all_f = [parent_f] + offspring_f
            all_x = [parent_x] + offspring_x
            best_idx = np.argmin(all_f)
            parent_x = all_x[best_idx].copy()
            parent_f = all_f[best_idx]

            # step size adaptation (1/5 rule)
            if len(offspring_f) >= 1:
                success_rate = sum(o < parent_f for o in offspring_f) / len(offspring_f)
                if success_rate > 0.2:
                    sigma *= 1.2
                else:
                    sigma /= 1.2
                sigma = np.clip(sigma, min_sigma, max_sigma)

        return best_f, best_x