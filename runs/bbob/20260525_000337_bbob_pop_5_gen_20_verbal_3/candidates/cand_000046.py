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
        calls = 0
        best_val = np.inf
        best_x = None

        def clip(x):
            return np.clip(x, lb, ub)

        def evaluate(x):
            nonlocal calls, best_val, best_x
            x = clip(x)
            val = func(x)
            calls += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            return val

        pop_size = max(10, min(50, dim * 5, self.budget // 10))
        pop_size = min(pop_size, self.budget)
        if pop_size < 4:
            pop = lb + (ub - lb) * rng.rand(pop_size, dim)
            for i in range(pop_size):
                evaluate(pop[i])
                if calls >= self.budget:
                    return best_val, best_x
        else:
            pop = lb + (ub - lb) * rng.rand(pop_size, dim)
            pop_val = np.full(pop_size, np.inf)
            for i in range(pop_size):
                pop_val[i] = evaluate(pop[i])
                if calls >= self.budget:
                    return best_val, best_x

            F = 0.8
            CR = 0.9
            max_gen = (self.budget - calls) // pop_size
            for gen in range(max_gen):
                F = 0.5 + 0.5 * rng.rand()  # dither
                for i in range(pop_size):
                    if calls >= self.budget:
                        return best_val, best_x
                    indices = list(range(pop_size))
                    indices.remove(i)
                    a, b, c = rng.choice(indices, 3, replace=False)
                    mutant = pop[a] + F * (pop[b] - pop[c])
                    mutant = clip(mutant)
                    trial = pop[i].copy()
                    cross_points = rng.rand(dim) < CR
                    if not np.any(cross_points):
                        cross_points[rng.randint(dim)] = True
                    trial[cross_points] = mutant[cross_points]
                    trial = clip(trial)
                    trial_val = evaluate(trial)
                    if trial_val < pop_val[i]:
                        pop[i] = trial
                        pop_val[i] = trial_val

        step0 = (ub - lb) / 20.0
        step_min = 1e-7 * np.max(ub - lb)
        step = step0.copy()
        x = best_x.copy()
        consecutive_failures = 0

        while calls < self.budget:
            improved_cycle = False
            dims_perm = rng.permutation(dim)
            for d in dims_perm:
                if calls >= self.budget:
                    return best_val, best_x
                x_new = x.copy()
                x_new[d] += step[d]
                x_new = clip(x_new)
                val_new = evaluate(x_new)
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
                    x = x_new.copy()
                    improved_cycle = True
                    consecutive_failures = 0
                    step[d] = min(step[d] * 1.2, step0[d])
                else:
                    x_new = x.copy()
                    x_new[d] -= step[d]
                    x_new = clip(x_new)
                    val_new = evaluate(x_new)
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        x = x_new.copy()
                        improved_cycle = True
                        consecutive_failures = 0
                        step[d] = min(step[d] * 1.2, step0[d])
                    else:
                        step[d] *= 0.5
                        if step[d] < step_min:
                            step[d] = step_min
            if not improved_cycle:
                consecutive_failures += 1
                if consecutive_failures >= 2 * dim:
                    noise = 0.01 * (ub - lb) * rng.randn(dim)
                    x = best_x + noise
                    x = clip(x)
                    step = step0.copy()
                    consecutive_failures = 0
        return best_val, best_x