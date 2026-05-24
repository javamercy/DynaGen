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

        # Stratified initial sampling
        n_init = min(100, self.budget // 10, 5 * dim)
        samples = np.empty((n_init, dim))
        for i in range(dim):
            samples[:, i] = self.rng.uniform(lb[i], ub[i], size=n_init)
        for i in range(n_init):
            if calls >= self.budget:
                return best_val, best_x
            evaluate(samples[i])

        # Differential Evolution with dithering
        pop_size = max(20, min(100, 4 * dim, (self.budget - calls) // 5))
        pop = lb + (ub - lb) * self.rng.rand(pop_size, dim)
        pop_val = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if calls >= self.budget:
                return best_val, best_x
            pop_val[i] = evaluate(pop[i])

        max_gen = (self.budget - calls) // pop_size
        for gen in range(max_gen):
            for i in range(pop_size):
                if calls >= self.budget:
                    return best_val, best_x
                idxs = list(range(pop_size))
                idxs.remove(i)
                a, b, c = self.rng.choice(idxs, 3, replace=False)
                F = 0.8 + 0.2 * self.rng.rand()
                CR = 0.9
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = clip(mutant)
                trial = pop[i].copy()
                cross = self.rng.rand(dim) < CR
                if not np.any(cross):
                    cross[self.rng.randint(dim)] = True
                trial[cross] = mutant[cross]
                trial = clip(trial)
                trial_val = evaluate(trial)
                if trial_val < pop_val[i]:
                    pop[i] = trial
                    pop_val[i] = trial_val

        # Adaptive Coordinate Descent with restarts
        step0 = (ub - lb) / 20.0
        step_min = 1e-12 * np.max(ub - lb)
        x0 = best_x.copy()
        step = step0.copy()
        no_improve = 0
        max_no_improve = 20 * dim

        while calls < self.budget:
            improved = False
            for d in range(dim):
                if calls >= self.budget:
                    break
                # positive step
                x_new = x0.copy()
                x_new[d] += step[d]
                x_new = clip(x_new)
                val_new = evaluate(x_new)
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
                    x0 = x_new.copy()
                    step[d] *= 1.2
                    improved = True
                    no_improve = 0
                else:
                    # negative step
                    x_new = x0.copy()
                    x_new[d] -= step[d]
                    x_new = clip(x_new)
                    val_new = evaluate(x_new)
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        x0 = x_new.copy()
                        step[d] *= 1.2
                        improved = True
                        no_improve = 0
                    else:
                        step[d] *= 0.5
                        if step[d] < step_min:
                            step[d] = step_min
            if not improved:
                no_improve += 1
                if no_improve >= max_no_improve:
                    # restart from random point
                    if calls >= self.budget:
                        break
                    x0 = lb + (ub - lb) * self.rng.rand(dim)
                    step = step0.copy()
                    no_improve = 0
                    # evaluate new starting point
                    val = evaluate(x0)
                    if val < best_val:
                        best_val = val
                        best_x = x0.copy()

        return best_val, best_x