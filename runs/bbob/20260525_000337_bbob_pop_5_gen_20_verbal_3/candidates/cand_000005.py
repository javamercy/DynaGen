import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
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

        # Differential Evolution
        pop_size = max(10, min(50, dim * 5, self.budget // 10))
        pop = lb + (ub - lb) * self.rng.rand(pop_size, dim)
        pop_val = np.full(pop_size, np.inf)
        for i in range(pop_size):
            pop_val[i] = evaluate(pop[i])
            if calls >= self.budget:
                return best_val, best_x

        F = 0.8
        CR = 0.9
        max_gen = (self.budget - calls) // pop_size
        for gen in range(max_gen):
            for i in range(pop_size):
                if calls >= self.budget:
                    return best_val, best_x
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = self.rng.choice(indices, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = clip(mutant)
                trial = pop[i].copy()
                cross_points = self.rng.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[self.rng.randint(dim)] = True
                trial[cross_points] = mutant[cross_points]
                trial = clip(trial)
                trial_val = evaluate(trial)
                if trial_val < pop_val[i]:
                    pop[i] = trial
                    pop_val[i] = trial_val
                if calls >= self.budget:
                    return best_val, best_x

        # Pattern Search
        step0 = (ub - lb) / 20.0
        step_min = 1e-7 * np.max(ub - lb)
        x0 = best_x.copy()
        step = step0.copy()
        while calls < self.budget:
            improved = False
            for d in range(dim):
                if calls >= self.budget:
                    break
                x_new = x0.copy()
                x_new[d] += step[d]
                x_new = clip(x_new)
                val_new = evaluate(x_new)
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
                    x0 = x_new.copy()
                    improved = True
                else:
                    x_new = x0.copy()
                    x_new[d] -= step[d]
                    x_new = clip(x_new)
                    val_new = evaluate(x_new)
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        x0 = x_new.copy()
                        improved = True
            if not improved:
                step = step * 0.5
                if np.all(step < step_min):
                    break
        return best_val, best_x