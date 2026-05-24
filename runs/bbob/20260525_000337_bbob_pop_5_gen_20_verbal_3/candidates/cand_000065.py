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

        # If budget too small, just random search
        if self.budget < 10:
            for _ in range(self.budget):
                x = lb + (ub - lb) * self.rng.rand(dim)
                evaluate(x)
            return best_val, best_x

        # Differential Evolution
        pop_size = max(10, min(50, dim * 5, self.budget // 10))
        if pop_size > self.budget:
            pop_size = self.budget
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

        # Adaptive Coordinate Search
        step0 = (ub - lb) / 20.0
        step_min = 1e-7 * np.max(ub - lb)
        x0 = best_x.copy()
        step = step0.copy()
        no_improve_count = 0
        max_no_improve = max(10, int(0.05 * self.budget))
        while calls < self.budget:
            improved = False
            for d in range(dim):
                if calls >= self.budget:
                    break
                # try positive step
                x_new = x0.copy()
                x_new[d] += step[d]
                x_new = clip(x_new)
                val_new = evaluate(x_new)
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
                    x0 = x_new.copy()
                    improved = True
                    step[d] *= 1.2  # slight increase on success
                else:
                    # try negative step
                    x_new = x0.copy()
                    x_new[d] -= step[d]
                    x_new = clip(x_new)
                    val_new = evaluate(x_new)
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        x0 = x_new.copy()
                        improved = True
                        step[d] *= 1.2
                    else:
                        step[d] *= 0.5
                # ensure step not too small
                if step[d] < step_min:
                    step[d] = step_min
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
            # restart if stagnation
            if no_improve_count >= max_no_improve and calls < self.budget:
                # perturb best point
                noise = 0.1 * (ub - lb) * self.rng.randn(dim)
                pert = clip(x0 + noise)
                val_pert = evaluate(pert)
                if val_pert < best_val:
                    best_val = val_pert
                    best_x = pert.copy()
                    x0 = pert.copy()
                else:
                    x0 = best_x.copy()
                step = step0.copy()
                no_improve_count = 0
        return best_val, best_x