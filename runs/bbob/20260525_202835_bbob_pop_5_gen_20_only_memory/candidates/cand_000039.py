import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(10 * dim, budget // 2))
        self.F_start = 0.9
        self.F_end = 0.4
        self.CR_start = 0.9
        self.CR_end = 0.6
        self.local_budget = max(1, int(0.2 * budget))

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        n = self.popsize
        # LHS initialization
        pop = np.empty((n, dim))
        for d in range(dim):
            cut = np.linspace(lb[d], ub[d], n + 1)
            u = np.random.default_rng(self.seed + d).random(n)
            perm = np.random.default_rng(self.seed + 1000 + d).permutation(n)
            pop[perm, d] = cut[:-1] + u * (cut[1:] - cut[:-1])
        pop = np.clip(pop, lb, ub)
        fitness = np.full(n, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0
        # Initial evaluation
        for i in range(n):
            if evaluations >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evaluations += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        # Main DE loop
        max_gen = max(1, (self.budget - evaluations) // n)
        generation = 0
        while evaluations < self.budget and generation < max_gen:
            t = generation / max_gen
            F = self.F_start + (self.F_end - self.F_start) * t
            CR = self.CR_start + (self.CR_end - self.CR_start) * t
            for i in range(n):
                if evaluations >= self.budget:
                    break
                indices = list(range(n))
                indices.remove(i)
                a, b = self.rng.choice(indices, 2, replace=False)
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                cross_points = self.rng.random(dim) < CR
                if not np.any(cross_points):
                    cross_points[self.rng.integers(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evaluations += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            generation += 1
        # Local search on best
        if evaluations < self.budget and self.local_budget > 0:
            local_remaining = min(self.local_budget, self.budget - evaluations)
            step = 0.1 * (ub - lb)
            for _ in range(3):  # up to 3 sweeps
                if local_remaining <= 0:
                    break
                improved_any = False
                order = self.rng.permutation(dim)
                for i in order:
                    if local_remaining <= 0:
                        break
                    # Positive direction
                    x_new = best_x.copy()
                    x_new[i] = np.clip(best_x[i] + step[i], lb[i], ub[i])
                    val_new = func(x_new)
                    local_remaining -= 1
                    evaluations += 1
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        report_best(best_val, best_x)
                        step[i] *= 2.0
                        improved_any = True
                        continue
                    # Negative direction
                    if local_remaining <= 0:
                        break
                    x_new = best_x.copy()
                    x_new[i] = np.clip(best_x[i] - step[i], lb[i], ub[i])
                    val_new = func(x_new)
                    local_remaining -= 1
                    evaluations += 1
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        report_best(best_val, best_x)
                        step[i] *= 2.0
                        improved_any = True
                    else:
                        step[i] *= 0.5
                if not improved_any and local_remaining > 0:
                    # Random direction perturbation
                    d = self.rng.normal(0, 1, dim)
                    d = d / np.linalg.norm(d)
                    scale = np.mean(step) * 0.5
                    x_new = np.clip(best_x + scale * d, lb, ub)
                    val_new = func(x_new)
                    local_remaining -= 1
                    evaluations += 1
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        report_best(best_val, best_x)
        return best_val, best_x