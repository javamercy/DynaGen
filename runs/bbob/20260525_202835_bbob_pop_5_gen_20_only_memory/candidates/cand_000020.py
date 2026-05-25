import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(10 * dim, budget // 2))
        self.F = 0.9
        self.CR = 0.9
        self.local_budget = int(0.15 * budget)
        self.stagnation_limit = 10

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        n = self.popsize
        # Latin hypercube initialization
        pop = np.empty((n, dim))
        for d in range(dim):
            cuts = np.linspace(lb[d], ub[d], n+1)
            u = self.rng.random(n)
            perm = self.rng.permutation(n)
            pop[perm, d] = cuts[:-1] + u * (cuts[1:] - cuts[:-1])
        pop = np.clip(pop, lb, ub)
        fitness = np.full(n, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0
        # Initial evaluation
        for i in range(n):
            if evaluations >= self.budget:
                break
            val = func(pop[i])
            evaluations += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)
        gen = 0
        stagnation_count = 0
        while evaluations < self.budget:
            # DE generation
            new_pop = pop.copy()
            for i in range(n):
                if evaluations >= self.budget:
                    break
                idxs = [j for j in range(n) if j != i]
                a, b, c = self.rng.choice(idxs, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                cross = self.rng.random(dim) < self.CR
                if not np.any(cross):
                    cross[self.rng.integers(dim)] = True
                trial = np.where(cross, mutant, pop[i])
                val = func(trial)
                evaluations += 1
                if val < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            pop = new_pop
            gen += 1
            # Diversity check
            if evaluations < self.budget:
                diversity = np.mean(pop.std(axis=0))
                bound_range = np.mean(ub - lb)
                if best_val < np.inf and diversity < 0.1 * bound_range:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                if stagnation_count >= self.stagnation_limit:
                    # Restart worst half
                    idx_sort = np.argsort(fitness)
                    keep = n // 2
                    worst_idx = idx_sort[keep:]
                    for idx in worst_idx:
                        if evaluations >= self.budget:
                            break
                        for d in range(dim):
                            pop[idx, d] = self.rng.uniform(lb[d], ub[d])
                        pop[idx] = np.clip(pop[idx], lb, ub)
                        val = func(pop[idx])
                        evaluations += 1
                        fitness[idx] = val
                        if val < best_val:
                            best_val = val
                            best_x = pop[idx].copy()
                            report_best(best_val, best_x)
                    stagnation_count = 0
            # Local search on best
            if self.local_budget > 0 and evaluations < self.budget:
                best_x, best_val, used = self._local_search(func, best_x, best_val, lb, ub, evaluations)
                evaluations += used
                self.local_budget -= used
        return best_val, best_x

    def _local_search(self, func, x, val, lb, ub, evaluations):
        step = 0.1 * (ub - lb)
        dim = self.dim
        used = 0
        improvement = True
        while self.local_budget > 0 and evaluations + used < self.budget:
            if not improvement:
                step *= 0.5
                improvement = True
            improved = False
            for i in range(dim):
                if self.local_budget <= 0 or evaluations + used >= self.budget:
                    break
                # Positive direction
                x_new = x.copy()
                x_new[i] = np.clip(x[i] + step[i], lb[i], ub[i])
                val_new = func(x_new)
                used += 1
                self.local_budget -= 1
                if val_new < val:
                    val = val_new
                    x = x_new.copy()
                    report_best(val, x)
                    step[i] *= 1.5
                    improved = True
                    continue
                # Negative direction
                if self.local_budget <= 0 or evaluations + used >= self.budget:
                    break
                x_new = x.copy()
                x_new[i] = np.clip(x[i] - step[i], lb[i], ub[i])
                val_new = func(x_new)
                used += 1
                self.local_budget -= 1
                if val_new < val:
                    val = val_new
                    x = x_new.copy()
                    report_best(val, x)
                    step[i] *= 1.5
                    improved = True
            if not improved:
                improvement = False
        return x, val, used