import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(5 * dim, budget // 3))
        self.F = 0.5
        self.CR = 0.5
        # local search budget: remaining after DE
        self.local_budget = max(1, int(0.25 * budget))

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        n = self.popsize
        budget = self.budget
        evals = 0

        # Latin Hypercube Sampling for initial population
        pop = np.empty((n, dim))
        for d in range(dim):
            cuts = np.linspace(lb[d], ub[d], n+1)
            u = self.rng.uniform(0, 1, n)
            perm = self.rng.permutation(n)
            pop[perm, d] = cuts[:-1] + u * (cuts[1:] - cuts[:-1])
        pop = np.clip(pop, lb, ub)

        fitness = np.full(n, np.inf)
        best_x = None
        best_val = np.inf

        # initial evaluation
        for i in range(n):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # DE loop
        while evals < budget:
            for i in range(n):
                if evals >= budget:
                    break
                idx = list(range(n))
                idx.remove(i)
                a, b, c = self.rng.choice(idx, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                cross = self.rng.random(dim) < self.CR
                if not np.any(cross):
                    cross[self.rng.integers(dim)] = True
                trial = np.where(cross, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        # local search on best solution
        if self.local_budget > 0 and evals < budget:
            used = self._intensive_local_search(func, best_x, best_val, lb, ub, evals, budget)
            evals += used
            # update best_x and best_val from local search modifications (done within function)
        return best_val, best_x

    def _intensive_local_search(self, func, x, val, lb, ub, evals, budget):
        dim = self.dim
        step = 0.1 * (ub - lb)
        used = 0
        local_budget_left = self.local_budget
        max_iters = 3
        for _ in range(max_iters):
            if local_budget_left <= 0 or evals + used >= budget:
                break
            improved_any = False
            order = self.rng.permutation(dim)
            for i in order:
                if local_budget_left <= 0 or evals + used >= budget:
                    break
                # positive direction
                x_new = x.copy()
                x_new[i] = np.clip(x[i] + step[i], lb[i], ub[i])
                val_new = func(x_new)
                used += 1
                local_budget_left -= 1
                if val_new < val:
                    val = val_new
                    x = x_new.copy()
                    report_best(val, x)
                    step[i] *= 2.0
                    improved_any = True
                    continue
                # negative direction
                if local_budget_left <= 0 or evals + used >= budget:
                    break
                x_new = x.copy()
                x_new[i] = np.clip(x[i] - step[i], lb[i], ub[i])
                val_new = func(x_new)
                used += 1
                local_budget_left -= 1
                if val_new < val:
                    val = val_new
                    x = x_new.copy()
                    report_best(val, x)
                    step[i] *= 2.0
                    improved_any = True
                else:
                    step[i] *= 0.5
            if not improved_any:
                if local_budget_left <= 0 or evals + used >= budget:
                    break
                d = self.rng.normal(0, 1, dim)
                d = d / np.linalg.norm(d)
                scale = np.mean(step) * 0.5
                x_new = np.clip(x + scale * d, lb, ub)
                val_new = func(x_new)
                used += 1
                local_budget_left -= 1
                if val_new < val:
                    val = val_new
                    x = x_new.copy()
                    report_best(val, x)
        # Update best_x and best_val in caller via return
        # But we need to update the caller's variables. Simpler: return modified x, val, and then caller updates
        # However caller has local references; we'll return x and val and let caller overwrite its best?
        # But caller already has evals passed? Better: update through mutation of mutable object? Not possible.
        # We'll override optimizer's hidden parameters? No.
        # Instead, we modify the function signature: since we pass best_x and best_val as references, but Python doesn't have call by reference for scalars.
        # We can return (x, val, used) and let caller update.
        # But caller currently doesn't use return. We'll adjust: the local search returns (new_x, new_val, used) and caller updates best_x, best_val.
        # So we'll change the call site.
        return x, val, used