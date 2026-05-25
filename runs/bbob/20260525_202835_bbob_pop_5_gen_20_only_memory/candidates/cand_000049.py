import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(4 * dim, budget // 2))
        self.F = 0.7
        self.CR = 0.8
        self.restart_threshold = max(1, int(0.15 * budget))
        self.local_budget = max(1, int(0.1 * budget))

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        n = self.popsize
        budget = self.budget
        evals = 0

        # LHS initialization
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
        stagnation_counter = 0

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
                stagnation_counter = 0

        # DE loop
        while evals < budget:
            # restart if stagnated
            if stagnation_counter >= self.restart_threshold:
                new_pop = [best_x.copy()]
                for _ in range(1, n):
                    x = self.rng.uniform(lb, ub, dim)
                    new_pop.append(x)
                pop = np.array(new_pop)
                for i in range(1, n):
                    if evals >= budget:
                        break
                    val = func(pop[i])
                    evals += 1
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                fitness[0] = best_val
                stagnation_counter = 0

            # DE generation
            for i in range(n):
                if evals >= budget:
                    break
                idx_best = np.argmin(fitness[:i] + fitness[i+1:])  # exclude current? Actually use full pop
                # current-to-best/1
                idxs = list(range(n))
                idxs.remove(i)
                r1, r2 = self.rng.choice(idxs, 2, replace=False)
                mutant = pop[i] + self.F * (pop[idx_best] - pop[i]) + self.F * (pop[r1] - pop[r2])
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
                        stagnation_counter = 0
                else:
                    stagnation_counter += 1

        # local search on best
        if self.local_budget > 0 and evals < budget:
            x = best_x.copy()
            val = best_val
            step = 0.1 * (ub - lb)
            local_used = 0
            for _ in range(3):  # max 3 passes
                if local_used >= self.local_budget or evals + local_used >= budget:
                    break
                improved = False
                order = self.rng.permutation(dim)
                for i in order:
                    if local_used >= self.local_budget or evals + local_used >= budget:
                        break
                    # try positive
                    x_new = x.copy()
                    x_new[i] = np.clip(x[i] + step[i], lb[i], ub[i])
                    val_new = func(x_new)
                    local_used += 1
                    if val_new < val:
                        val = val_new
                        x = x_new.copy()
                        report_best(val, x)
                        step[i] *= 2.0
                        improved = True
                        continue
                    # try negative
                    if local_used >= self.local_budget or evals + local_used >= budget:
                        break
                    x_new = x.copy()
                    x_new[i] = np.clip(x[i] - step[i], lb[i], ub[i])
                    val_new = func(x_new)
                    local_used += 1
                    if val_new < val:
                        val = val_new
                        x = x_new.copy()
                        report_best(val, x)
                        step[i] *= 2.0
                        improved = True
                    else:
                        step[i] *= 0.5
                if not improved:
                    # random perturbation
                    if local_used >= self.local_budget or evals + local_used >= budget:
                        break
                    d = self.rng.normal(0, 1, dim)
                    d = d / np.linalg.norm(d)
                    scale = np.mean(step) * 0.5
                    x_new = np.clip(x + scale * d, lb, ub)
                    val_new = func(x_new)
                    local_used += 1
                    if val_new < val:
                        val = val_new
                        x = x_new.copy()
                        report_best(val, x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        return best_val, best_x