import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(10, budget // 10))
        self.F_init = 0.8
        self.F_final = 0.2
        self.CR_init = 0.2
        self.CR_final = 0.9
        self.local_budget = max(1, int(0.5 * budget))
        self.de_budget = budget - self.local_budget
        self.stagnation_counter = 0
        self.restart_threshold = max(1, int(0.15 * budget))
        self.perturb_interval = max(1, int(0.05 * budget))

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        n = self.popsize
        evals = 0

        # Latin Hypercube Sampling
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

        # Initial evaluation
        for i in range(n):
            if evals >= self.budget:
                break
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)
                self.stagnation_counter = 0

        # DE main loop
        generation = 0
        while evals < self.de_budget:
            # Restart if stagnation
            if self.stagnation_counter >= self.restart_threshold:
                new_pop = [best_x.copy()]
                for _ in range(1, n):
                    x = self.rng.uniform(lb, ub, dim)
                    new_pop.append(x)
                pop = np.array(new_pop)
                for i in range(1, n):
                    if evals >= self.de_budget:
                        break
                    val = func(pop[i])
                    evals += 1
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                fitness[0] = best_val
                self.stagnation_counter = 0

            # Perturb best
            if evals % self.perturb_interval == 0 and evals < self.de_budget:
                d = self.rng.standard_cauchy(dim)
                d = d / (np.linalg.norm(d) + 1e-10)
                step = 0.1 * (ub - lb) * self.rng.uniform(0.1, 0.5)
                x_new = np.clip(best_x + step * d, lb, ub)
                val_new = func(x_new)
                evals += 1
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
                    report_best(best_val, best_x)
                    self.stagnation_counter = 0

            # Adaptive parameters
            max_gens = max(1, self.de_budget // n)
            progress = min(1.0, generation / max_gens)
            F = self.F_init + (self.F_final - self.F_init) * progress
            CR = self.CR_init + (self.CR_final - self.CR_init) * progress

            for i in range(n):
                if evals >= self.de_budget:
                    break
                idx = list(range(n))
                idx.remove(i)
                a, b, c = self.rng.choice(idx, 3, replace=False)
                F_eff = F if self.rng.random() < 0.9 else F * (1 + 0.5 * self.rng.normal())
                F_eff = np.clip(F_eff, 0.2, 1.0)
                mutant = pop[a] + F_eff * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                cross = self.rng.random(dim) < CR
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
                        self.stagnation_counter = 0
                else:
                    self.stagnation_counter += 1
            generation += 1

        # Intensive local search using remaining budget
        if evals < self.budget:
            x, val, used = self._intensive_local_search(func, best_x, best_val, lb, ub, evals)
            evals += used
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        return best_val, best_x

    def _intensive_local_search(self, func, x, val, lb, ub, evals):
        dim = self.dim
        step = 0.1 * (ub - lb)
        used = 0
        local_budget_left = self.local_budget
        max_rounds = 5
        for round in range(max_rounds):
            if local_budget_left <= 0 or evals + used >= self.budget:
                break
            improved = False
            order = self.rng.permutation(dim)
            for i in order:
                if local_budget_left <= 0 or evals + used >= self.budget:
                    break
                # Positive direction
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
                    improved = True
                    continue
                # Negative direction
                if local_budget_left <= 0 or evals + used >= self.budget:
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
                    improved = True
                else:
                    step[i] *= 0.5
            if not improved:
                if local_budget_left <= 0 or evals + used >= self.budget:
                    break
                d = self.rng.normal(0, 1, dim)
                d = d / (np.linalg.norm(d) + 1e-10)
                scale = np.mean(step) * 0.5
                x_new = np.clip(x + scale * d, lb, ub)
                val_new = func(x_new)
                used += 1
                local_budget_left -= 1
                if val_new < val:
                    val = val_new
                    x = x_new.copy()
                    report_best(val, x)
        return x, val, used