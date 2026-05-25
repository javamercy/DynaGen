import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(20, min(4 * dim, budget // 2))
        self.stall_limit = max(10, budget // 20)
        self.local_budget = max(5, int(0.15 * budget))
        self.de_budget = budget - self.local_budget

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        pop = self.rng.uniform(lb, ub, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0
        # initial evaluation
        for i in range(popsize):
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
        if best_x is None:
            best_x = pop[0].copy()
            best_val = fitness[0]
            report_best(best_val, best_x)
        # DE phase
        max_gens = self.de_budget // popsize
        gen = 0
        generations_since_improvement = 0
        while evaluations < self.de_budget and gen < max_gens:
            progress = gen / max_gens if max_gens > 0 else 1.0
            base_F = 0.9 - 0.7 * progress  # from 0.9 to 0.2
            base_CR = 0.2 + 0.7 * progress  # from 0.2 to 0.9
            F = np.clip(base_F + self.rng.uniform(-0.1, 0.1), 0.1, 1.0)
            CR = np.clip(base_CR + self.rng.uniform(-0.1, 0.1), 0.1, 1.0)
            for i in range(popsize):
                if evaluations >= self.de_budget:
                    break
                idx_best = np.argmin(fitness)
                candidates = [j for j in range(popsize) if j != i]
                r1, r2 = self.rng.choice(candidates, 2, replace=False)
                mutant = pop[i] + F * (pop[idx_best] - pop[i]) + F * (pop[r1] - pop[r2])
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
                        generations_since_improvement = 0
                    else:
                        generations_since_improvement += 1
                else:
                    generations_since_improvement += 1
            gen += 1
            # restart if stalled
            if generations_since_improvement > self.stall_limit:
                n_restart = popsize // 2
                restart_indices = self.rng.choice(popsize, n_restart, replace=False)
                for idx in restart_indices:
                    if evaluations >= self.budget:
                        break
                    new_x = self.rng.uniform(lb, ub)
                    val = func(new_x)
                    evaluations += 1
                    pop[idx] = new_x
                    fitness[idx] = val
                    if val < best_val:
                        best_val = val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)
                generations_since_improvement = 0
        # local search phase
        step_size = (ub - lb) * 0.01
        while evaluations < self.budget:
            trial = best_x + self.rng.normal(0, step_size)
            trial = np.clip(trial, lb, ub)
            val = func(trial)
            evaluations += 1
            if val < best_val:
                best_val = val
                best_x = trial.copy()
                report_best(best_val, best_x)
        return best_val, best_x