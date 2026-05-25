import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(20, min(4 * dim, self.budget // 2))
        self.stall_limit = max(10, budget // 20)

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
            best_x = self.rng.uniform(lb, ub)
            best_val = func(best_x)
            evaluations += 1
            report_best(best_val, best_x)
        # main loop
        generations_since_improvement = 0
        while evaluations < self.budget:
            F = self.rng.uniform(0.5, 1.0)
            CR = self.rng.uniform(0.5, 1.0)
            for i in range(popsize):
                if evaluations >= self.budget:
                    break
                # select three distinct indices different from i
                candidates = [j for j in range(popsize) if j != i]
                if len(candidates) < 3:
                    continue
                r1, r2, r3 = self.rng.choice(candidates, 3, replace=False)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
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
        return best_val, best_x