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
        budget = self.budget
        rng = self.rng

        pop_size = max(4, min(20, budget // 3, 2 * dim))
        if pop_size > budget:
            pop_size = budget
        if pop_size < 4:
            pop_size = max(2, budget)

        pop = rng.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf

        evals = 0
        for i in range(pop_size):
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        if evals >= budget:
            return best_val, best_x

        remaining = budget - evals
        DE_budget = int(0.6 * remaining)
        local_budget = remaining - DE_budget

        # DE phase: DE/current-to-best/1 with exponential crossover, decreasing F and CR
        F_initial = 0.9
        F_final = 0.4
        CR_initial = 0.9
        CR_final = 0.2

        for iter in range(DE_budget):
            progress = iter / max(1, DE_budget - 1)
            F = F_initial - (F_initial - F_final) * progress
            CR = CR_initial - (CR_initial - CR_final) * progress

            for i in range(pop_size):
                if evals >= budget:
                    break
                a, b = rng.choice([j for j in range(pop_size) if j != i], 2, replace=False)
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)

                j_start = rng.randint(dim)
                L = 1
                while rng.rand() < CR and L < dim:
                    L += 1
                trial = pop[i].copy()
                for k in range(L):
                    j = (j_start + k) % dim
                    trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)

                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        # Local search phase: random perturbations with decaying step
        if local_budget > 0:
            step_init = 0.2 * (ub - lb)
            for i in range(local_budget):
                if evals >= budget:
                    break
                progress = i / max(1, local_budget - 1)
                step = step_init * (1 - 0.5 * progress)
                trial = best_x + rng.randn(dim) * step
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                    worst_idx = np.argmax(fitness)
                    if val < fitness[worst_idx]:
                        pop[worst_idx] = trial
                        fitness[worst_idx] = val

        return best_val, best_x