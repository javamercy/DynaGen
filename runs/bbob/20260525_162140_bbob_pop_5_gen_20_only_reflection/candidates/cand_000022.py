import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        NP = max(4, min(10*dim, budget // 2 - 1))
        if NP > budget:
            NP = budget

        pop = lb + (ub - lb) * rng.rand(NP, dim)
        fitness = np.zeros(NP)
        for i in range(NP):
            fitness[i] = func(pop[i])
        evals = NP

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_val = fitness[best_idx]
        report_best(best_val, best_x)

        H = 5
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        archive_F = []
        archive_CR = []
        k = 0
        restarts = 0
        max_restarts = 3
        diversity_threshold = 0.01 * np.mean(ub - lb)

        while evals < budget:
            # Restart condition
            if restarts < max_restarts and evals > NP:
                pop_std = np.mean(np.std(pop, axis=0))
                if pop_std < diversity_threshold and evals + NP - 1 <= budget:
                    new_pop = [best_x]
                    for _ in range(NP - 1):
                        new_pop.append(lb + (ub - lb) * rng.rand(dim))
                    pop = np.array(new_pop)
                    fitness[0] = best_val
                    for i in range(1, NP):
                        fitness[i] = func(pop[i])
                        evals += 1
                    restarts += 1
                    continue

            for i in range(NP):
                if evals >= budget:
                    break
                candidates = list(range(NP))
                candidates.remove(i)
                a, b, c = pop[rng.choice(candidates, 3, replace=False)]
                r = rng.randint(H)
                F = rng.cauchy(loc=M_F[r], scale=0.1)
                if F <= 0:
                    F = 0.1
                if F > 1:
                    F = 1.0
                CR = rng.normal(loc=M_CR[r], scale=0.1)
                CR = np.clip(CR, 0, 1)
                mutant = a + F * (b - c)
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_val:
                        best_val = trial_fitness
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                    archive_F.append(F)
                    archive_CR.append(CR)

            if len(archive_F) > 0:
                M_F[k] = np.mean(archive_F)
                M_CR[k] = np.mean(archive_CR)
                k = (k + 1) % H
                archive_F = []
                archive_CR = []

        return best_val, best_x