import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = float('inf')
        self.best_x = None

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        evals = 0
        popsize = min(budget, max(4, min(4 * dim, 20)))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fitness = np.full(popsize, np.inf)
        for i in range(popsize):
            val = func(pop[i])
            pop_fitness[i] = val
            evals += 1
            if val < self.best_value:
                self.best_value = val
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x

        CR = 0.9
        no_improve_gen = 0
        while evals < budget:
            improved = False
            for i in range(popsize):
                if evals >= budget:
                    break
                indices = list(range(popsize))
                indices.remove(i)
                rng.shuffle(indices)
                a, b, c = indices[:3]
                F = 0.5 + 0.5 * rng.rand()
                mutant = pop[a] + F * (pop[b] - pop[c])
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val <= pop_fitness[i]:
                    pop_fitness[i] = val
                    pop[i] = trial
                    if val < self.best_value:
                        self.best_value = val
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        improved = True
            if improved:
                no_improve_gen = 0
            else:
                no_improve_gen += 1
            if no_improve_gen >= 2 * dim and evals < budget:
                n_restart = popsize // 2
                if n_restart > 0:
                    worst_indices = np.argsort(pop_fitness)[-n_restart:]
                    for idx in worst_indices:
                        if idx == np.argmin(pop_fitness):
                            continue
                        pop[idx] = lb + (ub - lb) * rng.rand(dim)
                        val = func(pop[idx])
                        evals += 1
                        pop_fitness[idx] = val
                        if val < self.best_value:
                            self.best_value = val
                            self.best_x = pop[idx].copy()
                            report_best(self.best_value, self.best_x)
                        if evals >= budget:
                            break
                no_improve_gen = 0
        return self.best_value, self.best_x