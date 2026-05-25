import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = min(max(4, min(5 * self.dim, self.budget // 2)), self.budget)
        if pop_size < 4:
            best_x = None
            best_f = np.inf
            for _ in range(self.budget):
                x = self.rng.uniform(lb, ub)
                f = func(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    report_best(best_f, best_x)
            return best_f, best_x
        points = self.rng.uniform(lb, ub, size=(pop_size, self.dim))
        pop_fitness = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0
        for i in range(pop_size):
            x = points[i]
            f = func(x)
            evals += 1
            pop_fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
        F_pop = self.rng.uniform(0.1, 1.0, pop_size)
        CR_pop = self.rng.uniform(0.0, 1.0, pop_size)
        tau1 = 0.1
        tau2 = 0.1
        while evals < self.budget:
            target = self.rng.randint(pop_size)
            old_F = F_pop[target]
            old_CR = CR_pop[target]
            if self.rng.rand() < tau1:
                F_pop[target] = 0.1 + self.rng.rand() * 0.9
            else:
                F_pop[target] = old_F
            if self.rng.rand() < tau2:
                CR_pop[target] = self.rng.rand()
            else:
                CR_pop[target] = old_CR
            F = F_pop[target]
            CR = CR_pop[target]
            indices = [i for i in range(pop_size) if i != target]
            if len(indices) < 3:
                F_pop[target] = old_F
                CR_pop[target] = old_CR
                continue
            a, b, c = self.rng.choice(indices, 3, replace=False)
            mutant = points[a] + F * (points[b] - points[c])
            trial = points[target].copy()
            j_rand = self.rng.randint(self.dim)
            for j in range(self.dim):
                if self.rng.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lb, ub)
            f_trial = func(trial)
            evals += 1
            if f_trial < best_f:
                best_f = f_trial
                best_x = trial.copy()
                report_best(best_f, best_x)
            if f_trial < pop_fitness[target]:
                points[target] = trial
                pop_fitness[target] = f_trial
            else:
                F_pop[target] = old_F
                CR_pop[target] = old_CR
        return best_f, best_x