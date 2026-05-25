import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        rng = self.rng
        budget = self.budget

        # Minimum budget: random search
        if budget < 3:
            while evals < budget:
                x = lb + (ub - lb) * rng.rand(self.dim)
                val = func(x)
                evals += 1
                if val < self.best_val:
                    self.best_val = val
                    self.best_x = x.copy()
                    report_best(self.best_val, self.best_x)
            return self.best_val, self.best_x

        # Population size
        pop_size = max(4, min(20, budget // 5))
        if pop_size > budget:
            pop_size = budget

        # Initialize population
        pop = lb + (ub - lb) * rng.rand(pop_size, self.dim)
        pop_fit = np.full(pop_size, np.inf)
        # Self-adaptive parameters
        F = 0.5 + 0.5 * rng.rand(pop_size)
        CR = 0.5 * rng.rand(pop_size)

        # Initial evaluations
        for i in range(pop_size):
            if evals >= budget:
                break
            pop_fit[i] = func(pop[i])
            evals += 1
            if pop_fit[i] < self.best_val:
                self.best_val = pop_fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)

        # Main loop
        while evals < budget:
            new_pop = pop.copy()
            new_F = F.copy()
            new_CR = CR.copy()
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Extract indices for mutation
                idxs = [j for j in range(pop_size) if j != i]
                a, b = rng.choice(idxs, 2, replace=False)
                # Self-adaptation of F and CR
                if rng.rand() < 0.1:
                    new_F[i] = 0.1 + 0.9 * rng.rand()
                else:
                    new_F[i] = F[i]
                if rng.rand() < 0.1:
                    new_CR[i] = rng.rand()
                else:
                    new_CR[i] = CR[i]
                # Current-to-best/1 mutation
                mutant = pop[i] + new_F[i] * (self.best_x - pop[i]) + new_F[i] * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                # Exponential crossover
                trial = pop[i].copy()
                n = rng.randint(self.dim)
                L = 0
                idx = n
                while L < self.dim and rng.rand() < new_CR[i]:
                    trial[idx] = mutant[idx]
                    idx = (idx + 1) % self.dim
                    L += 1
                trial_fit = func(trial)
                evals += 1
                if trial_fit < self.best_val:
                    self.best_val = trial_fit
                    self.best_x = trial.copy()
                    report_best(self.best_val, self.best_x)
                if trial_fit < pop_fit[i]:
                    new_pop[i] = trial
                    pop_fit[i] = trial_fit
                    new_F[i] = new_F[i]
                    new_CR[i] = new_CR[i]
                else:
                    new_F[i] = F[i]
                    new_CR[i] = CR[i]
            pop = new_pop
            F = new_F
            CR = new_CR
        return self.best_val, self.best_x