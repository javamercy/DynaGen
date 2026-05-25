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
        budget = self.budget
        rng = self.rng

        if budget < 4:
            while evals < budget:
                x = lb + (ub - lb) * rng.rand(self.dim)
                val = func(x)
                evals += 1
                if val < self.best_val:
                    self.best_val = val
                    self.best_x = x.copy()
                    report_best(self.best_val, self.best_x)
            return self.best_val, self.best_x

        pop_size = max(4, min(20, budget // 5))
        F = 0.8
        CR = 0.9
        stagnation_limit = max(1, budget // 10)

        # Initialize population
        pop = lb + (ub - lb) * rng.rand(pop_size, self.dim)
        pop_fit = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= budget:
                break
            pop_fit[i] = func(pop[i])
            evals += 1
            if pop_fit[i] < self.best_val:
                self.best_val = pop_fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)

        no_improve_evals = 0

        while evals < budget:
            # Restart if stagnation
            if no_improve_evals >= stagnation_limit:
                pop = lb + (ub - lb) * rng.rand(pop_size, self.dim)
                pop[0] = self.best_x.copy()
                pop_fit = np.full(pop_size, np.inf)
                pop_fit[0] = self.best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    pop_fit[i] = func(pop[i])
                    evals += 1
                    if pop_fit[i] < self.best_val:
                        self.best_val = pop_fit[i]
                        self.best_x = pop[i].copy()
                        report_best(self.best_val, self.best_x)
                        no_improve_evals = 0
                    else:
                        no_improve_evals += 1
                no_improve_evals = 0
                continue

            new_pop = pop.copy()
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select two distinct random indices not equal to i
                idxs = [j for j in range(pop_size) if j != i]
                a, b = rng.choice(idxs, 2, replace=False)
                # current-to-best/1 mutation
                mutant = pop[i] + F * (self.best_x - pop[i]) + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                # Exponential crossover
                trial = pop[i].copy()
                start = rng.randint(self.dim)
                for k in range(self.dim):
                    idx = (start + k) % self.dim
                    if rng.rand() < CR:
                        trial[idx] = mutant[idx]
                    else:
                        break
                trial_fit = func(trial)
                evals += 1
                if trial_fit < self.best_val:
                    self.best_val = trial_fit
                    self.best_x = trial.copy()
                    report_best(self.best_val, self.best_x)
                    no_improve_evals = 0
                else:
                    no_improve_evals += 1
                # Greedy selection
                if trial_fit < pop_fit[i]:
                    new_pop[i] = trial
                    pop_fit[i] = trial_fit
            pop = new_pop

        return self.best_val, self.best_x