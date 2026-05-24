import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = None
        self.best_x = None

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        evals = 0
        popsize = min(budget, max(4, min(5*dim, 20)))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        fit = np.zeros(popsize)
        for i in range(popsize):
            fit[i] = func(pop[i])
            evals += 1
            if evals == 1 or fit[i] < self.best_value:
                self.best_value = fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x

        # Adaptive parameter memory
        memory_size = 10
        F_memory = [0.8] * memory_size
        CR_memory = [0.8] * memory_size
        memory_idx = 0
        F = 0.8
        CR = 0.9

        stagnation_limit = max(10 * dim, int(0.2 * budget))
        stagnation_counter = 0

        while evals < budget:
            # Create arrays to store successful parameters
            successful_F = []
            successful_CR = []
            for i in range(popsize):
                if evals >= budget:
                    break
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2 = candidates[0], candidates[1]
                mutant = pop[i] + F * (self.best_x - pop[i]) + F * (pop[r1] - pop[r2])
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fit = func(trial)
                evals += 1
                if trial_fit <= fit[i]:
                    fit[i] = trial_fit
                    pop[i] = trial
                    successful_F.append(F)
                    successful_CR.append(CR)
                    if trial_fit < self.best_value:
                        self.best_value = trial_fit
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1
                if evals >= budget:
                    break
            if evals >= budget:
                break
            # Update F and CR with successful parameters
            if len(successful_F) > 0:
                # Lehmer mean for F
                F_new = np.sum(np.array(successful_F)**2) / np.sum(successful_F)
                # Arithmetic mean for CR
                CR_new = np.mean(successful_CR)
                F_memory[memory_idx] = min(F_new, 0.9)
                CR_memory[memory_idx] = min(CR_new, 1.0)
                memory_idx = (memory_idx + 1) % memory_size
                F = np.mean(F_memory)
                CR = np.mean(CR_memory)
            # Restart if stagnation and enough budget left
            if stagnation_counter >= stagnation_limit and evals < budget - popsize:
                stagnation_counter = 0
                # Keep best individual
                new_pop = [self.best_x.copy()]
                # Generate new individuals
                for _ in range(popsize - 1):
                    if rng.rand() < 0.5:
                        sigma = (ub - lb) * 0.2 * (1 + rng.rand())
                        new_x = self.best_x + rng.randn(dim) * sigma
                        new_x = np.clip(new_x, lb, ub)
                    else:
                        new_x = lb + (ub - lb) * rng.rand(dim)
                    new_pop.append(new_x)
                # Evaluate new individuals
                for j, x in enumerate(new_pop[1:], start=1):
                    if evals >= budget:
                        break
                    fit_val = func(x)
                    evals += 1
                    pop[j] = x
                    fit[j] = fit_val
                    if fit_val < self.best_value:
                        self.best_value = fit_val
                        self.best_x = x.copy()
                        report_best(self.best_value, self.best_x)
                # Reset adaptive parameters
                F = 0.8
                CR = 0.9
                F_memory = [0.8] * memory_size
                CR_memory = [0.9] * memory_size
                memory_idx = 0
        return self.best_value, self.best_x