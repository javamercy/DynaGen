import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
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
        pop_fitness = np.zeros(popsize)
        for i in range(popsize):
            pop_fitness[i] = func(pop[i])
            evals += 1
            if evals == 1 or pop_fitness[i] < self.best_value:
                self.best_value = pop_fitness[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x
        
        # memory for F and CR
        F_mean = 0.5
        CR_mean = 0.9
        archive_F = []
        archive_CR = []
        
        evals_since_improvement = 0
        gen = 0
        while evals < budget:
            successful_F = []
            successful_CR = []
            for i in range(popsize):
                # sample F and CR from Cauchy and Gaussian
                F_i = rng.standard_cauchy() * 0.1 + F_mean
                F_i = np.clip(F_i, 0.0, 2.0)
                CR_i = rng.randn() * 0.1 + CR_mean
                CR_i = np.clip(CR_i, 0.0, 1.0)
                
                # current-to-best/1
                best_idx = np.argmin(pop_fitness)
                candidates = list(range(popsize))
                candidates.remove(best_idx)
                candidates.remove(i)
                if len(candidates) < 2:
                    # not enough diversity, fallback to random
                    r1, r2 = rng.randint(popsize, size=2)
                    while r1 == i or r2 == i or r1 == r2:
                        r1 = rng.randint(popsize)
                        r2 = rng.randint(popsize)
                else:
                    r1, r2 = rng.choice(candidates, 2, replace=False)
                mutant = pop[i] + F_i * (pop[best_idx] - pop[i]) + F_i * (pop[r1] - pop[r2])
                
                # binomial crossover
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR_i or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                
                trial_fitness = func(trial)
                evals += 1
                evals_since_improvement += 1
                if trial_fitness <= pop_fitness[i]:
                    pop_fitness[i] = trial_fitness
                    pop[i] = trial
                    successful_F.append(F_i)
                    successful_CR.append(CR_i)
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        evals_since_improvement = 0
                if evals >= budget:
                    break
            if evals >= budget:
                break
            # update memory with successful parameters
            if len(successful_F) > 0:
                F_mean = (1 - 0.1) * F_mean + 0.1 * np.mean(successful_F)
                CR_mean = (1 - 0.1) * CR_mean + 0.1 * np.mean(successful_CR)
            # stagnation restart
            if evals_since_improvement > 0.2 * budget:
                # reinitialize all but best
                for i in range(popsize):
                    if i != best_idx:
                        pop[i] = lb + (ub - lb) * rng.rand(dim)
                        pop_fitness[i] = func(pop[i])
                        evals += 1
                        if pop_fitness[i] < self.best_value:
                            self.best_value = pop_fitness[i]
                            self.best_x = pop[i].copy()
                            report_best(self.best_value, self.best_x)
                            evals_since_improvement = 0
                        if evals >= budget:
                            break
                if evals >= budget:
                    break
                evals_since_improvement = 0
                # reset memory to default
                F_mean = 0.5
                CR_mean = 0.9
            gen += 1
        return self.best_value, self.best_x