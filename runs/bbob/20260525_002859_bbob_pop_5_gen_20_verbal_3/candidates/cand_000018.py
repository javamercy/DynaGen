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
        # Adaptive parameters
        mu_F = 0.5
        mu_CR = 0.5
        memory_size = 5
        F_memory = []
        CR_memory = []
        gen_since_improve = 0
        stagnation_limit = max(1, int(np.sqrt(budget / popsize)))
        while evals < budget:
            improvement = False
            for i in range(popsize):
                # Generate F and CR
                F = rng.uniform(0, 1)  # simplified: uniform random
                CR = rng.uniform(0, 1)
                # Mutation current-to-best/1
                best_idx = np.argmin(pop_fitness)
                candidates = list(range(popsize))
                candidates.remove(i)
                if best_idx in candidates:
                    candidates.remove(best_idx)
                rng.shuffle(candidates)
                a, b = candidates[:2]
                mutant = pop[i] + F * (pop[best_idx] - pop[i]) + F * (pop[a] - pop[b])
                # Crossover
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness <= pop_fitness[i]:
                    pop_fitness[i] = trial_fitness
                    pop[i] = trial
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        improvement = True
                if evals >= budget:
                    break
            if improvement:
                gen_since_improve = 0
            else:
                gen_since_improve += 1
            if gen_since_improve >= stagnation_limit and evals < budget:
                # Restart: reinitialize all except best
                best_idx = np.argmin(pop_fitness)
                for i in range(popsize):
                    if i != best_idx:
                        pop[i] = lb + (ub - lb) * rng.rand(dim)
                        pop_fitness[i] = func(pop[i])
                        evals += 1
                        if pop_fitness[i] < self.best_value:
                            self.best_value = pop_fitness[i]
                            self.best_x = pop[i].copy()
                            report_best(self.best_value, self.best_x)
                        if evals >= budget:
                            break
                gen_since_improve = 0
        return self.best_value, self.best_x