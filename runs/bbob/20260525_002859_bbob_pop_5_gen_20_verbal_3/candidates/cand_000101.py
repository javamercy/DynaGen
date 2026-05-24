import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = float('inf')
        self.best_x = None

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget must be positive")
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
            if pop_fitness[i] < self.best_value:
                self.best_value = pop_fitness[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x
        H = 5
        F_memory = np.ones(H) * 0.5
        CR_memory = np.ones(H) * 0.5
        memory_counter = 0
        stagnation_generations = 0
        max_stagnation = max(10, int(budget / (2 * popsize)))
        best_so_far = self.best_value
        while evals < budget:
            F_success = []
            CR_success = []
            improved = False
            for i in range(popsize):
                r = rng.randint(H)
                F = F_memory[r] + 0.1 * rng.randn()
                CR = CR_memory[r] + 0.1 * rng.randn()
                F = np.clip(F, 0, 1)
                CR = np.clip(CR, 0, 1)
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                trial = pop[i].copy()
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
                        improved = True
                    F_success.append(F)
                    CR_success.append(CR)
                if evals >= budget:
                    break
            if evals >= budget:
                break
            if len(F_success) > 0:
                sum_F = np.sum(F_success)
                sum_F2 = np.sum(np.square(F_success))
                mean_F = sum_F2 / sum_F if sum_F > 0 else 0.5
                mean_CR = np.mean(CR_success)
                F_memory[memory_counter % H] = mean_F
                CR_memory[memory_counter % H] = mean_CR
                memory_counter += 1
            if improved:
                stagnation_generations = 0
                best_so_far = self.best_value
            else:
                stagnation_generations += 1
            # Restart if stagnation
            if stagnation_generations >= max_stagnation and evals < budget:
                # Increase population size, but ensure we don't exceed budget
                new_popsize = min(int(popsize * 1.5), budget - evals - 1)
                if new_popsize > popsize:
                    popsize = new_popsize
                # Reinitialize population
                pop = lb + (ub - lb) * rng.rand(popsize, dim)
                pop_fitness = np.zeros(popsize)
                for i in range(popsize):
                    pop_fitness[i] = func(pop[i])
                    evals += 1
                    if pop_fitness[i] < self.best_value:
                        self.best_value = pop_fitness[i]
                        self.best_x = pop[i].copy()
                        report_best(self.best_value, self.best_x)
                    if evals >= budget:
                        break
                if evals >= budget:
                    break
                # Reset memory
                F_memory = np.ones(H) * 0.5
                CR_memory = np.ones(H) * 0.5
                memory_counter = 0
                stagnation_generations = 0
                max_stagnation = max(10, int(budget / (2 * popsize)))
        return self.best_value, self.best_x