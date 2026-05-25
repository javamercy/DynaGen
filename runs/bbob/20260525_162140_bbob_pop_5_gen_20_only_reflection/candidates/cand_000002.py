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

        # Population size
        NP = min(10 * dim, max(4, budget // 2 - 1))
        if NP < 4:
            NP = 4
        # Ensure budget can accommodate initial population and at least one generation
        if budget < NP:
            NP = budget  # degenerate case: just random search
        
        # Initialize population
        pop = lb + (ub - lb) * rng.rand(NP, dim)
        fitness = np.zeros(NP)
        for i in range(NP):
            fitness[i] = func(pop[i])
        
        # Initial best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_val = fitness[best_idx]
        report_best(best_val, best_x)
        
        func_evals = NP
        
        # Main DE loop
        if NP > 1:
            F = 0.8
            CR = 0.9
            max_gen = (budget - func_evals) // NP
            for gen in range(max_gen):
                for i in range(NP):
                    # Select three distinct random indices
                    candidates = list(range(NP))
                    candidates.remove(i)
                    idx = rng.choice(candidates, 3, replace=False)
                    a, b, c = pop[idx[0]], pop[idx[1]], pop[idx[2]]
                    mutant = a + F * (b - c)
                    # Binomial crossover
                    trial = pop[i].copy()
                    j_rand = rng.randint(dim)
                    for j in range(dim):
                        if rng.rand() < CR or j == j_rand:
                            trial[j] = mutant[j]
                    # Clip to bounds
                    trial = np.clip(trial, lb, ub)
                    # Evaluate
                    trial_fitness = func(trial)
                    func_evals += 1
                    if trial_fitness < fitness[i]:
                        pop[i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < best_val:
                            best_val = trial_fitness
                            best_x = trial.copy()
                            report_best(best_val, best_x)
                    if func_evals >= budget:
                        break
                if func_evals >= budget:
                    break
        return best_val, best_x