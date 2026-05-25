import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(10*dim, budget // 2))
        if self.pop_size > budget:
            self.pop_size = budget

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = self.pop_size
        rng = self.rng
        budget = self.budget

        # Initialize population uniformly
        pop = lb + rng.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

        # Evaluate initial population
        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # Main loop
        while evals < budget:
            # Compute schedule based on fraction of budget used
            frac = evals / budget
            F = 0.9 - 0.8 * frac  # from 0.9 to 0.1
            CR = 0.1 + 0.8 * frac  # from 0.1 to 0.9

            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select three distinct indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]  # DE/rand/1

                # Mutation: DE/rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)

                # Exponential crossover
                j_start = rng.randint(dim)
                L = 1
                while rng.rand() < CR and L < dim:
                    L += 1
                trial = pop[i].copy()
                for k in range(L):
                    j = (j_start + k) % dim
                    trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)

                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        return best_val, best_x