import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Population size: at least 4, at most 10*dim, and not more than half budget
        self.pop_size = max(4, min(10*dim, budget // 2))
        if self.pop_size > budget:
            self.pop_size = budget
        self.F_low = 0.1
        self.F_high = 1.0
        self.tau_F = 0.1
        self.tau_CR = 0.1

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = self.pop_size
        rng = self.rng
        budget = self.budget

        # Initialize population within bounds
        pop = lb + rng.rand(pop_size, dim) * (ub - lb)
        # Initialize F and CR for each individual
        F = 0.5 + 0.5 * rng.rand(pop_size)  # uniform in [0.5, 1.0]
        CR = 0.5 * rng.rand(pop_size)        # uniform in [0, 0.5]
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

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

        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Generate new F and CR for this individual (self-adaptation)
                if rng.rand() < self.tau_F:
                    new_F = self.F_low + rng.rand() * (self.F_high - self.F_low)
                else:
                    new_F = F[i]
                if rng.rand() < self.tau_CR:
                    new_CR = rng.rand()
                else:
                    new_CR = CR[i]

                # Mutation: DE/current-to-best/1
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b = candidates[:2]  # only need two distinct indices
                mutant = pop[i] + new_F * (best_x - pop[i]) + new_F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)

                # Crossover: binomial with new_CR
                j_rand = rng.randint(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.rand() < new_CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop[i][j]
                trial = np.clip(trial, lb, ub)

                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    F[i] = new_F
                    CR[i] = new_CR
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        return best_val, best_x