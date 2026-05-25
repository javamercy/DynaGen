import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Population size
        pop_size = min(max(4 * dim, 10), budget // 2)
        pop_size = max(pop_size, 3)  # at least 3 for mutation

        # Initialize population uniformly
        pop = lb + rng.uniform(size=(pop_size, dim)) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        evals = 0
        best_x = None
        best_f = np.inf

        for i in range(pop_size):
            if evals >= budget:
                break
            f = func(pop[i])
            evals += 1
            fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = pop[i].copy()
                report_best(best_f, best_x)

        # Main DE loop
        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break

                # Choose three distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)

                # Random F and CR per individual
                F = rng.uniform(0.5, 1.0)
                CR = rng.uniform(0.3, 0.9)

                # Mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                # Bound handling: clip
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.uniform() < CR or j == j_rand:
                        trial[j] = mutant[j]

                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)

        return best_f, best_x