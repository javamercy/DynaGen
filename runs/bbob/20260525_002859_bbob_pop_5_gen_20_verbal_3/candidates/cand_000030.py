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

        # Smaller population for exploitation
        popsize = max(4, min(2 * dim, 10))
        if popsize > budget // 2:
            popsize = max(4, budget // 2)
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        fitness = np.full(popsize, np.inf)
        best_val = np.inf
        best_x = None

        # Self-adaptive parameters biased for exploitation
        F = 0.2 + 0.2 * rng.rand(popsize)  # initial in [0.2, 0.4]
        CR = 0.1 * rng.rand(popsize)        # initial in [0, 0.1]
        tau_F = 0.1
        tau_CR = 0.1

        evals = 0

        # Initial evaluations
        for i in range(popsize):
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)
            if evals >= budget:
                return best_val, best_x

        # Main DE loop
        while evals < budget:
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            new_F = F.copy()
            new_CR = CR.copy()

            for i in range(popsize):
                # Update F and CR with exploitation-friendly ranges
                if rng.rand() < tau_F:
                    new_F[i] = 0.1 + 0.4 * rng.rand()  # [0.1, 0.5]
                else:
                    new_F[i] = F[i]
                if rng.rand() < tau_CR:
                    new_CR[i] = 0.3 * rng.rand()  # [0, 0.3]
                else:
                    new_CR[i] = CR[i]

                # DE/best/1 mutation
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b = candidates[:2]
                mutant = best_x + new_F[i] * (pop[a] - pop[b])

                # Binomial crossover
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < new_CR[i] or j == j_rand:
                        trial[j] = mutant[j]

                # Bounds enforcement
                trial = np.clip(trial, lb, ub)

                # Evaluation
                val = func(trial)
                evals += 1

                # Selection (greedy)
                if val <= new_fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = val
                    F[i] = new_F[i]
                    CR[i] = new_CR[i]
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                else:
                    # Keep old parameters
                    pass

                if evals >= budget:
                    break

            if evals >= budget:
                break

            pop = new_pop
            fitness = new_fitness

        # Final local search near best
        while evals < budget:
            std = 0.01 * (ub - lb)
            candidate = best_x + rng.randn(dim) * std
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = candidate.copy()
                report_best(best_val, best_x)

        return best_val, best_x