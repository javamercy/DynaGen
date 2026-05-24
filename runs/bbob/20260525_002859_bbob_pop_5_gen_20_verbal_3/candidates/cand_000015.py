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
        evals = 0

        # population size: max(4, min(5*dim, 20)), capped by budget
        popsize = min(budget, max(4, min(5*dim, 20)))
        if popsize < 4:
            popsize = 4
        # initialize population
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        fitness = np.full(popsize, np.inf)
        best_val = np.inf
        best_x = None

        # self-adaptive parameters per individual
        F = 0.5 + 0.5 * rng.rand(popsize)  # initial in [0.5, 1]
        CR = 0.5 * rng.rand(popsize)       # initial in [0, 0.5]
        tau_F = 0.1
        tau_CR = 0.1

        # initial evaluations
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

        # main DE loop
        while evals < budget:
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            new_F = F.copy()
            new_CR = CR.copy()

            for i in range(popsize):
                # generate new F and CR for this individual
                if rng.rand() < tau_F:
                    new_F[i] = 0.1 + 0.8 * rng.rand()  # uniform in [0.1, 0.9]
                else:
                    new_F[i] = F[i]
                if rng.rand() < tau_CR:
                    new_CR[i] = rng.rand()  # uniform in [0, 1]
                else:
                    new_CR[i] = CR[i]

                # select three distinct indices
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]

                # mutation
                mutant = pop[a] + new_F[i] * (pop[b] - pop[c])

                # binomial crossover
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < new_CR[i] or j == j_rand:
                        trial[j] = mutant[j]

                # clip to bounds
                trial = np.clip(trial, lb, ub)

                # evaluate
                val = func(trial)
                evals += 1

                if val <= new_fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = val
                    # accept new parameters
                    F[i] = new_F[i]
                    CR[i] = new_CR[i]
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                else:
                    # keep old parameters (already in F, CR)
                    pass

                if evals >= budget:
                    break

            if evals >= budget:
                break

            pop = new_pop
            fitness = new_fitness

        return best_val, best_x