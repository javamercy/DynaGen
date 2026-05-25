import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng
        budget = self.budget

        # population size: proportional to dim, but not too large
        pop_size = max(4, min(10 * dim, budget // 3))
        if pop_size > budget:
            pop_size = budget

        # initialize population
        pop = rng.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        # initialize F and CR for each individual
        F = 0.5 * rng.rand(pop_size) + 0.5  # [0.5, 1.0]
        CR = 0.1 * rng.rand(pop_size) + 0.8  # [0.8, 0.9] but actually 0.1*r + 0.8 gives [0.8,0.9]

        best_x = None
        best_val = np.inf
        evals = 0

        # initial evaluation
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

        # parameters for adaptation
        tau1 = 0.1
        tau2 = 0.1
        Fl = 0.1
        Fu = 0.9

        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                # generate new F and CR for this individual
                new_F = F[i] + tau1 * rng.uniform(-1, 1)
                new_F = max(Fl, min(Fu, new_F))
                new_CR = CR[i] + tau2 * rng.uniform(-1, 1)
                new_CR = max(0, min(1, new_CR))

                # mutation: DE/rand/1
                indices = list(range(pop_size))
                indices.remove(i)
                rng.shuffle(indices)
                a, b, c = indices[:3]
                mutant = pop[a] + new_F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)

                # binomial crossover
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