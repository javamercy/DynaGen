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

        # Initialize best with random point
        best_x = rng.uniform(lb, ub)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        # Population size
        NP = max(4, 2 * dim)
        if evals + NP > budget:
            NP = budget - evals

        if NP > 0:
            pop = rng.uniform(lb, ub, size=(NP, dim))
            fitness = np.full(NP, np.inf)
            for i in range(NP):
                if evals >= budget:
                    break
                val = func(pop[i])
                evals += 1
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_x = pop[i].copy()
                    report_best(best_val, best_x)

            # DE parameters
            F = 0.8
            CR = 0.9

            while evals < budget:
                for i in range(NP):
                    if evals >= budget:
                        break
                    # Mutation: select three distinct random indices different from i
                    indices = list(range(NP))
                    indices.remove(i)
                    rng.shuffle(indices)
                    a, b, c = indices[0], indices[1], indices[2]
                    mutant = pop[a] + F * (pop[b] - pop[c])
                    # Crossover
                    trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                    j_rand = rng.randint(dim)
                    trial[j_rand] = mutant[j_rand]
                    trial = np.clip(trial, lb, ub)
                    # Evaluation
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