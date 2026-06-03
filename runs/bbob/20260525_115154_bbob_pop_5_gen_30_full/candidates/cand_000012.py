import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = max(4 * dim, 10)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)
        if pop_size < 3:
            pop_size = 3
        # mutation factor and crossover rate memory
        F = 0.5
        CR = 0.9
        successful_F = []
        successful_CR = []
        # initialize population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_val:
                best_val = fitness[i]
                best_x = pop[i].copy()
                report_best(best_val, best_x)
        stagnation = 0
        stagnation_limit = max(pop_size, budget // 20)
        while evals < budget:
            successful_F.clear()
            successful_CR.clear()
            for i in range(pop_size):
                if evals >= budget:
                    break
                # select three distinct random indices different from i
                indices = list(range(pop_size))
                indices.remove(i)
                r1, r2, r3 = rng.choice(indices, 3, replace=False)
                # sample F and CR
                Fi = rng.cauchy(F, 0.1)
                Fi = np.clip(Fi, 0, 1)
                CRi = rng.normal(CR, 0.1)
                CRi = np.clip(CRi, 0, 1)
                # mutant
                mutant = pop[r1] + Fi * (pop[r2] - pop[r3])
                # binomial crossover
                j_rand = rng.integers(dim)
                trial = np.where(rng.random(dim) < CRi, mutant, pop[i])
                # ensure at least one component from mutant
                trial[j_rand] = mutant[j_rand]
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    fitness[i] = val
                    pop[i] = trial
                    successful_F.append(Fi)
                    successful_CR.append(CRi)
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                else:
                    stagnation += 1
            # update F and CR based on successful values
            if successful_F:
                F = np.mean(successful_F)
                CR = np.mean(successful_CR)
            # restart if stagnation
            if evals >= budget:
                break
            if stagnation >= stagnation_limit:
                # replace worst half with random points
                worst_indices = np.argsort(fitness)[-pop_size//2:]
                for idx in worst_indices:
                    if evals >= budget:
                        break
                    new_x = rng.uniform(lb, ub)
                    new_val = func(new_x)
                    evals += 1
                    fitness[idx] = new_val
                    pop[idx] = new_x
                    if new_val < best_val:
                        best_val = new_val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)
                stagnation = 0
                # reset F and CR
                F = 0.5
                CR = 0.9
        return best_val, best_x