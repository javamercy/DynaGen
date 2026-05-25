import numpy as np
from numpy.random import RandomState

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub

        # population size
        pop_size = max(4 * dim, 10)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)

        # initialize population and adaptive parameters
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        F = rng.uniform(0.4, 0.9, size=pop_size)
        CR = rng.uniform(0.1, 0.9, size=pop_size)
        best_val = np.inf
        best_x = np.zeros(dim)
        evals = 0

        # initial evaluation
        for i in range(pop_size):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_val:
                best_val = fitness[i]
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # stagnation tracking
        no_improve_evals = 0
        stagnation_limit = max(10, budget // 50)
        last_best = best_val

        while evals < budget:
            # DE generation
            for i in range(pop_size):
                if evals >= budget:
                    break
                # select three distinct indices
                candidates = list(range(pop_size))
                candidates.remove(i)
                idx = rng.choice(candidates, size=3, replace=False)
                a, b, c = idx
                # mutation
                mutant = pop[a] + F[i] * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR[i], mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    fitness[i] = trial_fit
                    pop[i] = trial
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                else:
                    # reduce F and CR for unsuccessful
                    F[i] = rng.uniform(0.4, 0.9)
                    CR[i] = rng.uniform(0.1, 0.9)

            # check improvement
            if best_val < last_best:
                no_improve_evals = 0
                last_best = best_val
                # adapt successful F, CR? Not done here to keep simple
            else:
                no_improve_evals += pop_size

            # restart if stagnation
            if no_improve_evals >= stagnation_limit and evals < budget - 2:
                # reinitialize population except best
                # compute range for perturbation
                scale = 0.2 * (ub - lb)
                for i in range(pop_size):
                    if rng.rand() < 0.9:  # 90% probability to replace
                        # sample near best
                        new_x = best_x + scale * rng.randn(dim)
                        new_x = np.clip(new_x, lb, ub)
                        new_fit = func(new_x)
                        evals += 1
                        pop[i] = new_x
                        fitness[i] = new_fit
                        if new_fit < best_val:
                            best_val = new_fit
                            best_x = new_x.copy()
                            report_best(best_val, best_x)
                        if evals >= budget:
                            break
                        # reinitialize F and CR
                        F[i] = rng.uniform(0.4, 0.9)
                        CR[i] = rng.uniform(0.1, 0.9)
                no_improve_evals = 0

        return best_val, best_x