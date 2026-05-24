import numpy as np
import math

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

        # initial population size
        popsize = max(4, min(5 * dim, budget // 2))
        if popsize > budget:
            popsize = budget

        # initialize population
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        fitness = np.full(popsize, np.inf)
        best_val = np.inf
        best_x = None

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

        # archives for successful parameters
        F_archive = []
        CR_archive = []

        while evals < budget:
            # draw F and CR
            if len(F_archive) > 0:
                F = rng.choice(F_archive)
            else:
                F = 0.5 + 0.1 * rng.randn()
                F = np.clip(F, 0.1, 0.9)
            if len(CR_archive) > 0:
                CR = rng.choice(CR_archive)
            else:
                CR = 0.9 + 0.1 * rng.randn()
                CR = np.clip(CR, 0.0, 1.0)

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            num_success = 0

            for i in range(popsize):
                # mutation type selection: 50% current-to-best/1, 50% rand/1
                if rng.rand() < 0.5:
                    # current-to-best/1
                    # select two distinct random indices != i
                    candidates = list(range(popsize))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    r1, r2 = candidates[:2]
                    mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
                else:
                    # rand/1
                    candidates = list(range(popsize))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    a, b, c = candidates[:3]
                    mutant = pop[a] + F * (pop[b] - pop[c])

                # binomial crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)

                val = func(trial)
                evals += 1
                if val <= new_fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                    F_archive.append(F)
                    CR_archive.append(CR)
                    num_success += 1
                    # limit archive size
                    if len(F_archive) > popsize:
                        F_archive.pop(0)
                        CR_archive.pop(0)
                if evals >= budget:
                    break
            if evals >= budget:
                break

            # update population
            pop = new_pop
            fitness = new_fitness

            # adjust population size based on success rate
            success_rate = num_success / popsize
            if success_rate > 0.5 and popsize < budget // 2:
                popsize = min(popsize + 1, budget // 2)
                # increase pop by adding random points
                extra = popsize - len(pop)
                if extra > 0:
                    new_individuals = lb + (ub - lb) * rng.rand(extra, dim)
                    pop = np.vstack([pop, new_individuals])
                    fitness = np.hstack([fitness, np.full(extra, np.inf)])
                    for j in range(extra):
                        val = func(pop[-(extra - j)])
                        evals += 1
                        fitness[-(extra - j)] = val
                        if val < best_val:
                            best_val = val
                            best_x = pop[-(extra - j)].copy()
                            report_best(best_val, best_x)
                        if evals >= budget:
                            break
            elif success_rate < 0.25 and popsize > 4:
                popsize = max(popsize - 1, 4)
                # remove worst individuals
                worst_indices = np.argsort(fitness)[-1:]  # remove one worst
                pop = np.delete(pop, worst_indices, axis=0)
                fitness = np.delete(fitness, worst_indices)

            if evals >= budget:
                break

        return best_val, best_x