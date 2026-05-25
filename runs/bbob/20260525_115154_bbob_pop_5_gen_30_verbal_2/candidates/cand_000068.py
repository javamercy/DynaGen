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
        pop_size = min(max(4 * dim, 4), 100, budget // 2)
        if pop_size < 4:
            pop_size = min(4, budget)
        if pop_size < 3:
            pop_size = budget

        # initialize
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

        # DE parameters
        F = 0.7
        CR = 0.9
        max_gen = max(1, budget // pop_size)
        gen_no_improve_thresh = max(1, int(0.2 * max_gen))
        gen_no_improve = 0
        prev_best = best_val

        generation = 0
        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                # pbest selection (top 20%)
                sorted_idx = np.argsort(fitness)
                p_best_num = max(1, int(0.2 * pop_size))
                pbest_idx = rng.choice(sorted_idx[:p_best_num])

                # random a != i
                candidates = [j for j in range(pop_size) if j != i]
                a = rng.choice(candidates)
                # random b != i and != a
                b_options = [j for j in candidates if j != a]
                if len(b_options) == 0:
                    b = a
                else:
                    b = rng.choice(b_options)

                x_i = pop[i]
                x_pbest = pop[pbest_idx]
                x_a = pop[a]
                x_b = pop[b]

                # mutation: DE/rand-to-pbest/1
                mutant = x_i + F * (x_pbest - x_i) + F * (x_a - x_b)
                mutant = np.clip(mutant, lb, ub)

                # binomial crossover
                j_rand = rng.randint(dim)
                trial = x_i.copy()
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # evaluate
                trial_fit = func(trial)
                evals += 1

                if trial_fit < fitness[i]:
                    fitness[i] = trial_fit
                    pop[i] = trial
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # check improvement
            if best_val < prev_best:
                gen_no_improve = 0
                prev_best = best_val
            else:
                gen_no_improve += 1

            # restart if stuck
            if gen_no_improve >= gen_no_improve_thresh and evals < budget:
                # keep best, reinitialize others
                new_pop = np.empty_like(pop)
                new_fitness = np.full(pop_size, np.inf)
                new_pop[0] = best_x
                new_fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    new_pop[i] = rng.uniform(lb, ub, dim)
                    new_fitness[i] = func(new_pop[i])
                    evals += 1
                    if new_fitness[i] < best_val:
                        best_val = new_fitness[i]
                        best_x = new_pop[i].copy()
                        report_best(best_val, best_x)
                pop = new_pop
                fitness = new_fitness
                gen_no_improve = 0
                prev_best = best_val

            generation += 1

        return best_val, best_x