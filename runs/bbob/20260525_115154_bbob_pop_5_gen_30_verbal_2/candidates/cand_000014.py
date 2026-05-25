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

        # Determine population size
        pop_size = max(4, min(4 * dim, budget // 2))
        if pop_size > budget:
            pop_size = budget

        # Fallback to random search if population is too small for DE
        if pop_size < 4:
            best_val = np.inf
            best_x = None
            evals = 0
            for _ in range(budget):
                x = rng.uniform(lb, ub, size=dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        # Initialize population
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
        F = 0.8
        CR = 0.9
        max_stagnation_generations = 20
        stagnation_counter = 0
        prev_best_val = best_val

        # Main DE loop
        generation = 0
        while evals < budget:
            # Check stagnation and restart if needed
            if generation > 0 and stagnation_counter >= max_stagnation_generations:
                # Restart: keep best point, reinitialize rest randomly
                # Evaluate the best point? Already evaluated, so keep as is
                new_pop = np.empty((pop_size, dim))
                new_fitness = np.empty(pop_size)
                new_pop[0] = best_x.copy()
                new_fitness[0] = best_val
                for i in range(1, pop_size):
                    new_pop[i] = rng.uniform(lb, ub, size=dim)
                    if evals >= budget:
                        break
                    new_fitness[i] = func(new_pop[i])
                    evals += 1
                    if new_fitness[i] < best_val:
                        best_val = new_fitness[i]
                        best_x = new_pop[i].copy()
                        report_best(best_val, best_x)
                # If budget exhausted during restart, break
                if evals >= budget:
                    pop = new_pop[:pop_size]
                    fitness = new_fitness[:pop_size]
                    break
                pop = new_pop
                fitness = new_fitness
                stagnation_counter = 0
                prev_best_val = best_val
                generation = 0
                continue

            # One generation of DE
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select three distinct indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                ids = rng.choice(candidates, size=3, replace=False)
                a, b, c = ids
                # Mutant
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    fitness[i] = trial_fit
                    pop[i] = trial
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # After generation, update stagnation counter
            if best_val < prev_best_val:
                stagnation_counter = 0
                prev_best_val = best_val
            else:
                stagnation_counter += 1
            generation += 1

        return best_val, best_x