import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = float('inf')
        self.best_x = None

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        evals = 0

        # population size
        popsize = min(budget, max(4, min(4 * dim, 20)))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fitness = np.full(popsize, np.inf)

        # initial evaluations
        for i in range(popsize):
            pop_fitness[i] = func(pop[i])
            evals += 1
            if pop_fitness[i] < self.best_value:
                self.best_value = pop_fitness[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)

        if evals >= budget:
            return self.best_value, self.best_x

        stagnation_counter = 0
        while evals < budget:
            # one generation
            for i in range(popsize):
                if evals >= budget:
                    break
                # generate trial vector
                F = 0.5 + 0.5 * rng.rand()
                CR = 0.9
                # select three distinct random indices different from i
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                # binomial crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness <= pop_fitness[i]:
                    pop_fitness[i] = trial_fitness
                    pop[i] = trial
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1

            if evals >= budget:
                break

            # restart if stagnation
            if stagnation_counter >= dim:
                # sort by fitness
                sorted_indices = np.argsort(pop_fitness)
                worst_inds = sorted_indices[popsize // 2:]
                best_individual = self.best_x
                noise_scale = 0.2 * (ub - lb)
                for idx in worst_inds:
                    if evals >= budget:
                        break
                    new_point = best_individual + noise_scale * rng.randn(dim)
                    new_point = np.clip(new_point, lb, ub)
                    new_fitness = func(new_point)
                    evals += 1
                    pop[idx] = new_point
                    pop_fitness[idx] = new_fitness
                    if new_fitness < self.best_value:
                        self.best_value = new_fitness
                        self.best_x = new_point.copy()
                        report_best(self.best_value, self.best_x)
                stagnation_counter = 0

        return self.best_value, self.best_x