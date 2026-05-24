import numpy as np
class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = None
        self.best_x = None

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        evals = 0
        popsize = min(budget, max(4, min(5*dim, 100)))
        # initial population
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fitness = np.full(popsize, np.inf)
        for i in range(popsize):
            pop_fitness[i] = func(pop[i])
            evals += 1
            if evals == 1 or pop_fitness[i] < self.best_value:
                self.best_value = pop_fitness[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x
        # initialize F and CR for each individual
        F = np.full(popsize, 0.5)
        CR = np.full(popsize, 0.9)
        tau1 = 0.1
        tau2 = 0.1
        stagnation_counter = 0
        max_stagnation = max(5, int(budget / (2 * popsize)))
        while evals < budget:
            improved_this_gen = False
            for i in range(popsize):
                if evals >= budget:
                    break
                # mutation: current-to-best/1
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2 = candidates[0], candidates[1]
                mutant = pop[i] + F[i] * (self.best_x - pop[i]) + F[i] * (pop[r1] - pop[r2])
                # binomial crossover
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                # evaluation
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness <= pop_fitness[i]:
                    pop_fitness[i] = trial_fitness
                    pop[i] = trial
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        improved_this_gen = True
                    # update F, CR for successful individual? Not in jDE original, but we keep adaptation
                # jDE parameter adaptation (regardless of success)
                if rng.rand() < tau1:
                    F[i] = 0.1 + 0.9 * rng.rand()
                if rng.rand() < tau2:
                    CR[i] = rng.rand()
            if evals >= budget:
                break
            if improved_this_gen:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            # restart condition
            if stagnation_counter >= max_stagnation and evals < budget:
                # keep best, reinitialize rest
                new_pop = np.zeros((popsize, dim))
                new_fitness = np.full(popsize, np.inf)
                new_pop[0] = self.best_x
                new_fitness[0] = self.best_value
                for i in range(1, popsize):
                    x = lb + (ub - lb) * rng.rand(dim)
                    x = np.clip(x, lb, ub)
                    f = func(x)
                    evals += 1
                    new_pop[i] = x
                    new_fitness[i] = f
                    if f < self.best_value:
                        self.best_value = f
                        self.best_x = x.copy()
                        report_best(self.best_value, self.best_x)
                pop = new_pop
                pop_fitness = new_fitness
                # reset F and CR for new population
                F[:] = 0.5
                CR[:] = 0.9
                stagnation_counter = 0
                # local search around best after restart
                initial_scale = 0.1 * (ub - lb)
                num_local = min(budget - evals, dim * 2)
                for _ in range(num_local):
                    if evals >= budget:
                        break
                    direction = rng.randn(dim)
                    direction = direction / (np.linalg.norm(direction) + 1e-20)
                    step = rng.uniform(0, 1) * initial_scale
                    trial = self.best_x + step * direction
                    trial = np.clip(trial, lb, ub)
                    f = func(trial)
                    evals += 1
                    if f < self.best_value:
                        self.best_value = f
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                if evals >= budget:
                    break
        return self.best_value, self.best_x