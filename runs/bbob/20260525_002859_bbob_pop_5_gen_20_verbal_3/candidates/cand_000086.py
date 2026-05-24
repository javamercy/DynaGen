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
        popsize = min(budget, max(4, min(5 * dim, 20)))
        best_value = float('inf')
        best_x = None

        # initialize population
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fitness = np.full(popsize, np.inf)
        for i in range(popsize):
            val = func(pop[i])
            pop_fitness[i] = val
            evals += 1
            if val < best_value:
                best_value = val
                best_x = pop[i].copy()
                self.best_value = best_value
                self.best_x = best_x
                report_best(best_value, best_x)
        if evals >= budget:
            return best_value, best_x

        # initialize F and CR for each individual
        F = 0.5 * np.ones(popsize)
        CR = 0.9 * np.ones(popsize)
        tau1 = 0.1
        tau2 = 0.1
        F_l = 0.1
        F_u = 0.9

        no_improve_gen = 0
        while evals < budget:
            improved = False
            for i in range(popsize):
                if evals >= budget:
                    break
                # adapt F and CR
                if rng.rand() < tau1:
                    F[i] = F_l + rng.rand() * (F_u - F_l)
                if rng.rand() < tau2:
                    CR[i] = rng.rand()
                # select three distinct individuals different from i
                indices = list(range(popsize))
                indices.remove(i)
                rng.shuffle(indices)
                a, b, c = indices[:3]
                # mutation
                mutant = pop[a] + F[i] * (pop[b] - pop[c])
                # crossover
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]
                # bound clamping
                trial = np.clip(trial, lb, ub)
                # evaluation
                trial_fitness = func(trial)
                evals += 1
                # selection
                if trial_fitness <= pop_fitness[i]:
                    pop_fitness[i] = trial_fitness
                    pop[i] = trial
                    if trial_fitness < best_value:
                        best_value = trial_fitness
                        best_x = trial.copy()
                        self.best_value = best_value
                        self.best_x = best_x
                        report_best(best_value, best_x)
                        improved = True
            if not improved:
                no_improve_gen += 1
            else:
                no_improve_gen = 0
            if no_improve_gen >= dim and evals < budget:
                n_restart = popsize // 2
                if n_restart > 0:
                    # find best index
                    best_idx = np.argmin(pop_fitness)
                    for _ in range(n_restart):
                        if evals >= budget:
                            break
                        # select a random individual different from best
                        idx = rng.randint(popsize)
                        while idx == best_idx:
                            idx = rng.randint(popsize)
                        # reinitialize by perturbing best
                        delta = 0.1 * (ub - lb) * (2 * rng.rand(dim) - 1)
                        pop[idx] = np.clip(best_x + delta, lb, ub)
                        val = func(pop[idx])
                        evals += 1
                        pop_fitness[idx] = val
                        # reset F and CR for this individual
                        F[idx] = 0.5
                        CR[idx] = 0.9
                        if val < best_value:
                            best_value = val
                            best_x = pop[idx].copy()
                            self.best_value = best_value
                            self.best_x = best_x
                            report_best(best_value, best_x)
                no_improve_gen = 0
        return best_value, best_x