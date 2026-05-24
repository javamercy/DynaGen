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
        popsize = min(budget, max(4, min(6*dim, 30)))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fitness = np.zeros(popsize)
        for i in range(popsize):
            pop_fitness[i] = func(pop[i])
            evals += 1
            if evals == 1 or pop_fitness[i] < self.best_value:
                self.best_value = pop_fitness[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x

        archive = []
        max_archive = 2 * popsize
        CR = 0.95
        stagnation_counter = 0
        max_stagnation = max(5, int(budget / (2 * popsize)))

        while evals < budget:
            improved_this_gen = False
            for i in range(popsize):
                if evals >= budget:
                    break
                F = 0.8 + 0.4 * rng.rand()
                # current-to-rand/1: no pbest bias
                candidates_r1 = list(range(popsize))
                candidates_r1.remove(i)
                r1 = rng.choice(candidates_r1)
                combined = list(range(popsize)) + list(range(len(archive)))
                union_points = [pop[j] for j in range(popsize)] + archive
                exclude_set = {i, r1}
                r2_idx = rng.randint(0, len(union_points))
                while r2_idx in exclude_set:
                    r2_idx = rng.randint(0, len(union_points))
                r2 = union_points[r2_idx]
                r3_idx = rng.randint(0, len(union_points))
                while r3_idx in exclude_set or r3_idx == r2_idx:
                    r3_idx = rng.randint(0, len(union_points))
                r3 = union_points[r3_idx]
                mutant = pop[i] + F * (pop[r1] - pop[i]) + F * (r2 - r3)
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness <= pop_fitness[i]:
                    old_x = pop[i].copy()
                    pop_fitness[i] = trial_fitness
                    pop[i] = trial
                    if not np.array_equal(old_x, self.best_x) and len(archive) < max_archive:
                        archive.append(old_x)
                    elif len(archive) >= max_archive:
                        idx = rng.randint(0, max_archive)
                        archive[idx] = old_x
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        improved_this_gen = True
            if evals >= budget:
                break
            if improved_this_gen:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            if stagnation_counter >= max_stagnation and evals + popsize - 1 <= budget:
                new_pop = np.zeros((popsize, dim))
                new_fitness = np.zeros(popsize)
                new_pop[0] = self.best_x
                new_fitness[0] = self.best_value
                n_gaussian = popsize // 2
                for i in range(1, n_gaussian):
                    x = self.best_x + 0.2 * (ub - lb) * rng.randn(dim)
                    x = np.clip(x, lb, ub)
                    f = func(x)
                    evals += 1
                    new_pop[i] = x
                    new_fitness[i] = f
                    if f < self.best_value:
                        self.best_value = f
                        self.best_x = x.copy()
                        report_best(self.best_value, self.best_x)
                for i in range(n_gaussian, popsize):
                    x = lb + (ub - lb) * rng.rand(dim)
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
                archive = []
                stagnation_counter = 0
                if evals >= budget:
                    break
        return self.best_value, self.best_x