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
        popsize = min(budget, max(4, min(5*dim, 20)))
        # initial population
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

        # archive for inferior solutions
        archive = []
        max_archive = popsize
        CR = 0.9
        p = 0.2  # pbest proportion
        stagnation_counter = 0
        max_stagnation = max(5, int(budget / (2 * popsize)))

        while evals < budget:
            # sort population by fitness for pbest selection
            sorted_indices = np.argsort(pop_fitness)
            pbest_size = max(1, int(p * popsize))
            pbest_indices = sorted_indices[:pbest_size]
            improved_this_gen = False
            # generate a single F for this generation (jittered)
            F = 0.7 + 0.3 * rng.rand()
            for i in range(popsize):
                if evals >= budget:
                    break
                # mutation
                # pbest selection
                pbest_idx = rng.choice(pbest_indices)
                pbest_x = pop[pbest_idx]
                # select r1 from population excluding i
                candidates_r1 = list(range(popsize))
                candidates_r1.remove(i)
                r1 = rng.choice(candidates_r1)
                # select r2 from pop union archive, excluding i and r1
                combined = list(range(popsize)) + list(range(len(archive)))
                # map archive indices to negative or large numbers? Use separate list
                # easier: create list of points
                union_points = [pop[j] for j in range(popsize)] + archive
                # indices in union: 0..popsize-1 for pop, popsize.. for archive
                # exclude indices corresponding to i and r1
                exclude_set = {i, r1}
                r2_idx = rng.randint(0, len(union_points))
                while r2_idx in exclude_set:
                    r2_idx = rng.randint(0, len(union_points))
                r2 = union_points[r2_idx]
                # mutant
                mutant = pop[i] + F * (pbest_x - pop[i]) + F * (pop[r1] - r2)
                # crossover
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                # evaluation
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness <= pop_fitness[i]:
                    # replace
                    old_x = pop[i].copy()
                    pop_fitness[i] = trial_fitness
                    pop[i] = trial
                    # add old to archive if not best
                    if old_x is not self.best_x and len(archive) < max_archive:
                        archive.append(old_x)
                    elif len(archive) >= max_archive:
                        # randomly replace an archive entry
                        idx = rng.randint(0, max_archive)
                        archive[idx] = old_x
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        improved_this_gen = True
                # if no replacement, trial not added to archive
            if evals >= budget:
                break
            if improved_this_gen:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            # restart condition
            if stagnation_counter >= max_stagnation and evals + popsize - 1 <= budget:
                # keep best point, reinitialize rest, clear archive
                new_pop = np.zeros((popsize, dim))
                new_fitness = np.zeros(popsize)
                new_pop[0] = self.best_x
                new_fitness[0] = self.best_value
                for i in range(1, popsize):
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