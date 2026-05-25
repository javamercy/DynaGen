import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        # population size
        pop_size = max(5, min(10 * dim, self.budget // 4))
        if pop_size > self.budget:
            pop_size = self.budget
        # initialize
        pop = lb + self.rng.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0
        for i in range(pop_size):
            if evals >= self.budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_val:
                best_val = fitness[i]
                best_x = pop[i].copy()
        # report initial best if any
        if best_x is not None:
            from helper import report_best
            report_best(best_val, best_x)
        # main loop
        gen = 0
        while evals < self.budget:
            for target_idx in range(pop_size):
                if evals >= self.budget:
                    break
                # select three distinct random individuals different from target
                candidates = list(range(pop_size))
                candidates.remove(target_idx)
                self.rng.shuffle(candidates)
                a, b, c = candidates[:3]
                # mutation
                F = 0.8
                mutant = pop[a] + F * (pop[b] - pop[c])
                # binomial crossover
                CR = 0.9
                trial = pop[target_idx].copy()
                j_rand = self.rng.randint(dim)
                for j in range(dim):
                    if self.rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                # clip to bounds
                trial = np.clip(trial, lb, ub)
                # evaluate
                trial_val = func(trial)
                evals += 1
                if trial_val < fitness[target_idx]:
                    pop[target_idx] = trial
                    fitness[target_idx] = trial_val
                    if trial_val < best_val:
                        best_val = trial_val
                        best_x = trial.copy()
                        from helper import report_best
                        report_best(best_val, best_x)
            gen += 1
        return best_val, best_x