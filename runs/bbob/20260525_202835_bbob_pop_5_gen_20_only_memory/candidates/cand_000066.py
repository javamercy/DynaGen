import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(4 * dim, budget // 4))  # leave budget for restarts
        self.F = 0.8
        self.CR = 0.9
        self.stag_limit = max(dim, budget // (10 * dim))  # evaluations without improvement to trigger restart

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop = self.rng.uniform(lb, ub, size=(self.popsize, dim))
        fitness = np.full(self.popsize, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0
        # initial evaluation
        for i in range(self.popsize):
            if evaluations >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evaluations += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        last_improvement_eval = evaluations  # track when best last improved
        while evaluations < self.budget:
            # generate new population using DE/rand/1/bin
            for i in range(self.popsize):
                if evaluations >= self.budget:
                    break
                # select three distinct random indices, all different from i
                indices = [j for j in range(self.popsize) if j != i]
                r1, r2, r3 = self.rng.choice(indices, 3, replace=False)
                # mutation: DE/rand/1
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                cross_points = self.rng.random(dim) < self.CR
                if not np.any(cross_points):
                    cross_points[self.rng.integers(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evaluations += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        last_improvement_eval = evaluations
            # after generation, check stagnation
            if evaluations - last_improvement_eval >= self.stag_limit and evaluations < self.budget:
                # restart worst 20% of population
                num_restart = max(1, self.popsize // 5)
                # sort by fitness descending
                order = np.argsort(fitness)[::-1]  # worst first
                for idx in order[:num_restart]:
                    if evaluations >= self.budget:
                        break
                    # generate new random point
                    new_x = self.rng.uniform(lb, ub, size=dim)
                    val = func(new_x)
                    evaluations += 1
                    pop[idx] = new_x
                    fitness[idx] = val
                    if val < best_val:
                        best_val = val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)
                        last_improvement_eval = evaluations
        return best_val, best_x