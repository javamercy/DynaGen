import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        # larger population for exploration
        self.pop_size = max(5, min(5 * dim, budget // 2))
        self.restart_threshold = max(3 * dim, 10)
        self.div_threshold = 0.1  # trigger restart if avg pairwise distance < 0.1 * (ub-lb) mean

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        pop_size = self.pop_size
        rng = self.rng

        if pop_size <= 1:
            best_x = rng.uniform(lb, ub, dim)
            best_val = func(best_x)
            report_best(best_val, best_x)
            evals = 1
            while evals < budget:
                x = rng.uniform(lb, ub, dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        # Initialize population
        pop = rng.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        F = 0.7
        CR = 0.5
        no_improve = 0

        while evals < budget:
            improved_this_gen = False
            # DE/rand/1 mutation (exploratory)
            for i in range(pop_size):
                if evals >= budget:
                    break
                # select three distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 3:
                    continue
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover with low CR
                cross_points = rng.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    improved_this_gen = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # local search (only one evaluation per generation)
            if evals < budget:
                sigma = 0.01 * (ub - lb)
                x = best_x + sigma * rng.randn(dim)
                x = np.clip(x, lb, ub)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)

            if improved_this_gen:
                F *= 1.05
                F = min(F, 0.9)
                no_improve = 0
            else:
                F *= 0.95
                F = max(F, 0.2)
                no_improve += 1

            # Check for diversity-based restart
            if no_improve >= self.restart_threshold:
                # compute average pairwise distance
                mean_dist = 0.0
                count = 0
                for i in range(pop_size):
                    for j in range(i+1, pop_size):
                        mean_dist += np.linalg.norm(pop[i] - pop[j])
                        count += 1
                if count > 0:
                    mean_dist /= count
                range_scale = np.mean(ub - lb)
                if mean_dist < self.div_threshold * range_scale:
                    # restart with half near best, half random
                    num_focused = max(1, int(0.5 * pop_size))
                    new_pop = rng.uniform(lb, ub, (pop_size, dim))
                    if best_x is not None:
                        for j in range(num_focused):
                            new_pop[j] = best_x + 0.1 * rng.randn(dim) * (ub - lb)
                            new_pop[j] = np.clip(new_pop[j], lb, ub)
                        new_pop[0] = best_x.copy()
                    # evaluate new population except best already known
                    new_fitness = np.full(pop_size, np.inf)
                    new_fitness[0] = best_val
                    for i in range(1, pop_size):
                        if evals >= budget:
                            break
                        x = new_pop[i]
                        val = func(x)
                        evals += 1
                        new_fitness[i] = val
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            report_best(best_val, best_x)
                    pop = new_pop
                    fitness = new_fitness
                    F = 0.7
                    no_improve = 0

        return best_val, best_x