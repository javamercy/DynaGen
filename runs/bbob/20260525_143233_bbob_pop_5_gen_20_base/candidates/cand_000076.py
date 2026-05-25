import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(8, min(5 * dim, budget // 3))
        self.restart_diversity_threshold = 0.1  # fraction of the range
        self.CR = 0.9
        self.success_history = []

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget
        rng = self.rng

        if pop_size < 3:
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

        generation = 0
        max_gen = (budget - evals) // pop_size if pop_size > 0 else 0

        while evals < budget and generation < max_gen:
            improved_this_gen = False
            success_CR = []
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 3:
                    continue
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                F = rng.uniform(0.5, 1.0)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)
                CR = self.CR
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
                    success_CR.append(CR)
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # Adapt CR based on success
            if len(success_CR) > 0:
                self.CR = 0.8 * self.CR + 0.2 * min(1.0, max(0.5, np.mean(success_CR)))
            else:
                self.CR = min(1.0, self.CR * 1.05)

            # Local perturbation around best using Cauchy
            if evals < budget:
                local_evals = min(2, budget - evals)
                for _ in range(local_evals):
                    sigma = 0.05 * (ub - lb)
                    x = best_x + sigma * rng.standard_cauchy(dim)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)

            # Diversity check and restart
            if evals < budget:
                # compute mean pairwise distance in population
                mean_dist = 0.0
                count = 0
                for i in range(pop_size):
                    for j in range(i+1, pop_size):
                        mean_dist += np.linalg.norm(pop[i] - pop[j])
                        count += 1
                if count > 0:
                    mean_dist /= count
                range_norm = np.linalg.norm(ub - lb)
                diversity = mean_dist / range_norm if range_norm > 0 else 0
                if diversity < self.restart_diversity_threshold:
                    new_pop = rng.uniform(lb, ub, (pop_size, dim))
                    new_pop[0] = best_x.copy()
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
                    self.CR = 0.9
                else:
                    # Every 5 generations, sample a few random points for exploration
                    if generation % 5 == 0 and evals < budget:
                        num_random = min(3, budget - evals)
                        for _ in range(num_random):
                            x = rng.uniform(lb, ub, dim)
                            val = func(x)
                            evals += 1
                            if val < best_val:
                                best_val = val
                                best_x = x.copy()
                                report_best(best_val, best_x)

            generation += 1

        return best_val, best_x