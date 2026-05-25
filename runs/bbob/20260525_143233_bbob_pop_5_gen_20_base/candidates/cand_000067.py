import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        # initial pop size
        self.pop_size_init = max(10, min(5 * dim, budget // 4))
        # minimum pop size
        self.pop_size_min = max(4, dim // 2)
        self.restart_threshold = max(10, 2 * dim)
        self.CR = 0.9

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # fallback for tiny population
        pop_size = self.pop_size_init
        if pop_size < 4:
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

        # initial population
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

        no_improve = 0
        generation = 0

        while evals < budget:
            # linear population reduction
            frac = evals / budget
            current_pop_size = max(self.pop_size_min, int(self.pop_size_init * (1 - frac * 0.8)))
            # ensure not larger than current pop
            while len(pop) > current_pop_size:
                # remove worst individuals
                worst_idx = np.argmax(fitness)
                pop = np.delete(pop, worst_idx, axis=0)
                fitness = np.delete(fitness, worst_idx)
            pop_size = len(pop)

            improved_this_gen = False
            success_CR = []
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 4:
                    continue
                r1, r2, r3, r4 = rng.choice(candidates, size=4, replace=False)
                F = rng.uniform(0.5, 1.0)
                # DE/rand/2
                mutant = pop[r1] + F * (pop[r2] - pop[r3]) + F * (rng.uniform(0.5, 1.0) * (pop[r4] - pop[r2]))
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

            # local search around best with decreasing step
            if evals < budget and best_x is not None:
                local_evals = min(1, budget - evals)
                sigma = 0.1 * (ub - lb) * (1 - evals / budget)
                for _ in range(local_evals):
                    x = best_x + sigma * rng.randn(dim)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)

            # adapt CR
            if len(success_CR) > 0:
                self.CR = 0.8 * self.CR + 0.2 * min(1.0, max(0.1, np.mean(success_CR)))
            else:
                self.CR = max(0.1, self.CR * 0.95)

            if improved_this_gen:
                no_improve = 0
            else:
                no_improve += 1

            # restart
            if no_improve >= self.restart_threshold:
                new_pop = rng.uniform(lb, ub, (pop_size, dim))
                if best_x is not None:
                    num_focused = max(1, int(0.3 * pop_size))
                    for j in range(num_focused):
                        # differential perturbation around best
                        idx = rng.choice(pop_size, 2, replace=False)
                        new_pop[j] = best_x + 0.5 * (pop[idx[0]] - pop[idx[1]])
                        new_pop[j] = np.clip(new_pop[j], lb, ub)
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
                no_improve = 0
                self.CR = 0.9
            generation += 1

        return best_val, best_x