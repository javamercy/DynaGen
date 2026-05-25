import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(3 * dim, budget // 5))
        self.restart_threshold = max(10, 2 * dim)
        self.CR = 0.9
        self.local_iters = max(1, int(0.05 * budget))

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = self.pop_size
        budget = self.budget
        rng = self.rng
        dim = self.dim

        if pop_size < 2:
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

        no_improve = 0
        generation = 0
        max_gen = (budget - evals) // pop_size if pop_size > 0 else 0
        local_evals_used = 0
        local_interval = max(1, int(0.1 * max_gen))

        while evals < budget and generation < max_gen:
            improved_this_gen = False
            success_CR = []
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 2:
                    continue
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                F = rng.uniform(0.5, 1.0)
                mutant = best_x + F * (pop[r1] - pop[r2])
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

            if len(success_CR) > 0:
                self.CR = 0.8 * self.CR + 0.2 * min(1.0, max(0.1, np.mean(success_CR)))
            else:
                self.CR = max(0.1, self.CR * 0.95)

            if improved_this_gen:
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.restart_threshold:
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
                no_improve = 0
                self.CR = 0.9

            if no_improve % local_interval == 0 and no_improve > 0:
                sigma = 0.2 * (ub - lb)
                for _ in range(self.local_iters):
                    if evals >= budget:
                        break
                    trial = best_x + rng.normal(0, sigma, dim)
                    trial = np.clip(trial, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        sigma *= 0.95
                    else:
                        sigma *= 0.9

            generation += 1

        return best_val, best_x