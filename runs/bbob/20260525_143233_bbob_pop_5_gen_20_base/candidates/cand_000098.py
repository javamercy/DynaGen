import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        # Small population for exploitation
        self.pop_size = max(4, min(3 * dim, budget // 6))
        self.restart_threshold = max(10, 2 * dim)
        self.CR = 0.2  # low crossover
        self.local_search_budget = max(5, int(0.2 * budget))

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
        # Reserve budget for local search at the end
        main_budget = budget - self.local_search_budget
        max_gen = (main_budget - evals) // pop_size if pop_size > 0 else 0

        while evals < main_budget and generation < max_gen:
            improved_this_gen = False
            success_CR = []
            for i in range(pop_size):
                if evals >= main_budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 2:
                    continue
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                F = rng.uniform(0.3, 0.6)  # low mutation
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
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

            # Adapt CR: keep low
            if len(success_CR) > 0:
                self.CR = 0.8 * self.CR + 0.2 * min(0.5, max(0.05, np.mean(success_CR)))
            else:
                self.CR = max(0.05, self.CR * 0.9)

            if improved_this_gen:
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.restart_threshold:
                # Soft restart: reinitialize around best
                sigma = 0.1 * (ub - lb)  # 10% of domain
                new_pop = best_x + rng.normal(0, sigma, (pop_size, dim))
                new_pop = np.clip(new_pop, lb, ub)
                new_pop[0] = best_x.copy()  # keep best
                new_fitness = np.full(pop_size, np.inf)
                new_fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= main_budget:
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
                self.CR = 0.2
            generation += 1

        # Local search: coordinate-wise refinement
        for _ in range(self.local_search_budget):
            if evals >= budget:
                break
            # Perturb best coordinate by small step
            coord = rng.randint(0, dim)
            step = 0.01 * (ub[coord] - lb[coord]) * rng.choice([-1, 1])
            x_trial = best_x.copy()
            x_trial[coord] = np.clip(x_trial[coord] + step, lb[coord], ub[coord])
            val = func(x_trial)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
                report_best(best_val, best_x)
        return best_val, best_x