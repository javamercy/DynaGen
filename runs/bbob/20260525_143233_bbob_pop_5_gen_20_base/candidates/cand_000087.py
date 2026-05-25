import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(6, min(6 * dim, budget // 3))
        self.restart_threshold = max(15, 2 * dim)
        self.CR = 0.9
        self.success_CR = []

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget
        rng = self.rng

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

        while evals < budget and generation < max_gen:
            improved_this_gen = False
            success_CR_local = []
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
                # 30% chance to replace trial with random point
                if rng.rand() < 0.3:
                    trial = rng.uniform(lb, ub, dim)
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    improved_this_gen = True
                    success_CR_local.append(CR)
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # Local search around best with larger step
            if evals < budget:
                local_evals = min(5, budget - evals)
                for _ in range(local_evals):
                    sigma = 0.1 * (ub - lb)
                    x = best_x + sigma * rng.randn(dim)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)

            # Adapt CR based on success
            if len(success_CR_local) > 0:
                new_CR = 0.8 * self.CR + 0.2 * min(1.0, max(0.3, np.mean(success_CR_local)))
                self.CR = np.clip(new_CR, 0.3, 0.99)
            else:
                self.CR = max(0.3, self.CR * 0.95)

            if improved_this_gen:
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.restart_threshold:
                # Partial restart: keep best and top 10% individuals, reinit rest
                num_keep = max(1, int(0.1 * pop_size))
                order = np.argsort(fitness)
                new_pop = np.empty_like(pop)
                new_fitness = np.full(pop_size, np.inf)
                for idx in range(num_keep):
                    orig_idx = order[idx]
                    new_pop[idx] = pop[orig_idx].copy()
                    new_fitness[idx] = fitness[orig_idx]
                for idx in range(num_keep, pop_size):
                    if evals >= budget:
                        break
                    x = rng.uniform(lb, ub, dim)
                    new_pop[idx] = x
                    val = func(x)
                    evals += 1
                    new_fitness[idx] = val
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