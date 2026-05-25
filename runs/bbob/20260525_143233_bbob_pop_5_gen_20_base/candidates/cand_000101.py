import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(5 * dim, budget // 4))
        self.restart_threshold = max(10, 2 * dim)
        self.CR = 0.9

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
        # initial diversity measure for partial restart threshold
        range_width = ub - lb
        if np.any(range_width == 0):
            range_width = np.ones(dim)
        init_spread = np.std(pop, axis=0) / range_width
        init_avg_spread = np.mean(init_spread)

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
                # choose mutation strategy
                if rng.uniform() < 0.5:
                    # current-to-best/1
                    F = self._sample_F(rng)
                    mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
                else:
                    # rand/1
                    F = self._sample_F(rng)
                    mutant = pop[r1] + F * (pop[r2] - pop[r1])
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

            # Adapt CR
            if len(success_CR) > 0:
                self.CR = 0.8 * self.CR + 0.2 * min(1.0, max(0.1, np.mean(success_CR)))
            else:
                self.CR = max(0.1, self.CR * 0.95)

            if improved_this_gen:
                no_improve = 0
            else:
                no_improve += 1

            # Restart strategies
            if no_improve >= self.restart_threshold:
                # full restart
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
            elif no_improve >= self.restart_threshold // 2:
                # partial restart if diversity low
                spread = np.std(pop, axis=0) / range_width
                avg_spread = np.mean(spread)
                if avg_spread < 0.1 * init_avg_spread:
                    worst_indices = np.argsort(fitness)[-pop_size//2:]
                    for idx in worst_indices:
                        if evals >= budget:
                            break
                        new_x = rng.uniform(lb, ub, dim)
                        val = func(new_x)
                        evals += 1
                        pop[idx] = new_x
                        fitness[idx] = val
                        if val < best_val:
                            best_val = val
                            best_x = new_x.copy()
                            report_best(best_val, best_x)
                    no_improve = 0

            generation += 1

        return best_val, best_x

    def _sample_F(self, rng):
        # truncated Cauchy for large steps
        while True:
            F = rng.standard_cauchy() * 0.3 + 0.5
            if 0.2 <= F <= 2.0:
                return F