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

        while evals < budget and generation < max_gen:
            improved_this_gen = False
            success_CR = []
            # Sort indices by fitness for partial restart later
            sorted_indices = np.argsort(fitness)
            worst_half_indices = sorted_indices[pop_size//2:] if pop_size > 1 else []

            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 3:
                    continue
                # Select 3 distinct random indices for rand/1 mutation
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                F = rng.uniform(0.5, 1.5)  # slightly larger range for diversity
                # DE/rand/1
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

            # Adapt CR
            if len(success_CR) > 0:
                self.CR = 0.8 * self.CR + 0.2 * min(1.0, max(0.1, np.mean(success_CR)))
            else:
                self.CR = max(0.1, self.CR * 0.95)

            if improved_this_gen:
                no_improve = 0
            else:
                no_improve += 1

            # Partial restart when stagnation: reinitialize worst half while preserving best
            if no_improve >= self.restart_threshold:
                # Keep best
                best_idx = np.argmin(fitness)
                worst_indices = [i for i in range(pop_size) if i != best_idx]
                # Reinitialize worst half (excluding best)
                num_reinit = len(worst_indices) // 2
                if num_reinit > 0:
                    reinit_indices = rng.choice(worst_indices, size=num_reinit, replace=False)
                    for idx in reinit_indices:
                        if evals >= budget:
                            break
                        x = rng.uniform(lb, ub, dim)
                        pop[idx] = x
                        val = func(x)
                        evals += 1
                        fitness[idx] = val
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            report_best(best_val, best_x)
                    no_improve = 0
                    self.CR = 0.9  # reset CR

            generation += 1

        return best_val, best_x