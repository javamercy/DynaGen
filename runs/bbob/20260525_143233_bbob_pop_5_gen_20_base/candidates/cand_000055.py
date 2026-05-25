import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(4 * dim, budget // 2))
        self.pop_size = min(self.pop_size, budget)
        self.pop_size = max(self.pop_size, 1)
        self.restart_threshold = max(5, 2 * dim)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        pop_size = self.pop_size
        rng = self.rng

        if pop_size == 1:
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

        F = 0.5
        CR = 0.9
        no_improve = 0
        generation = 0

        while evals < budget:
            improved_this_gen = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 3:
                    continue
                a, b, c = rng.choice(candidates, size=3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
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
            if improved_this_gen:
                F *= 1.1
                F = min(F, 0.9)
                no_improve = 0
            else:
                F *= 0.9
                F = max(F, 0.1)
                no_improve += 1
            # partial restart: replace worst half
            if no_improve >= self.restart_threshold:
                # sort by fitness
                idx = np.argsort(fitness)
                pop_sorted = pop[idx]
                fitness_sorted = fitness[idx]
                # keep best half, replace worst half
                n_keep = pop_size // 2
                n_replace = pop_size - n_keep
                pop[:n_keep] = pop_sorted[:n_keep]
                fitness[:n_keep] = fitness_sorted[:n_keep]
                for j in range(n_replace):
                    if evals >= budget:
                        break
                    x = rng.uniform(lb, ub, dim)
                    val = func(x)
                    evals += 1
                    pop[n_keep + j] = x
                    fitness[n_keep + j] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                # reset parameters
                F = 0.5
                no_improve = 0
            generation += 1
        return best_val, best_x