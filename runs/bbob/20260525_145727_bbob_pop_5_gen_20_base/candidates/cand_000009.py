import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(10*dim, budget // 4))
        if self.pop_size > budget:
            self.pop_size = budget
        self.F = 0.8
        self.CR = 0.9
        self.stagnation_limit = max(5 * dim, 20)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = self.pop_size
        F = self.F
        CR = self.CR
        rng = self.rng

        pop = lb + rng.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

        for i in range(pop_size):
            if evals >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        stagnation_counter = 0
        while evals < self.budget:
            improved = False
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                jrand = rng.randint(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == jrand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop[i, j]
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        improved = True
            if not improved:
                stagnation_counter += pop_size
            else:
                stagnation_counter = 0
            if stagnation_counter >= self.stagnation_limit and evals < self.budget:
                # Restart: keep best, reinitialize others
                new_pop = [best_x.copy()] if best_x is not None else []
                remaining = pop_size - len(new_pop)
                if remaining > 0:
                    new_points = lb + rng.rand(remaining, dim) * (ub - lb)
                    for idx in range(remaining):
                        if evals >= self.budget:
                            break
                        x = new_points[idx]
                        val = func(x)
                        evals += 1
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            report_best(best_val, best_x)
                        new_pop.append(x)
                    # Trim to pop_size
                    pop = np.array(new_pop[:pop_size])
                    fitness = np.full(pop_size, np.inf)
                    # Update fitness for all points
                    for i in range(pop_size):
                        if evals >= self.budget:
                            break
                        x = pop[i]
                        val = func(x)
                        evals += 1
                        fitness[i] = val
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            report_best(best_val, best_x)
                stagnation_counter = 0
        return best_val, best_x