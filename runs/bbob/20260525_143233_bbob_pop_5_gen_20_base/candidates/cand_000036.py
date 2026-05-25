import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        # population size: at least 4, at most budget/2, scales with dim
        self.pop_size = max(4, min(4 * dim, budget // 2))
        if self.pop_size > budget:
            self.pop_size = budget
        # restart threshold: number of generations without improvement before restart
        self.restart_threshold = max(5, int(budget / (4 * self.pop_size)) if self.pop_size > 0 else 5)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        evals = 0
        best_val = np.inf
        best_x = None

        # Handle degenerate pop_size
        if pop_size <= 0:
            x = np.random.uniform(lb, ub, dim)
            best_val = func(x)
            best_x = x.copy()
            report_best(best_val, best_x)
            evals = 1
            while evals < self.budget:
                x = np.random.uniform(lb, ub, dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        # Initialize population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
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

        # Constants
        F = 0.7
        max_gen = (self.budget - evals) // pop_size if pop_size > 0 else 0
        no_improve = 0
        gen = 0
        while evals < self.budget and gen < max_gen:
            # Adaptive CR: decreases linearly from 0.9 to 0.2
            CR = 0.9 - 0.7 * (gen / max_gen) if max_gen > 0 else 0.9
            CR = np.clip(CR, 0.2, 0.9)
            improved = False
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # Select three distinct indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 3:
                    continue
                a, b, c = np.random.choice(candidates, size=3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # Exponential crossover
                trial = pop[i].copy()
                start = np.random.randint(0, dim)
                L = 0
                while L < dim and np.random.rand() < CR:
                    idx = (start + L) % dim
                    trial[idx] = mutant[idx]
                    L += 1
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    improved = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            if improved:
                no_improve = 0
            else:
                no_improve += 1
            # Restart if stagnation detected
            if no_improve >= self.restart_threshold:
                sorted_indices = np.argsort(fitness)
                n_keep = max(1, int(0.1 * pop_size))
                keep_indices = sorted_indices[:n_keep]
                new_pop = np.empty_like(pop)
                new_fitness = np.full(pop_size, np.inf)
                for idx, keep_idx in enumerate(keep_indices):
                    new_pop[idx] = pop[keep_idx].copy()
                    new_fitness[idx] = fitness[keep_idx]
                for i in range(n_keep, pop_size):
                    if evals >= self.budget:
                        break
                    x = np.random.uniform(lb, ub, dim)
                    new_pop[i] = x
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
            gen += 1
        return best_val, best_x