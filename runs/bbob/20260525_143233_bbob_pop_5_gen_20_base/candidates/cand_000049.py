import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.pop_size = max(4, min(4 * dim, budget // 2))
        self.restart_threshold = max(5, 2 * dim)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget

        if pop_size <= 0:
            best_x = np.random.uniform(lb, ub, dim)
            best_val = func(best_x)
            report_best(best_val, best_x)
            evals = 1
            while evals < budget:
                x = np.random.uniform(lb, ub, dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i].copy()
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        F = 0.5
        CR = 0.9
        local_sigma_factor = 0.01
        no_improve = 0
        generation = 0
        max_gen = (budget - evals) // pop_size if pop_size > 0 else 0

        while evals < budget and generation < max_gen:
            improved_this_gen = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 2:
                    continue
                r1, r2 = np.random.choice(candidates, size=2, replace=False)
                mutant = best_x + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # Exponential crossover
                n = dim
                start = np.random.randint(0, n)
                L = 0
                while np.random.rand() < CR and L < n:
                    L += 1
                trial = pop[i].copy()
                for j in range(L):
                    idx = (start + j) % n
                    trial[idx] = mutant[idx]
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
            # Local search around best
            if evals < budget:
                local_evals = min(2, budget - evals)
                for _ in range(local_evals):
                    sigma = local_sigma_factor * (ub - lb)
                    x = best_x + sigma * np.random.randn(dim)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
            if improved_this_gen:
                F *= 1.1
                F = min(F, 0.9)
                no_improve = 0
            else:
                F *= 0.9
                F = max(F, 0.1)
                no_improve += 1
            if no_improve >= self.restart_threshold:
                new_pop = np.random.uniform(lb, ub, (pop_size, dim))
                if best_x is not None:
                    num_focused = max(1, int(0.3 * pop_size))
                    for j in range(num_focused):
                        new_pop[j] = best_x + 0.1 * np.random.randn(dim) * (ub - lb)
                        new_pop[j] = np.clip(new_pop[j], lb, ub)
                    new_pop[0] = best_x.copy()
                new_fitness = np.full(pop_size, np.inf)
                new_fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    x = new_pop[i].copy()
                    val = func(x)
                    evals += 1
                    new_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                pop = new_pop
                fitness = new_fitness
                F = 0.5
                no_improve = 0
            generation += 1

        return best_val, best_x