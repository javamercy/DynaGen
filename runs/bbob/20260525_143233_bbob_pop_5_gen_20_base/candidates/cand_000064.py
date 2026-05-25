import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        np.random.seed(seed)
        self.dim = dim
        self.budget = budget
        self.pop_size = max(10, min(6 * dim, budget // 3))
        self.F_min = 0.2
        self.F_max = 1.4
        self.CR = 0.9
        self.restart_threshold = max(10, 2 * dim)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget

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

        no_improve = 0
        gen = 0
        max_gen = (budget - evals) // pop_size if pop_size else 0

        while evals < budget and gen < max_gen:
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            improved_this_gen = False

            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = [j for j in range(pop_size) if j != i]
                if len(candidates) < 5:
                    continue
                r = np.random.choice(candidates, size=5, replace=False)
                F = np.random.uniform(self.F_min, self.F_max)
                mutant = pop[r[0]] + F * (pop[r[1]] - pop[r[2]]) + F * (pop[r[3]] - pop[r[4]])
                mutant = np.clip(mutant, lb, ub)
                cross = np.random.rand(dim) < self.CR
                if not np.any(cross):
                    cross[np.random.randint(dim)] = True
                trial = np.where(cross, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = val
                    improved_this_gen = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # random immigrant replacement
            if gen > 0 and evals < budget:
                worst_idx = np.argmax(new_fitness)
                x_rand = np.random.uniform(lb, ub, dim)
                val = func(x_rand)
                evals += 1
                new_pop[worst_idx] = x_rand
                new_fitness[worst_idx] = val
                if val < best_val:
                    best_val = val
                    best_x = x_rand.copy()
                    report_best(best_val, best_x)

            pop = new_pop
            fitness = new_fitness

            if improved_this_gen:
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.restart_threshold:
                pop = np.random.uniform(lb, ub, (pop_size, dim))
                if best_x is not None:
                    num_keepers = max(1, int(0.2 * pop_size))
                    for j in range(num_keepers):
                        offset = 0.2 * (ub - lb) * np.random.randn(dim)
                        pop[j] = np.clip(best_x + offset, lb, ub)
                    pop[0] = best_x.copy()
                fitness = np.full(pop_size, np.inf)
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
                no_improve = 0

            gen += 1

        return best_val, best_x