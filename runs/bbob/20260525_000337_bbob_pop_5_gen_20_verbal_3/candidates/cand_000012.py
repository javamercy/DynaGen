import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed)
        dim = self.dim
        budget = self.budget
        lb = func.bounds.lb
        ub = func.bounds.ub

        pop_size = max(10, 4 * dim)
        if pop_size * 2 > budget:
            pop_size = max(4, budget // 2)

        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        evals = 0
        best_val = np.inf
        best_x = None

        # Initial evaluation
        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                try:
                    report_best(best_val, best_x)
                except NameError:
                    pass

        CR = 0.9
        stagnation_gen = 0
        max_stagnation = max(10, int(budget / (pop_size * 5)))

        while evals < budget:
            generation = 0  # not used, but placeholder
            improved = False
            indices = list(range(pop_size))
            np.random.shuffle(indices)

            for i in indices:
                if evals >= budget:
                    break

                if np.random.rand() < 0.6:
                    # DE/rand/1
                    candidates = [j for j in range(pop_size) if j != i]
                    r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                    F = np.random.uniform(0.2, 1.0)
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])
                    trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                else:
                    # DE/current-to-rand/1 (no crossover)
                    candidates = [j for j in range(pop_size) if j != i]
                    r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                    K = np.random.uniform(0.2, 1.0)
                    F = np.random.uniform(0.2, 1.0)
                    trial = pop[i] + K * (pop[r1] - pop[i]) + F * (pop[r2] - pop[r3])

                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1

                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        improved = True
                        try:
                            report_best(best_val, best_x)
                        except NameError:
                            pass

            if improved:
                stagnation_gen = 0
            else:
                stagnation_gen += 1

            if stagnation_gen >= max_stagnation and evals < budget - pop_size:
                # Restart: keep best, reinitialize others
                best_fit = best_val
                best_vec = best_x.copy()
                new_pop = np.random.uniform(lb, ub, (pop_size, dim))
                new_pop[0] = best_vec
                new_fitness = np.full(pop_size, np.inf)
                new_fitness[0] = best_fit
                for j in range(1, pop_size):
                    if evals >= budget:
                        break
                    val = func(new_pop[j])
                    evals += 1
                    new_fitness[j] = val
                    if val < best_val:
                        best_val = val
                        best_x = new_pop[j].copy()
                        try:
                            report_best(best_val, best_x)
                        except NameError:
                            pass
                pop = new_pop
                fitness = new_fitness
                stagnation_gen = 0

        return best_val, best_x