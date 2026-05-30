import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(3, min(budget // 2, 10 * dim))

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        pop_size = self.pop_size
        F = 0.8
        CR = 0.9

        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        evals = 0
        best_x = None
        best_val = np.inf

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

        stagnation_counter = 0
        stagnation_limit = max(2, min(20, int(0.1 * budget / pop_size)))

        while evals < budget:
            improved = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                # rand/1 mutation
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)

                j_rand = np.random.randint(0, dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]

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

            if improved:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= stagnation_limit and evals < budget:
                # restart: keep best, reinitialize rest
                pop[0] = best_x.copy()
                fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    pop[i] = np.random.uniform(lb, ub, dim)
                    val = func(pop[i])
                    evals += 1
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                stagnation_counter = 0

        return best_val, best_x