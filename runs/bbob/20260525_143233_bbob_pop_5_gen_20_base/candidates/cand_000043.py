import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.pop_size = max(4, min(4*dim, budget // 2))
        self.restart_threshold = max(5, 5*dim)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget
        evals = 0

        # initialize random point
        best_x = np.random.uniform(lb, ub, dim)
        best_val = func(best_x)
        evals += 1
        report_best(best_val, best_x)

        # population initialization
        pop = np.random.uniform(lb, ub, (pop_size, dim))
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

        F = 0.7
        CR = 0.9
        no_improve = 0
        generation = 0
        max_gen = budget // pop_size

        while evals < budget and generation < max_gen:
            improved_this_gen = False
            # generate new population via DE/best/1
            new_pop = np.empty_like(pop)
            new_fitness = np.empty(pop_size)
            for i in range(pop_size):
                if evals >= budget:
                    break
                # select r1 != r2 != i
                indices = list(range(pop_size))
                indices.remove(i)
                r1, r2 = np.random.choice(indices, 2, replace=False)
                mutant = best_x + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                cross = np.random.rand(dim) < CR
                if not np.any(cross):
                    cross[np.random.randint(dim)] = True
                trial = np.where(cross, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                if val < fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = val
                    improved_this_gen = True
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
            if evals >= budget:
                break
            pop = new_pop
            fitness = new_fitness
            # adapt F
            if improved_this_gen:
                F = min(F * 1.1, 0.9)
                no_improve = 0
            else:
                F = max(F * 0.9, 0.1)
                no_improve += 1
            # restart if needed
            if no_improve >= self.restart_threshold:
                # reinitialize population, keep best
                pop = np.random.uniform(lb, ub, (pop_size, dim))
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
                F = 0.7
                no_improve = 0
            generation += 1

        return best_val, best_x