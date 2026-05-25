import numpy as np
import random

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        np.random.seed(self.seed)
        random.seed(self.seed)
        pop_size = min(10 * dim, max(20, budget // 3))
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fcalls = 0
        # Evaluate initial population
        for i in range(pop_size):
            if fcalls >= budget:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            fcalls += 1
            pop_f[i] = val
            if val < best_f:
                best_f = val
                best_x = x.copy()
                report_best(best_f, best_x)
        # If budget exhausted after initialization, return
        if fcalls >= budget:
            return best_f, best_x
        # Stagnation parameters
        stagnation_limit = max(10, int(0.1 * budget))
        no_improve_evals = 0
        # Main loop: generations
        while fcalls < budget:
            # One generation: iterate over all individuals
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                # Select three distinct random indices not equal to i
                candidates = [j for j in range(pop_size) if j != i]
                r0, r1, r2 = random.sample(candidates, 3)
                # Adaptive F and CR for diversity
                F = 0.5 + 0.5 * random.random()
                CR = 0.6 + 0.4 * random.random()
                # DE/rand/1 mutation
                mutant = pop[r0] + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
                trial = pop[i].copy()
                j_rand = random.randint(0, dim - 1)
                for j in range(dim):
                    if random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                val = func(trial)
                fcalls += 1
                # Greedy selection
                if val < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = val
                    if val < best_f:
                        best_f = val
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        no_improve_evals = 0
                    else:
                        no_improve_evals += 1
                else:
                    no_improve_evals += 1
            # After generation, check for stagnation and restart if needed
            if no_improve_evals >= stagnation_limit and fcalls < budget:
                # Partial restart: keep best, randomize rest
                new_pop = [best_x.copy()]
                new_pop_f = [best_f]
                for _ in range(pop_size - 1):
                    if fcalls >= budget:
                        break
                    new_ind = np.random.uniform(lb, ub, dim)
                    val = func(new_ind)
                    fcalls += 1
                    new_pop.append(new_ind)
                    new_pop_f.append(val)
                    if val < best_f:
                        best_f = val
                        best_x = new_ind.copy()
                        report_best(best_f, best_x)
                # Update population
                if len(new_pop) == pop_size:
                    pop = np.array(new_pop)
                    pop_f = np.array(new_pop_f)
                    no_improve_evals = 0
                else:
                    # Budget exhausted during restart, leave as is (should be safe)
                    break
        return best_f, best_x