import numpy as np
import random

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        # Initial population size: moderate
        init_pop_size = min(5 * dim, max(10, budget // 4))
        pop_size = init_pop_size
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
        # DE parameters
        F = 0.8
        CR = 0.9
        reduced = False
        # Main loop
        while fcalls < budget:
            # One generation
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                # Select two distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2 = random.sample(candidates, 2)
                # DE/current-to-best/1
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # Crossover
                trial = pop[i].copy()
                j_rand = random.randint(0, dim-1)
                for j in range(dim):
                    if random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                # Evaluate
                val = func(trial)
                fcalls += 1
                if val < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = val
                    if val < best_f:
                        best_f = val
                        best_x = trial.copy()
                        report_best(best_f, best_x)
            # After generation, check for population reduction
            if not reduced and fcalls >= budget // 2:
                # Reduce population size to elite set
                new_pop_size = max(2 * dim, 5)
                if new_pop_size < pop_size:
                    # Keep best individuals
                    sorted_indices = np.argsort(pop_f)[:new_pop_size]
                    pop = pop[sorted_indices]
                    pop_f = pop_f[sorted_indices]
                    pop_size = new_pop_size
                    # Lower F for exploitation
                    F = 0.5
                    reduced = True
        return best_f, best_x