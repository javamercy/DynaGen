import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        # dynamic population size
        self.pop_size = max(3, min(budget // 10, 5 * dim))

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        pop_size = self.pop_size

        # DE parameters (changed from parents: F and CR slightly different)
        F = 0.5
        CR = 0.5

        # initialization
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        # initial evaluations
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

        # main DE loop
        while evals < budget:
            # DE/rand/1 mutation (not best/1, to encourage exploration)
            for i in range(pop_size):
                if evals >= budget:
                    break
                # select three distinct random indices different from i
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                j_rand = np.random.randint(dim)
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

            # local refinement around best (adaptive step size)
            if evals < budget:
                sigma = 0.1 * (ub - lb).mean() * (1 - evals / budget)
                for _ in range(min(5, budget - evals)):
                    candidate = best_x + sigma * np.random.randn(dim)
                    candidate = np.clip(candidate, lb, ub)
                    val = func(candidate)
                    evals += 1
                    if val < best_val:
                        # line search along improvement direction
                        direction = candidate - best_x
                        line_length = 0.5
                        while evals < budget:
                            step = best_x + line_length * direction
                            step = np.clip(step, lb, ub)
                            v = func(step)
                            evals += 1
                            if v < best_val:
                                best_val = v
                                best_x = step.copy()
                                report_best(best_val, best_x)
                                line_length *= 2
                            else:
                                line_length *= 0.5
                                if line_length < 1e-8:
                                    break
                        # also update a random population member
                        idx = np.random.randint(pop_size)
                        pop[idx] = candidate
                        fitness[idx] = val
                        break  # after improvement, continue DE loop
                    else:
                        sigma *= 0.9

            # replace worst with best + noise for diversity
            if evals % max(1, budget // 10) < pop_size:
                if evals < budget:
                    num_repl = max(1, pop_size // 5)
                    worst_idx = np.argsort(fitness)[-num_repl:]
                    for idx in worst_idx:
                        if evals >= budget:
                            break
                        new_x = best_x + 0.05 * np.random.randn(dim) * (ub - lb)
                        new_x = np.clip(new_x, lb, ub)
                        val = func(new_x)
                        evals += 1
                        if val < fitness[idx]:
                            pop[idx] = new_x
                            fitness[idx] = val
                        if val < best_val:
                            best_val = val
                            best_x = new_x.copy()
                            report_best(best_val, best_x)

        return best_val, best_x