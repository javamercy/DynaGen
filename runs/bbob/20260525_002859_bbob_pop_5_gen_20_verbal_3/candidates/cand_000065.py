import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        popsize = min(budget, max(10, 5 * dim))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fit = np.zeros(popsize)
        evals = 0
        best_val = None
        best_x = None
        for i in range(popsize):
            pop_fit[i] = func(pop[i])
            evals += 1
            if i == 0 or pop_fit[i] < best_val:
                best_val = pop_fit[i]
                best_x = pop[i].copy()
                report_best(best_val, best_x)
        if evals >= budget:
            return best_val, best_x

        H = min(10, dim)
        MF = 0.5 * np.ones(H)
        MCR = 0.9 * np.ones(H)
        s = np.array([1.0, 1.0])
        n = np.array([2.0, 2.0])
        stagnation = 0
        max_stagnation = max(10, int(budget / (3 * popsize)))

        while evals < budget:
            parameters = []
            for _ in range(popsize):
                idx = rng.randint(H)
                F = rng.standard_cauchy() * 0.1 + MF[idx]
                while F <= 0:
                    F = rng.standard_cauchy() * 0.1 + MF[idx]
                F = min(F, 1.0)
                CR = rng.randn() * 0.1 + MCR[idx]
                CR = np.clip(CR, 0, 1)
                p = (s[0] + 1) / (s[0] + s[1] + 2)
                if rng.rand() < p:
                    strategy = 0
                else:
                    strategy = 1
                parameters.append((F, CR, strategy))

            improved_global = False
            successful_F = []
            successful_CR = []

            for i in range(popsize):
                if evals >= budget:
                    break
                F, CR, strategy = parameters[i]
                if strategy == 0:
                    candidates = list(range(popsize))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    r1, r2 = candidates[0], candidates[1]
                    mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
                else:
                    candidates = list(range(popsize))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    r1, r2, r3 = candidates[0], candidates[1], candidates[2]
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                fit = func(trial)
                evals += 1
                if fit <= pop_fit[i]:
                    pop[i] = trial
                    pop_fit[i] = fit
                    successful_F.append(F)
                    successful_CR.append(CR)
                    n[strategy] += 1
                    s[strategy] += 1
                    if fit < best_val:
                        best_val = fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        improved_global = True
                else:
                    n[strategy] += 1

            if evals >= budget:
                break

            if len(successful_F) > 0:
                weights = np.ones(len(successful_F))
                k = rng.randint(H)
                MF[k] = np.sum(np.array(successful_F) * weights) / np.sum(weights)
                MCR[k] = np.sum(np.array(successful_CR) * weights) / np.sum(weights)

            if improved_global:
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= max_stagnation and evals + popsize <= budget:
                new_pop = np.zeros((popsize, dim))
                new_pop[0] = best_x + 0.1 * (ub - lb) * rng.randn(dim)
                new_pop[0] = np.clip(new_pop[0], lb, ub)
                pop = new_pop
                pop_fit = np.zeros(popsize)
                for i in range(popsize):
                    if evals >= budget:
                        break
                    if i == 0:
                        pop_fit[0] = func(pop[0])
                    else:
                        pop[i] = lb + (ub - lb) * rng.rand(dim)
                        pop_fit[i] = func(pop[i])
                    evals += 1
                    if pop_fit[i] < best_val:
                        best_val = pop_fit[i]
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                stagnation = 0

        return best_val, best_x