import numpy as np
class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = None
        self.best_x = None

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        evals = 0
        popsize = min(budget, max(4, min(5*dim, 20)))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fitness = np.full(popsize, np.inf)
        for i in range(popsize):
            pop_fitness[i] = func(pop[i])
            evals += 1
            if evals == 1 or pop_fitness[i] < self.best_value:
                self.best_value = pop_fitness[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x
        # JADE memory
        H = 5
        muF = np.ones(H) * 0.5
        muCR = np.ones(H) * 0.5
        k = 0
        archive = []
        max_archive = popsize
        # stagnation tracking
        generations_no_improvement = 0
        while evals < budget:
            # check restart condition
            if popsize > 1:
                pop_std = np.std(pop, axis=0)
                ranges = ub - lb
                norm_std = np.mean(pop_std / ranges)
                if norm_std < 1e-2 and generations_no_improvement >= 5:
                    # restart
                    best_idx = np.argmin(pop_fitness)
                    for i in range(popsize):
                        if i != best_idx:
                            pop[i] = lb + (ub - lb) * rng.rand(dim)
                            pop_fitness[i] = func(pop[i])
                            evals += 1
                            if evals >= budget:
                                break
                            if pop_fitness[i] < self.best_value:
                                self.best_value = pop_fitness[i]
                                self.best_x = pop[i].copy()
                                report_best(self.best_value, self.best_x)
                    if evals >= budget:
                        break
                    muF = np.ones(H) * 0.5
                    muCR = np.ones(H) * 0.5
                    k = 0
                    archive.clear()
                    generations_no_improvement = 0
                    continue
            improved = False
            for i in range(popsize):
                # generate F and CR
                F = np.clip(muF[k] + 0.1 * rng.randn(), 0.1, 0.9)
                CR = np.clip(muCR[k] + 0.1 * rng.randn(), 0.0, 1.0)
                # use current-to-best mutation
                idxs = list(range(popsize))
                idxs.remove(i)
                if len(idxs) >= 2:
                    r1, r2 = rng.choice(idxs, 2, replace=False)
                else:
                    continue
                mutant = pop[i] + F * (self.best_x - pop[i]) + F * (pop[r1] - pop[r2])
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < pop_fitness[i]:
                    # archive the replaced
                    if len(archive) < max_archive:
                        archive.append(pop[i].copy())
                    else:
                        archive[rng.randint(max_archive)] = pop[i].copy()
                    pop_fitness[i] = trial_fitness
                    pop[i] = trial
                    # update successful F and CR
                    muF[k] = (1 - 0.1) * muF[k] + 0.1 * F
                    muCR[k] = (1 - 0.1) * muCR[k] + 0.1 * CR
                    k = (k + 1) % H
                    improved = True
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                if evals >= budget:
                    break
            if evals >= budget:
                break
            if improved:
                generations_no_improvement = 0
            else:
                generations_no_improvement += 1
        return self.best_value, self.best_x