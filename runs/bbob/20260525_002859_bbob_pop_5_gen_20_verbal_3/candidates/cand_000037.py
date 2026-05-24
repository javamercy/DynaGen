import numpy as np
class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = float('inf')
        self.best_x = None

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        popsize = max(4, min(5*dim, int(budget*0.1), 30))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        fit = np.full(popsize, np.inf)
        evals = 0
        for i in range(popsize):
            fit[i] = func(pop[i])
            evals += 1
            if fit[i] < self.best_value:
                self.best_value = fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x
        mu_F = 0.8
        mu_CR = 0.9
        memory_F = np.full(5, 0.8)
        memory_CR = np.full(5, 0.9)
        memory_idx = 0
        no_improv_evals = 0
        stag_limit = max(1, budget // 5)
        while evals < budget:
            old_best = self.best_value
            S_F = []
            S_CR = []
            for i in range(popsize):
                if evals >= budget:
                    break
                candidates = list(range(popsize))
                candidates.remove(i)
                a, b = rng.choice(candidates, size=2, replace=False)
                # current-to-pbest: choose pbest from top 20%
                pbest_size = max(1, int(0.2*popsize))
                sorted_idx = np.argsort(fit)
                pbest_idx = sorted_idx[rng.randint(pbest_size)]
                F = mu_F + 0.1 * rng.randn()
                F = np.clip(F, 0.1, 0.9)
                CR = mu_CR + 0.1 * rng.randn()
                CR = np.clip(CR, 0, 1)
                mutant = pop[i] + F * (pop[pbest_idx] - pop[i]) + F * (pop[a] - pop[b])
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                j_rand = rng.randint(dim)
                trial[j_rand] = mutant[j_rand]
                trial = np.clip(trial, lb, ub)
                trial_fit = func(trial)
                evals += 1
                if trial_fit <= fit[i]:
                    fit[i] = trial_fit
                    pop[i] = trial
                    S_F.append(F)
                    S_CR.append(CR)
                    if trial_fit < self.best_value:
                        self.best_value = trial_fit
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
            if evals >= budget:
                break
            if len(S_F) > 0:
                # update memory
                mean_F = np.mean(S_F)
                mean_CR = np.mean(S_CR)
                memory_F[memory_idx] = mean_F
                memory_CR[memory_idx] = mean_CR
                memory_idx = (memory_idx + 1) % 5
                mu_F = np.mean(memory_F)
                mu_CR = np.mean(memory_CR)
            if self.best_value < old_best:
                no_improv_evals = 0
            else:
                no_improv_evals += popsize
            if no_improv_evals >= stag_limit and evals < budget - popsize:
                # restart: keep best individual, reinitialize others
                best_idx = np.argmin(fit)
                reinit_size = popsize - 1
                new_pop = lb + (ub - lb) * rng.rand(reinit_size, dim)
                pop = np.vstack((pop[best_idx:best_idx+1], new_pop))
                fit[0] = fit[best_idx]
                for i in range(1, popsize):
                    fit[i] = func(pop[i])
                    evals += 1
                    if fit[i] < self.best_value:
                        self.best_value = fit[i]
                        self.best_x = pop[i].copy()
                        report_best(self.best_value, self.best_x)
                if evals >= budget:
                    break
                no_improv_evals = 0
                # reset memory to default
                memory_F[:] = 0.8
                memory_CR[:] = 0.9
                mu_F = 0.8
                mu_CR = 0.9
                memory_idx = 0
        return self.best_value, self.best_x