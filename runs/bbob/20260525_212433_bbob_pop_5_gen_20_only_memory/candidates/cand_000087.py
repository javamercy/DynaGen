import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        budget = self.budget
        rng = self.rng
        D = self.dim

        if budget < 3:
            best_val = float('inf')
            best_x = None
            while evals < budget:
                x = lb + (ub - lb) * rng.rand(D)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        N = max(4, min(budget // 20, 20))
        if N > budget:
            N = budget

        pop = lb + (ub - lb) * rng.rand(N, D)
        pop_fit = np.full(N, np.inf)
        best_val = float('inf')
        best_x = None

        for i in range(N):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            pop_fit[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        F = 0.5 * rng.rand(N) + 0.1
        CR = rng.rand(N)
        tau1 = 0.1
        tau2 = 0.1
        no_improve = 0
        restart_threshold = max(0.1 * budget, 10)

        while evals < budget:
            if no_improve >= restart_threshold and evals + N <= budget:
                for i in range(N):
                    if i == np.argmin(pop_fit):
                        continue
                    pop[i] = lb + (ub - lb) * rng.rand(D)
                    val = func(pop[i])
                    evals += 1
                    pop_fit[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                    F[i] = 0.5 * rng.rand() + 0.1
                    CR[i] = rng.rand()
                no_improve = 0
                continue

            new_pop = pop.copy()
            new_fit = pop_fit.copy()
            best_idx = np.argmin(pop_fit)
            for i in range(N):
                if evals >= budget:
                    break
                if rng.rand() < tau1:
                    Fi = 0.1 + 0.9 * rng.rand()
                else:
                    Fi = F[i]
                if rng.rand() < tau2:
                    CRi = rng.rand()
                else:
                    CRi = CR[i]

                idxs = [j for j in range(N) if j != i]
                a, b = rng.choice(idxs, 2, replace=False)
                mutant = pop[i] + Fi * (pop[best_idx] - pop[i]) + Fi * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                cross_points = rng.rand(D) < CRi
                if not np.any(cross_points):
                    cross_points[rng.randint(D)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                    no_improve = 0
                else:
                    no_improve += 1
                if val < pop_fit[i]:
                    new_pop[i] = trial
                    new_fit[i] = val
                    F[i] = Fi
                    CR[i] = CRi
            pop = new_pop
            pop_fit = new_fit

        return best_val, best_x