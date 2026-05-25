import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        rng = self.rng
        budget = self.budget
        dim = self.dim
        pop_size = max(2, min(10, budget // 10))
        F_best = 0.9
        F_diff = 0.5
        CR = 0.9
        stag_limit = max(1, budget // 20)
        no_improve = 0
        # initial population
        pop = lb + (ub - lb) * rng.rand(pop_size, dim)
        pop_fit = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            pop_fit[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
                no_improve = 0
            else:
                no_improve += 1
        while evals < budget:
            # stagnation handling: local restart
            if no_improve >= stag_limit:
                # contract around best
                scale = 0.1 * (ub - lb)
                for i in range(pop_size):
                    if i == 0:
                        pop[i] = self.best_x.copy()
                    else:
                        pop[i] = self.best_x + rng.randn(dim) * scale
                        pop[i] = np.clip(pop[i], lb, ub)
                    if evals >= budget:
                        break
                    val = func(pop[i])
                    evals += 1
                    pop_fit[i] = val
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = pop[i].copy()
                        report_best(self.best_val, self.best_x)
                        no_improve = 0
                    else:
                        no_improve += 1
                no_improve = 0
                continue
            # main DE loop
            new_pop = pop.copy()
            for i in range(pop_size):
                if evals >= budget:
                    break
                idxs = [j for j in range(pop_size) if j != i]
                a, b = rng.choice(idxs, 2, replace=False)
                mutant = pop[i] + F_best * (self.best_x - pop[i]) + F_diff * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                cross = rng.rand(dim) < CR
                if not np.any(cross):
                    cross[rng.randint(dim)] = True
                trial = np.where(cross, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < self.best_val:
                    self.best_val = val
                    self.best_x = trial.copy()
                    report_best(self.best_val, self.best_x)
                    no_improve = 0
                else:
                    no_improve += 1
                if val < pop_fit[i]:
                    new_pop[i] = trial
                    pop_fit[i] = val
            pop = new_pop
        return self.best_val, self.best_x