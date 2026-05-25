import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        np.random.seed(seed)
        self.NP = max(4, min(int(budget/2), 10*dim))
        self.tau_F = 0.1
        self.tau_CR = 0.1
        self.F_l = 0.1
        self.F_u = 0.9
        self.CR_l = 0.0
        self.CR_u = 1.0
        self.best_x = None
        self.best_val = float('inf')
        self.calls = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.NP, self.dim))
        fitness = np.full(self.NP, float('inf'))
        for i in range(self.NP):
            val = func(pop[i])
            self.calls += 1
            fitness[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        F = np.random.uniform(self.F_l, self.F_u, self.NP)
        CR = np.random.uniform(self.CR_l, self.CR_u, self.NP)
        no_improve_gens = 0
        stagnation_limit = 10
        max_restarts = 3
        restart_count = 0

        while self.calls < self.budget:
            best_prev = self.best_val
            for i in range(self.NP):
                if self.calls >= self.budget:
                    break
                if np.random.rand() < self.tau_F:
                    F[i] = np.random.uniform(self.F_l, self.F_u)
                if np.random.rand() < self.tau_CR:
                    CR[i] = np.random.uniform(self.CR_l, self.CR_u)
                candidates = list(range(self.NP))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                # choose base: best or random with prob 0.1
                if np.random.rand() < 0.1 and self.NP > 1:
                    base_idx = np.random.randint(self.NP)
                    base = pop[base_idx]
                else:
                    base = self.best_x
                mutant = base + F[i] * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < CR[i], mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                self.calls += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = trial.copy()
                        report_best(self.best_val, self.best_x)
            # after generation, check stagnation
            if self.best_val >= best_prev:
                no_improve_gens += 1
            else:
                no_improve_gens = 0
            if no_improve_gens >= stagnation_limit and restart_count < max_restarts:
                # reinitialize worst 20% of population
                reinit_count = max(1, int(0.2 * self.NP))
                if self.calls + reinit_count <= self.budget:
                    worst_indices = np.argsort(fitness)[-reinit_count:]
                    for idx in worst_indices:
                        pop[idx] = np.random.uniform(lb, ub)
                        val = func(pop[idx])
                        self.calls += 1
                        fitness[idx] = val
                        F[idx] = np.random.uniform(self.F_l, self.F_u)
                        CR[idx] = np.random.uniform(self.CR_l, self.CR_u)
                        if val < self.best_val:
                            self.best_val = val
                            self.best_x = pop[idx].copy()
                            report_best(self.best_val, self.best_x)
                    restart_count += 1
                    no_improve_gens = 0
        return self.best_val, self.best_x