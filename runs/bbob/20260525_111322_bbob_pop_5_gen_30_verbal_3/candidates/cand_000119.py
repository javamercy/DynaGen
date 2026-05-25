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
        # Population size
        pop_size = max(4, min(5 * self.dim, self.budget // 3))
        # LHS initialization
        points = np.zeros((pop_size, self.dim))
        for i in range(self.dim):
            perm = self.rng.permutation(pop_size)
            u = self.rng.rand(pop_size)
            points[:, i] = lb[i] + (perm + u) / pop_size * (ub[i] - lb[i])
        # Evaluate initial population
        pop_fitness = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0
        for i in range(pop_size):
            if evals >= self.budget:
                break
            x = points[i]
            f = func(x)
            evals += 1
            pop_fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
        # jDE parameters
        F_l = 0.1
        F_u = 0.9
        tau1 = 0.1
        tau2 = 0.1
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        # Restart parameters
        stall_counter = 0
        stall_limit = max(100, 2 * pop_size * self.dim)
        best_evals = evals
        # Main loop
        while evals < self.budget:
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # Mutation
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = self.rng.choice(candidates, 3, replace=False)
                mutant = points[a] + F[i] * (points[b] - points[c])
                # Crossover
                j_rand = self.rng.randint(self.dim)
                trial = points[i].copy()
                for j in range(self.dim):
                    if self.rng.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial < pop_fitness[i]:
                    points[i] = trial
                    pop_fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        stall_counter = 0
                    # Update F and CR (jDE)
                    if self.rng.rand() < tau1:
                        F[i] = F_l + self.rng.rand() * (F_u - F_l)
                    if self.rng.rand() < tau2:
                        CR[i] = self.rng.rand()
                else:
                    stall_counter += 1
            # Restart if stalled
            if evals - best_evals >= stall_limit and evals < self.budget:
                # Reinitialize population, keep best
                new_pop = np.zeros((pop_size, self.dim))
                for i in range(self.dim):
                    perm = self.rng.permutation(pop_size - 1) + 1
                    u = self.rng.rand(pop_size - 1)
                    new_pop[1:, i] = lb[i] + (perm + u) / pop_size * (ub[i] - lb[i])
                new_pop[0] = best_x.copy()
                points = new_pop
                # Reset F and CR
                F = np.full(pop_size, 0.5)
                CR = np.full(pop_size, 0.9)
                # Evaluate new population except best
                for i in range(1, pop_size):
                    if evals >= self.budget:
                        break
                    x = points[i]
                    f = func(x)
                    evals += 1
                    pop_fitness[i] = f
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                pop_fitness[0] = best_f
                stall_counter = 0
                best_evals = evals
        return best_f, best_x