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
        pop_size = min(self.budget, max(4, min(5 * self.dim, self.budget // 3)))

        # Latin Hypercube Sampling initialization
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

        # Differential Evolution parameters
        F = 0.5
        CR = 0.9

        # Local refinement parameters (intensified)
        local_freq = max(1, self.budget // 30)  # trigger more often
        step_size = 0.2 * (ub - lb)  # larger initial step for pattern search
        n_sweeps = 2  # number of full pattern search sweeps per trigger

        # Main DE loop
        while evals < self.budget:
            target_idx = self.rng.randint(pop_size)
            candidates = list(range(pop_size))
            candidates.remove(target_idx)
            if len(candidates) < 3:
                continue
            idx = self.rng.choice(candidates, 3, replace=False)
            a, b, c = idx
            # Mutation
            mutant = points[a] + F * (points[b] - points[c])
            # Crossover
            trial = points[target_idx].copy()
            j_rand = self.rng.randint(self.dim)
            for j in range(self.dim):
                if self.rng.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lb, ub)
            # Evaluate
            f_trial = func(trial)
            evals += 1
            if f_trial < pop_fitness[target_idx]:
                points[target_idx] = trial
                pop_fitness[target_idx] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
                    report_best(best_f, best_x)

            # Periodic local refinement (intensified pattern search)
            if evals % local_freq == 0 and evals < self.budget:
                for _ in range(n_sweeps):
                    if evals >= self.budget:
                        break
                    improved = False
                    for coord in range(self.dim):
                        if evals >= self.budget:
                            break
                        # Try positive step
                        pos = np.clip(best_x[coord] + step_size[coord], lb[coord], ub[coord])
                        x_pos = best_x.copy()
                        x_pos[coord] = pos
                        f_pos = func(x_pos)
                        evals += 1
                        if f_pos < best_f:
                            best_f = f_pos
                            best_x = x_pos
                            improved = True
                            report_best(best_f, best_x)
                            continue  # skip negative step if positive worked
                        # Try negative step
                        neg = np.clip(best_x[coord] - step_size[coord], lb[coord], ub[coord])
                        if neg == pos:
                            continue  # no step, avoid duplicate eval
                        x_neg = best_x.copy()
                        x_neg[coord] = neg
                        f_neg = func(x_neg)
                        evals += 1
                        if f_neg < best_f:
                            best_f = f_neg
                            best_x = x_neg
                            improved = True
                            report_best(best_f, best_x)
                    # Adapt step size
                    if improved:
                        step_size *= 1.5  # expand on success
                    else:
                        step_size *= 0.5  # contract on failure
                    # Keep step size within reasonable bounds
                    step_size = np.clip(step_size, 1e-6 * (ub - lb), 0.4 * (ub - lb))

        return best_f, best_x