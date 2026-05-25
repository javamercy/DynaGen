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
        pop_size = min(self.budget, max(4, min(5 * self.dim, self.budget // 3)))
        # Latin Hypercube Sampling initialization
        points = np.zeros((pop_size, self.dim))
        for i in range(self.dim):
            perm = self.rng.permutation(pop_size)
            u = self.rng.rand(pop_size)
            points[:, i] = lb[i] + (perm + u) / pop_size * (ub[i] - lb[i])
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
        # Parameters with initial values, will be updated dynamically
        F_min, F_max = 0.1, 0.9
        CR_min, CR_max = 0.1, 0.9
        local_ref_interval_max = pop_size
        local_ref_interval_min = 1
        sigma = 0.2 * (ub - lb)
        gen_evals = 0
        while evals < self.budget:
            # Compute current schedule based on remaining budget
            remaining = self.budget - evals
            if remaining <= 0:
                break
            progress = 1.0 - remaining / self.budget
            F_current = F_max - progress * (F_max - F_min)
            CR_current = CR_min + progress * (CR_max - CR_min)
            local_ref_interval = max(local_ref_interval_min, int(local_ref_interval_max * (1 - progress) + 0.5))
            # DE iteration
            target_idx = self.rng.randint(pop_size)
            candidates = list(range(pop_size))
            candidates.remove(target_idx)
            if len(candidates) >= 3:
                idx = self.rng.choice(candidates, 3, replace=False)
                a, b, c = idx
                mutant = points[a] + F_current * (points[b] - points[c])
                trial = points[target_idx].copy()
                j_rand = self.rng.randint(self.dim)
                for j in range(self.dim):
                    if self.rng.rand() < CR_current or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                gen_evals += 1
                if f_trial < pop_fitness[target_idx]:
                    points[target_idx] = trial
                    pop_fitness[target_idx] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
            else:
                if evals >= self.budget:
                    break
                x = lb + self.rng.rand(self.dim) * (ub - lb)
                f = func(x)
                evals += 1
                gen_evals += 1
                worst_idx = np.argmax(pop_fitness)
                if f < pop_fitness[worst_idx]:
                    points[worst_idx] = x
                    pop_fitness[worst_idx] = f
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
            # Local refinement with adaptive interval
            if gen_evals >= local_ref_interval and evals < self.budget:
                gen_evals = 0
                delta = sigma * self.rng.randn(self.dim)
                candidate = best_x + delta
                candidate = np.clip(candidate, lb, ub)
                f_candidate = func(candidate)
                evals += 1
                if f_candidate < best_f:
                    best_f = f_candidate
                    best_x = candidate.copy()
                    sigma *= 1.2
                    report_best(best_f, best_x)
                else:
                    sigma *= 0.8
        return best_f, best_x