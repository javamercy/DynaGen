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
        # Latin Hypercube Sampling
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
        # DE parameters
        F = 0.5
        CR = 0.9
        # Local refinement
        sigma = 0.2 * (ub - lb)
        local_ref_interval = max(1, pop_size)
        gen_evals = 0
        # Stagnation detection
        stagnation_threshold = max(pop_size * 5, 1)
        eval_since_improvement = 0
        # Main loop
        while evals < self.budget:
            # DE step: one trial
            target_idx = self.rng.randint(pop_size)
            candidates = list(range(pop_size))
            candidates.remove(target_idx)
            if len(candidates) >= 3:
                idx = self.rng.choice(candidates, 3, replace=False)
                a, b, c = idx
                mutant = points[a] + F * (points[b] - points[c])
                trial = points[target_idx].copy()
                j_rand = self.rng.randint(self.dim)
                for j in range(self.dim):
                    if self.rng.rand() < CR or j == j_rand:
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
                        eval_since_improvement = 0
                        report_best(best_f, best_x)
                    else:
                        eval_since_improvement += 1
                else:
                    eval_since_improvement += 1
            else:
                # fallback if not enough distinct indices
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
                        eval_since_improvement = 0
                        report_best(best_f, best_x)
                    else:
                        eval_since_improvement += 1
                else:
                    eval_since_improvement += 1
            # Local refinement
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
                    sigma *= 1.1
                    eval_since_improvement = 0
                    report_best(best_f, best_x)
                else:
                    sigma *= 0.9
                    eval_since_improvement += 1
            # Stagnation restart
            if eval_since_improvement >= stagnation_threshold and evals < self.budget:
                # Replace half of worst individuals with LHS samples
                num_restart = max(1, pop_size // 2)
                # Find indices of worst individuals (excluding best)
                fitness_argsort = np.argsort(pop_fitness)
                worst_indices = fitness_argsort[-num_restart:]
                # Ensure best index not in worst_indices (if best is among worst, skip? but best should be best)
                best_idx = np.argmin(pop_fitness)
                if best_idx in worst_indices:
                    # Replace the second worst instead
                    worst_indices = np.setdiff1d(worst_indices, [best_idx])
                    if len(worst_indices) < num_restart:
                        # add the next worst
                        extra = fitness_argsort[-(num_restart+1)]
                        if extra not in worst_indices:
                            worst_indices = np.append(worst_indices, extra)
                # Generate new points for worst
                for idx in worst_indices:
                    if evals >= self.budget:
                        break
                    # LHS sample for this individual (simple random for efficiency)
                    x_new = lb + self.rng.rand(self.dim) * (ub - lb)
                    f_new = func(x_new)
                    evals += 1
                    points[idx] = x_new
                    pop_fitness[idx] = f_new
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                        eval_since_improvement = 0
                        report_best(best_f, best_x)
                # Reset sigma for local refinement
                sigma = 0.2 * (ub - lb)
                eval_since_improvement = 0
        return best_f, best_x