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
        pop_size = max(4, min(5 * self.dim, self.budget // 3))
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
        CR = 0.9
        sigma = 0.2 * (ub - lb)
        stagnation_counter = 0
        max_stagnation = max(5, self.budget // 20)  # lower threshold for restart
        # local refinement parameters
        local_sigma = 0.1 * (ub - lb)
        while evals < self.budget:
            # DE iteration
            target_idx = self.rng.randint(pop_size)
            candidates = list(range(pop_size))
            candidates.remove(target_idx)
            if len(candidates) >= 2:
                # Bias towards best: use current-to-best/1 with high probability
                if self.rng.rand() < 0.7:
                    # DE/current-to-best/1
                    a, b = self.rng.choice(candidates, 2, replace=False)
                    F = 0.5 + self.rng.rand() * 0.5
                    K = self.rng.rand()
                    mutant = points[target_idx] + K * (best_x - points[target_idx]) + F * (points[a] - points[b])
                else:
                    # DE/rand/1
                    a, b, c = self.rng.choice(candidates, 3, replace=False)
                    F = 0.5 + self.rng.rand() * 0.5
                    mutant = points[a] + F * (points[b] - points[c])
                trial = points[target_idx].copy()
                j_rand = self.rng.randint(self.dim)
                for j in range(self.dim):
                    if self.rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial < pop_fitness[target_idx]:
                    points[target_idx] = trial
                    pop_fitness[target_idx] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1
            else:
                # fallback random point
                if evals >= self.budget:
                    break
                x = lb + self.rng.rand(self.dim) * (ub - lb)
                f = func(x)
                evals += 1
                worst_idx = np.argmax(pop_fitness)
                if f < pop_fitness[worst_idx]:
                    points[worst_idx] = x
                    pop_fitness[worst_idx] = f
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                        stagnation_counter = 0
                else:
                    stagnation_counter += 1
            # Local refinement: run after each DE iteration
            if evals < self.budget:
                # Try multiple local steps
                for _ in range(2):
                    if evals >= self.budget:
                        break
                    delta = local_sigma * self.rng.randn(self.dim)
                    candidate = best_x + delta
                    candidate = np.clip(candidate, lb, ub)
                    f_candidate = func(candidate)
                    evals += 1
                    if f_candidate < best_f:
                        best_f = f_candidate
                        best_x = candidate.copy()
                        local_sigma *= 1.1
                        report_best(best_f, best_x)
                        stagnation_counter = 0
                    else:
                        local_sigma *= 0.9
                        stagnation_counter += 1
            # Restart if stagnation
            if stagnation_counter >= max_stagnation and evals < self.budget:
                # Replace population except best with new LHS sample
                new_pop_size = pop_size - 1
                new_points = np.zeros((new_pop_size, self.dim))
                for i in range(self.dim):
                    perm = self.rng.permutation(new_pop_size)
                    u = self.rng.rand(new_pop_size)
                    new_points[:, i] = lb[i] + (perm + u) / new_pop_size * (ub[i] - lb[i])
                for i in range(new_pop_size):
                    if evals >= self.budget:
                        break
                    x = new_points[i]
                    f = func(x)
                    evals += 1
                    worst_idx = np.argmax(pop_fitness)
                    if f < pop_fitness[worst_idx]:
                        points[worst_idx] = x
                        pop_fitness[worst_idx] = f
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                            report_best(best_f, best_x)
                # Ensure best remains
                worst_idx = np.argmax(pop_fitness)
                points[worst_idx] = best_x.copy()
                pop_fitness[worst_idx] = best_f
                stagnation_counter = 0
                local_sigma = 0.1 * (ub - lb)  # reset local sigma
        return best_f, best_x