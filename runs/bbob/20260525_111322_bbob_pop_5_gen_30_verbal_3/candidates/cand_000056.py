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
        dim = self.dim
        pop_size = max(4, min(20, int(2 * dim)))
        # LHS initialization
        points = np.zeros((pop_size, dim))
        for i in range(dim):
            perm = self.rng.permutation(pop_size)
            u = self.rng.rand(pop_size)
            points[:, i] = lb[i] + (perm + u) / pop_size * (ub[i] - lb[i])
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0
        for i in range(pop_size):
            if evals >= self.budget:
                break
            x = points[i]
            f = func(x)
            evals += 1
            fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
        # DE parameters
        F = 0.8
        CR = 0.9
        # Local search parameters
        sigma = 0.2 * (ub - lb)
        local_step_size = 1.0
        local_num = 3
        gen_no_improve = 0
        max_no_improve = max(10, self.budget // 20)
        while evals < self.budget:
            # DE generation
            for target_idx in range(pop_size):
                if evals >= self.budget:
                    break
                candidates = [i for i in range(pop_size) if i != target_idx]
                if len(candidates) < 2:
                    continue
                r1, r2 = self.rng.choice(candidates, 2, replace=False)
                mutant = points[target_idx] + F * (best_x - points[target_idx]) + F * (points[r1] - points[r2])
                trial = points[target_idx].copy()
                j_rand = self.rng.randint(dim)
                for j in range(dim):
                    if self.rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[target_idx]:
                    points[target_idx] = trial
                    fitness[target_idx] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        gen_no_improve = 0
                    else:
                        gen_no_improve += 1
                else:
                    gen_no_improve += 1
            # Local search around best
            for _ in range(local_num):
                if evals >= self.budget:
                    break
                delta = sigma * local_step_size * self.rng.randn(dim)
                candidate = best_x + delta
                candidate = np.clip(candidate, lb, ub)
                f_candidate = func(candidate)
                evals += 1
                if f_candidate < best_f:
                    best_f = f_candidate
                    best_x = candidate.copy()
                    report_best(best_f, best_x)
                    local_step_size *= 1.1
                    gen_no_improve = 0
                else:
                    local_step_size *= 0.9
                    gen_no_improve += 1
            # Restart if stagnation
            if gen_no_improve >= max_no_improve and evals < self.budget:
                # Zoom around best
                shrink_factor = 0.5
                new_lb = np.maximum(lb, best_x - shrink_factor * (ub - lb))
                new_ub = np.minimum(ub, best_x + shrink_factor * (ub - lb))
                new_points = np.zeros((pop_size, dim))
                for i in range(dim):
                    perm = self.rng.permutation(pop_size)
                    u = self.rng.rand(pop_size)
                    new_points[:, i] = new_lb[i] + (perm + u) / pop_size * (new_ub[i] - new_lb[i])
                new_fitness = np.full(pop_size, np.inf)
                for i in range(pop_size):
                    if evals >= self.budget:
                        break
                    x = new_points[i]
                    f = func(x)
                    evals += 1
                    new_fitness[i] = f
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                # Replace population, keeping best
                points = new_points
                fitness = new_fitness
                # Ensure best is in population
                worst_idx = np.argmax(fitness)
                if best_f < fitness[worst_idx]:
                    points[worst_idx] = best_x.copy()
                    fitness[worst_idx] = best_f
                gen_no_improve = 0
        return best_f, best_x