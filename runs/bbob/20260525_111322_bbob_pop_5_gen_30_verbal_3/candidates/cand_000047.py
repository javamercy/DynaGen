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
        # population size
        N = min(self.budget, max(4, int(min(5 * dim, self.budget // 3))))
        # Latin Hypercube Sampling
        points = np.zeros((N, dim))
        for i in range(dim):
            perm = self.rng.permutation(N)
            u = self.rng.rand(N)
            points[:, i] = lb[i] + (perm + u) / N * (ub[i] - lb[i])
        fitness = np.full(N, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0
        for i in range(N):
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
        F = 0.5
        CR = 0.9
        # CMA parameters
        sigma = 0.3 * np.max(ub - lb)
        C = np.eye(dim)
        p_c = np.zeros(dim)
        cc = 0.2
        c1 = 1.0 / dim
        L = np.linalg.cholesky(C)  # lower triangular
        improvement = False
        gen = 0
        while evals < self.budget:
            # Shuffle order for each generation
            order = self.rng.permutation(N)
            best_step = np.zeros(dim)
            best_improved = False
            for idx in order:
                if evals >= self.budget:
                    break
                # Select three distinct individuals
                candidates = list(range(N))
                candidates.remove(idx)
                if len(candidates) < 3:
                    continue
                a, b, c = self.rng.choice(candidates, 3, replace=False)
                # Mutation: rotate difference by C
                diff = points[b] - points[c]
                rotated = F * L.dot(diff)
                mutant = points[a] + rotated
                # Crossover
                trial = points[idx].copy()
                j_rand = self.rng.randint(dim)
                for j in range(dim):
                    if self.rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                # Clip
                trial = np.clip(trial, lb, ub)
                # Evaluate
                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[idx]:
                    step = trial - points[idx]
                    points[idx] = trial
                    fitness[idx] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        best_step = step.copy()
                        best_improved = True
            # Update CMA after generation (only if best improved)
            if best_improved:
                # Evolution path update
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc)) * best_step / sigma
                # Rank-one update
                C = (1 - c1) * C + c1 * np.outer(p_c, p_c)
                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                # Adjust step size (success-based)
                sigma *= 1.1
            else:
                sigma *= 0.9
            sigma = np.clip(sigma, 1e-8 * np.max(ub-lb), 0.5 * np.max(ub-lb))
            # Recompute Cholesky
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                # Fallback to identity
                L = np.eye(dim)
            gen += 1
        return best_f, best_x